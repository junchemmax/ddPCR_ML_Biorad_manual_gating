import csv
import glob
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def parse_well_name(csv_path: str) -> str:
    name = os.path.basename(csv_path)
    stem = name.replace("_Amplitude.csv", "")
    match = re.search(r"([A-H](?:0[1-9]|1[0-2]))$", stem)
    if match:
        return match.group(1)
    parts = stem.split("_")
    return parts[-1] if parts else stem


def find_dataset_dirs(root_dir: str) -> list[str]:
    required_cluster = "*_ClusterData.csv"
    required_amplitude = "*_Amplitude.csv"
    dataset_dirs = []
    for current_dir, _, filenames in os.walk(root_dir):
        if any(name.endswith("_ClusterData.csv") for name in filenames) and any(
            name.endswith("_Amplitude.csv") for name in filenames
        ):
            dataset_dirs.append(current_dir)
    return sorted(dataset_dirs)


def mip_output_dir(data_dir: str) -> str:
    current_dir = os.path.abspath(data_dir)
    while True:
        if os.path.basename(current_dir).lower().startswith("mip_"):
            return os.path.join(current_dir, "Biorad_manual_gating_output")
        parent_dir = os.path.dirname(current_dir)
        if parent_dir == current_dir:
            return os.path.join(data_dir, "Biorad_manual_gating_output")
        current_dir = parent_dir


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python draw_cluster_gates.py <dataset_folder_or_ddPCR_data_root>")
        sys.exit(1)

    root_dir = os.path.abspath(sys.argv[1])
    dataset_dirs = find_dataset_dirs(root_dir)
    if not dataset_dirs:
        print(f"No dataset folders containing amplitude and ClusterData CSVs found in: {root_dir}")
        sys.exit(1)

    flowsom_output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(flowsom_output_dir, exist_ok=True)
    all_data_csv = os.path.join(flowsom_output_dir, "biorad_cluster_gate_data.csv")
    csv_fields = [
        "dataset", "well", "source_amplitude_csv", "x_gate", "y_gate",
        "x_min_ch2", "x_max_ch2", "y_min_ch1", "y_max_ch1", "target_1",
        "target_2", "cluster_id", "count", "ch1_mean", "ch2_mean",
    ]

    with open(all_data_csv, "w", encoding="utf-8", newline="") as all_data_fh:
        all_data_writer = csv.DictWriter(all_data_fh, fieldnames=csv_fields)
        all_data_writer.writeheader()
        for data_dir in dataset_dirs:
            process_dataset(data_dir, all_data_writer)

    print(f"All x/y data saved to: {all_data_csv}")


def process_dataset(data_dir: str, all_data_writer: csv.DictWriter) -> None:
    cluster_files = sorted(glob.glob(os.path.join(data_dir, "*_ClusterData.csv")))
    if not cluster_files:
        print(f"No *_ClusterData.csv files found in: {data_dir}")
        sys.exit(1)

    cluster_csv = cluster_files[0]
    with open(cluster_csv, "r", encoding="utf-8", errors="ignore", newline="") as fh:
        reader = csv.reader(fh)
        header = None
        rows = []
        for row in reader:
            if not row:
                continue
            cleaned = [cell.strip() for cell in row]
            if not any(cleaned):
                continue

            first = cleaned[0].lower()
            second = cleaned[1].lower() if len(cleaned) > 1 else ""
            if first == "well" and second == "target 1":
                header = cleaned[:]
                continue
            if header is not None:
                if first == "well" and second == "cluster 1":
                    break
                row_values = cleaned[:len(header)]
                if len(row_values) < len(header):
                    row_values.extend([""] * (len(header) - len(row_values)))
                rows.append(row_values)

    if header is None:
        raise ValueError(f"Could not find the target-gating ClusterData section in: {cluster_csv}")

    cluster_df = pd.DataFrame(rows, columns=header)
    cluster_df["Well"] = cluster_df["Well"].astype(str).str.strip()

    numeric_cols = ["Count", "Ch1 Mean", "Ch1 StdDev", "Ch2 Mean", "Ch2 StdDev"]
    for col in numeric_cols:
        if col in cluster_df.columns:
            cluster_df[col] = pd.to_numeric(cluster_df[col].astype(str).str.replace(",", "", regex=False), errors="coerce")

    # Normalize target columns to 0/1 while keeping well rows that may contain blank cells.
    for target_col in ["Target 1", "Target 2"]:
        if target_col in cluster_df.columns:
            cluster_df[target_col] = cluster_df[target_col].astype(str).str.strip()

    amplitude_csvs = sorted(glob.glob(os.path.join(data_dir, "*_Amplitude.csv")))
    if not amplitude_csvs:
        print(f"No *_Amplitude.csv files found in: {data_dir}")
        sys.exit(1)

    out_dir = mip_output_dir(data_dir)
    os.makedirs(out_dir, exist_ok=True)

    dataset_name = os.path.basename(os.path.abspath(data_dir))

    for amp_csv in amplitude_csvs:
        well = parse_well_name(amp_csv)
        well_rows = cluster_df[cluster_df["Well"] == well].copy()
        if well_rows.empty:
            print(f"No ClusterData rows for well {well} in {cluster_csv}; skipping.")
            continue

        amp_df = pd.read_csv(amp_csv, skiprows=3, low_memory=False)
        ch1_col = next((c for c in amp_df.columns if "Ch1" in str(c)), None)
        ch2_col = next((c for c in amp_df.columns if "Ch2" in str(c)), None)
        if ch1_col is None or ch2_col is None:
            print(f"Skipping {amp_csv}: amplitude columns not found.")
            continue

        amp_df = amp_df[[ch1_col, ch2_col]].dropna().copy()
        if amp_df.empty:
            print(f"Skipping {amp_csv}: no amplitude points available.")
            continue

        # Use the biological orientation requested by the user:
        #   Ch2 on x-axis, Ch1 on y-axis.
        # The best gate is the midpoint between the negative and positive mean values
        # of the corresponding target, i.e. target2 separates x and target1 separates y.
        target1_pos = well_rows[well_rows["Target 1"].astype(str).str.strip().isin(["1", "1.0"])].copy()
        target1_neg = well_rows[well_rows["Target 1"].astype(str).str.strip().isin(["0", "0.0"])].copy()
        target2_pos = well_rows[well_rows["Target 2"].astype(str).str.strip().isin(["1", "1.0"])].copy()
        target2_neg = well_rows[well_rows["Target 2"].astype(str).str.strip().isin(["0", "0.0"])].copy()

        y_gate = None
        x_gate = None

        if not target1_pos.empty and not target1_neg.empty:
            y_neg_mean = float(target1_neg["Ch1 Mean"].mean())
            y_pos_mean = float(target1_pos["Ch1 Mean"].mean())
            y_gate = 0.5 * (y_neg_mean + y_pos_mean)
        if not target2_pos.empty and not target2_neg.empty:
            x_neg_mean = float(target2_neg["Ch2 Mean"].mean())
            x_pos_mean = float(target2_pos["Ch2 Mean"].mean())
            x_gate = 0.5 * (x_neg_mean + x_pos_mean)

        if x_gate is None and y_gate is None:
            print(f"Plotting {well} without gate lines: no valid target means found in ClusterData.")

        all_data_writer.writerow({
            "dataset": dataset_name,
            "well": well,
            "source_amplitude_csv": os.path.basename(amp_csv),
            "x_gate": x_gate,
            "y_gate": y_gate,
            "x_min_ch2": amp_df[ch2_col].min(),
            "x_max_ch2": amp_df[ch2_col].max(),
            "y_min_ch1": amp_df[ch1_col].min(),
            "y_max_ch1": amp_df[ch1_col].max(),
            "target_1": "|".join(well_rows["Target 1"].dropna().astype(str).unique()),
            "target_2": "|".join(well_rows["Target 2"].dropna().astype(str).unique()),
            "cluster_id": "|".join(well_rows["Cluster ID"].dropna().astype(str).unique()),
            "count": "|".join(well_rows["Count"].dropna().astype(str).unique()),
            "ch1_mean": "|".join(well_rows["Ch1 Mean"].dropna().astype(str).unique()),
            "ch2_mean": "|".join(well_rows["Ch2 Mean"].dropna().astype(str).unique()),
        })

        plot_step = max(1, len(amp_df) // 10000)
        plot_df = amp_df.iloc[::plot_step]
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(
            plot_df[ch2_col].to_numpy(),
            plot_df[ch1_col].to_numpy(),
            s=1,
            alpha=0.45,
            color="gray",
            linewidths=0,
            rasterized=True,
        )

        if x_gate is not None:
            ax.axvline(x_gate, color="tab:blue", linestyle="--", linewidth=2, alpha=0.9)
        if y_gate is not None:
            ax.axhline(y_gate, color="tab:orange", linestyle="--", linewidth=2, alpha=0.9)

        # Show the mean of each quadrant as a marker, useful for validation.
        for _, row in well_rows.iterrows():
            try:
                x_mean = float(row["Ch2 Mean"])
                y_mean = float(row["Ch1 Mean"])
                ax.scatter([x_mean], [y_mean], s=12, edgecolors="black", linewidth=0.4, zorder=3)
            except (TypeError, ValueError):
                pass

        ax.set_xlabel("Ch2Amplitude")
        ax.set_ylabel("Ch1Amplitude")
        ax.set_title(f"{well} best gate from ClusterData (Ch2 x, Ch1 y)")
        if x_gate is not None and y_gate is not None:
            ax.text(
                0.02, 0.98,
                f"x gate={x_gate:.1f}\ny gate={y_gate:.1f}",
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=9,
                bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8}
            )
        ax.grid(False)
        fig.tight_layout()

        out_png_name = f"{dataset_name}_{well}_gated_cluster_lines.png"
        out_png = os.path.join(out_dir, out_png_name)
        try:
            fig.savefig(out_png, dpi=100, bbox_inches="tight")
            print(f"Saved gated plot -> {out_png}")
        except Exception as exc:
            print(f"Skipping PNG for {well}: {exc}")
        finally:
            plt.close(fig)

    print(f"Done. PNG files saved under: {out_dir}")


if __name__ == "__main__":
    main()
