import argparse
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.() -]+", "_", value)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Draw manual and predicted Bio-Rad gates together for each well."
    )
    parser.add_argument("--predictions-path", default="output/biorad_gate_model_predictions.csv")
    parser.add_argument("--amplitude-root", default="ddPCR_data")
    parser.add_argument("--output-dir", default="output/manual_vs_predicted_gates")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    predictions = pd.read_csv(args.predictions_path)
    required_columns = [
        "dataset", "x_gate", "y_gate", "predicted_x_gate", "predicted_y_gate",
    ]
    missing_columns = [column for column in required_columns if column not in predictions]
    if missing_columns:
        raise ValueError(f"Required prediction columns are missing: {', '.join(missing_columns)}")

    amplitude_paths = {
        os.path.splitext(os.path.basename(path))[0]: path
        for path in glob.glob(
            os.path.join(os.path.abspath(args.amplitude_root), "**", "*_Amplitude.csv"),
            recursive=True,
        )
    }
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    rows = predictions if args.limit is None else predictions.head(args.limit)
    created = 0
    skipped = 0
    for _, row in rows.iterrows():
        dataset = str(row["dataset"])
        amplitude_path = amplitude_paths.get(dataset)
        if amplitude_path is None:
            print(f"Skipping {dataset}: amplitude CSV not found.")
            skipped += 1
            continue

        amplitude = pd.read_csv(amplitude_path, skiprows=3, low_memory=False)
        amplitude = amplitude[["Ch1Amplitude", "Ch2Amplitude"]].dropna()
        if amplitude.empty:
            skipped += 1
            continue

        plot_step = max(1, len(amplitude) // 15000)
        plot_data = amplitude.iloc[::plot_step]
        figure, axis = plt.subplots(figsize=(7, 6))
        axis.scatter(
            plot_data["Ch2Amplitude"],
            plot_data["Ch1Amplitude"],
            s=1,
            alpha=0.35,
            color="0.45",
            linewidths=0,
            rasterized=True,
        )

        axis.axvline(
            float(row["x_gate"]), color="tab:blue", linewidth=2,
            label="Manual x gate",
        )
        axis.axhline(
            float(row["y_gate"]), color="tab:orange", linewidth=2,
            label="Manual y gate",
        )
        axis.axvline(
            float(row["predicted_x_gate"]), color="tab:blue", linestyle="--",
            linewidth=2, label="Predicted x gate",
        )
        axis.axhline(
            float(row["predicted_y_gate"]), color="tab:orange", linestyle="--",
            linewidth=2, label="Predicted y gate",
        )

        axis.set_xlabel("Ch2Amplitude")
        axis.set_ylabel("Ch1Amplitude")
        axis.set_title(f"{dataset}: manual vs predicted gates")
        axis.legend(loc="best", fontsize=8)
        axis.grid(False)
        figure.tight_layout()

        output_path = os.path.join(
            output_dir,
            f"{safe_filename(dataset)}_manual_vs_predicted.png",
        )
        figure.savefig(output_path, dpi=100, bbox_inches="tight")
        plt.close(figure)
        created += 1

    print(f"Saved plots: {created}")
    print(f"Skipped wells: {skipped}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()