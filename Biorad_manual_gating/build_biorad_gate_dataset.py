import csv
import argparse
import glob
import os
import re
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.stats import kurtosis, skew


warnings.filterwarnings("ignore", category=RuntimeWarning)


def parse_pipe_numbers(value: object) -> list[float]:
    if pd.isna(value) or str(value).strip() == "":
        return []
    numbers = []
    for item in str(value).split("|"):
        try:
            numbers.append(float(item))
        except ValueError:
            continue
    return numbers


def weighted_mean(values: list[float], weights: list[float]) -> float:
    if not values or len(values) != len(weights) or sum(weights) == 0:
        return np.nan
    return float(np.average(values, weights=weights))


def add_gate_features(gate_df: pd.DataFrame) -> pd.DataFrame:
    gate_df = gate_df.copy()
    for column in ["x_gate", "y_gate", "x_min_ch2", "x_max_ch2", "y_min_ch1", "y_max_ch1"]:
        gate_df[column] = pd.to_numeric(gate_df[column], errors="coerce")
    derived = gate_df.apply(
        lambda row: pd.Series(
            {
                "gate_has_x": int(pd.notna(row["x_gate"])),
                "gate_has_y": int(pd.notna(row["y_gate"])),
                "gate_n_cluster_rows": (
                    0 if pd.isna(row["quadrant"])
                    else sum(bool(value.strip()) for value in str(row["quadrant"]).split("|"))
                ),
                "gate_total_cluster_count": sum(parse_pipe_numbers(row["count"])),
                "gate_ch1_mean_weighted": weighted_mean(
                    parse_pipe_numbers(row["ch1_mean"]),
                    parse_pipe_numbers(row["count"]),
                ),
                "gate_ch2_mean_weighted": weighted_mean(
                    parse_pipe_numbers(row["ch2_mean"]),
                    parse_pipe_numbers(row["count"]),
                ),
                "gate_ch2_range": row["x_max_ch2"] - row["x_min_ch2"],
                "gate_ch1_range": row["y_max_ch1"] - row["y_min_ch1"],
            }
        ),
        axis=1,
    )
    return pd.concat([gate_df, derived], axis=1)


def write_missing_gate_report(
    data: pd.DataFrame,
    output_dir: str,
    unprocessed_rows: list[dict[str, object]],
) -> tuple[str, int, int]:
    missing_axes = data[["x_gate", "y_gate"]].isna()
    missing_mask = missing_axes.any(axis=1)
    report = data.loc[missing_mask].copy()
    report["missing_x_gate"] = missing_axes.loc[missing_mask, "x_gate"].to_numpy()
    report["missing_y_gate"] = missing_axes.loc[missing_mask, "y_gate"].to_numpy()
    report["source_amplitude_csv"] = report["dataset"].astype(str) + ".csv"
    report["processing_status"] = "processed"
    report["reason"] = report.apply(
        lambda row: "; ".join(
            reason
            for missing, reason in [
                (row["missing_x_gate"], "x gate unavailable"),
                (row["missing_y_gate"], "y gate unavailable"),
            ]
            if missing
        ),
        axis=1,
    )
    if unprocessed_rows:
        report = pd.concat(
            [report, pd.DataFrame(unprocessed_rows)],
            ignore_index=True,
            sort=False,
        )
    report_path = os.path.join(output_dir, "wells_with_missing_gates.csv")
    report.to_csv(report_path, index=False)
    return report_path, int(missing_mask.sum()), int(missing_axes.all(axis=1).sum())


def kde_peak_count(values: np.ndarray) -> int:
    if len(values) < 3 or np.ptp(values) == 0:
        return 1
    try:
        counts, _ = np.histogram(values, bins=128)
        if counts.max() == 0:
            return 1
        peaks, _ = find_peaks(counts, prominence=counts.max() * 0.05)
        return max(1, int(len(peaks)))
    except ValueError:
        return 1


def extract_features(data: np.ndarray) -> dict[str, float]:
    features: dict[str, float] = {"n_droplets": int(len(data))}
    for index, channel in enumerate(["ch1", "ch2"]):
        values = data[:, index]
        mean = float(np.mean(values))
        std = float(np.std(values))
        features[f"{channel}_std"] = std
        features[f"{channel}_iqr"] = float(np.percentile(values, 75) - np.percentile(values, 25))
        features[f"{channel}_cv"] = std / (mean + 1e-9)
        features[f"{channel}_skew"] = float(skew(values))
        features[f"{channel}_kurt"] = float(kurtosis(values))
        features[f"{channel}_bimodality"] = (features[f"{channel}_skew"] ** 2 + 1) / (features[f"{channel}_kurt"] + 3 + 1e-9)
        log_values = np.log1p(np.clip(values, 0, None))
        features[f"{channel}_log_std"] = float(np.std(log_values))
        features[f"{channel}_log_skew"] = float(skew(log_values))
        features[f"{channel}_outlier_frac"] = float(
            np.mean((values < mean - 3 * std) | (values > mean + 3 * std))
        )
        features[f"{channel}_kde_peaks"] = kde_peak_count(values)
    features["ch1_ch2_corr"] = float(np.corrcoef(data[:, 0], data[:, 1])[0, 1])
    return features


def parse_well_name(csv_path: str) -> str:
    name = os.path.basename(csv_path)
    stem = name.replace("_Amplitude.csv", "")
    match = re.search(r"([A-H](?:0[1-9]|1[0-2]))$", stem)
    if match:
        return match.group(1)
    parts = stem.split("_")
    return parts[-1] if parts else stem


def find_dataset_dirs(root_dir: str) -> list[str]:
    dataset_dirs = []
    for current_dir, _, filenames in os.walk(root_dir):
        if any(name.endswith("_Amplitude.csv") for name in filenames):
            dataset_dirs.append(current_dir)

    canonical_dirs = []
    for data_dir in dataset_dirs:
        if glob.glob(os.path.join(data_dir, "*_ClusterData.csv")):
            canonical_dirs.append(data_dir)
            continue

        amplitude_names = {
            os.path.basename(path)
            for path in glob.glob(os.path.join(data_dir, "*_Amplitude.csv"))
        }
        has_nested_cluster_copy = False
        for candidate_dir in dataset_dirs:
            if candidate_dir == data_dir or not candidate_dir.startswith(data_dir + os.sep):
                continue
            if not glob.glob(os.path.join(candidate_dir, "*_ClusterData.csv")):
                continue
            nested_amplitude_names = {
                os.path.basename(path)
                for path in glob.glob(os.path.join(candidate_dir, "*_Amplitude.csv"))
            }
            if amplitude_names and amplitude_names.issubset(nested_amplitude_names):
                has_nested_cluster_copy = True
                break
        if not has_nested_cluster_copy:
            canonical_dirs.append(data_dir)
    return sorted(canonical_dirs)


def read_cluster_data(data_dir: str) -> tuple[str | None, pd.DataFrame]:
    cluster_files = sorted(glob.glob(os.path.join(data_dir, "*_ClusterData.csv")))
    if not cluster_files:
        columns = ["Well", "Target 1", "Target 2", "Count", "Ch1 Mean", "Ch1 StdDev", "Ch2 Mean", "Ch2 StdDev"]
        return None, pd.DataFrame(columns=columns)

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
            cluster_df[col] = pd.to_numeric(
                cluster_df[col].astype(str).str.replace(",", "", regex=False),
                errors="coerce",
            )
    for target_col in ["Target 1", "Target 2"]:
        if target_col in cluster_df.columns:
            cluster_df[target_col] = cluster_df[target_col].astype(str).str.strip()
    return cluster_csv, cluster_df


def gate_positions(well_rows: pd.DataFrame) -> tuple[float | None, float | None]:
    target1_pos = well_rows[well_rows["Target 1"].astype(str).str.strip().isin(["1", "1.0"])]
    target1_neg = well_rows[well_rows["Target 1"].astype(str).str.strip().isin(["0", "0.0"])]
    target2_pos = well_rows[well_rows["Target 2"].astype(str).str.strip().isin(["1", "1.0"])]
    target2_neg = well_rows[well_rows["Target 2"].astype(str).str.strip().isin(["0", "0.0"])]

    x_gate = None
    y_gate = None
    if not target1_pos.empty and not target1_neg.empty:
        y_neg_mean = float(target1_neg["Ch1 Mean"].mean())
        y_pos_mean = float(target1_pos["Ch1 Mean"].mean())
        if np.isfinite(y_neg_mean) and np.isfinite(y_pos_mean):
            y_gate = 0.5 * (y_neg_mean + y_pos_mean)
    if not target2_pos.empty and not target2_neg.empty:
        x_neg_mean = float(target2_neg["Ch2 Mean"].mean())
        x_pos_mean = float(target2_pos["Ch2 Mean"].mean())
        if np.isfinite(x_neg_mean) and np.isfinite(x_pos_mean):
            x_gate = 0.5 * (x_neg_mean + x_pos_mean)
    return x_gate, y_gate


def estimate_single_class_gates(
    well_rows: pd.DataFrame,
    ch1_values: np.ndarray,
    ch2_values: np.ndarray,
) -> tuple[float | None, float | None]:
    def boundary(values: np.ndarray, positive: bool) -> float | None:
        finite_values = values[np.isfinite(values)]
        if not len(finite_values):
            return None
        edge = np.min(finite_values) if positive else np.max(finite_values)
        direction = -np.inf if positive else np.inf
        return float(np.nextafter(edge, direction))

    target1_states = set(
        well_rows["Target 1"].astype(str).str.strip()
    ) & {"0", "0.0", "1", "1.0"}
    target2_states = set(
        well_rows["Target 2"].astype(str).str.strip()
    ) & {"0", "0.0", "1", "1.0"}

    x_gate = None
    y_gate = None
    if len(target2_states) == 1:
        x_gate = boundary(ch2_values, next(iter(target2_states)) in {"1", "1.0"})
    if len(target1_states) == 1:
        y_gate = boundary(ch1_values, next(iter(target1_states)) in {"1", "1.0"})
    return x_gate, y_gate


def manual_label_columns(amplitude_df: pd.DataFrame) -> list[str]:
    amplitude_cols = {"Ch1Amplitude", "Ch2Amplitude"}
    label_cols = []
    for column in amplitude_df.columns:
        column_name = str(column)
        if column_name in amplitude_cols:
            continue
        if column_name.startswith("Unnamed"):
            continue
        if "Ch1" in column_name or "Ch2" in column_name:
            continue
        label_cols.append(column)
    return label_cols[:2]


def normalize_manual_label(value: object) -> str:
    if pd.isna(value):
        return ""
    label = str(value).strip().lower()
    if label in {"0", "0.0"}:
        return "0"
    if label in {"1", "1.0"}:
        return "1"
    if label == "u":
        return "u"
    return ""


def manual_axis_gate(values: np.ndarray, labels: pd.Series) -> tuple[float | None, str]:
    normalized = labels.map(normalize_manual_label).to_numpy()
    finite_mask = np.isfinite(values)
    valid_mask = finite_mask & np.isin(normalized, ["0", "1"])
    if not valid_mask.any():
        return None, "manual_labels_unavailable"

    axis_values = values[valid_mask]
    axis_labels = normalized[valid_mask]
    states = set(axis_labels)
    if states == {"1"}:
        return float(np.nextafter(np.min(axis_values), -np.inf)), "manual_single_positive_amplitude_bound"
    if states == {"0"}:
        return float(np.nextafter(np.max(axis_values), np.inf)), "manual_single_negative_amplitude_bound"

    neg_values = axis_values[axis_labels == "0"]
    pos_values = axis_values[axis_labels == "1"]
    max_neg = float(np.max(neg_values))
    min_pos = float(np.min(pos_values))
    if max_neg <= min_pos:
        return 0.5 * (max_neg + min_pos), "manual_class_boundary"

    unique_values = np.unique(axis_values)
    if len(unique_values) == 1:
        return float(unique_values[0]), "manual_class_optimized_threshold"
    order = np.argsort(axis_values)
    sorted_values = axis_values[order]
    sorted_is_positive = axis_labels[order] == "1"
    cumulative_pos = np.cumsum(sorted_is_positive)
    total_pos = int(np.sum(sorted_is_positive))
    total_neg = len(sorted_is_positive) - total_pos
    change_indices = np.where(sorted_values[:-1] != sorted_values[1:])[0]
    pos_left = cumulative_pos[change_indices]
    neg_left = (change_indices + 1) - pos_left
    neg_right = total_neg - neg_left
    middle_thresholds = (sorted_values[change_indices] + sorted_values[change_indices + 1]) / 2
    thresholds = np.concatenate(
        [
            [np.nextafter(sorted_values[0], -np.inf)],
            middle_thresholds,
            [np.nextafter(sorted_values[-1], np.inf)],
        ]
    )
    errors = np.concatenate([[total_neg], pos_left + neg_right, [total_pos]])
    best_index = int(np.argmin(errors))
    return float(thresholds[best_index]), "manual_class_optimized_threshold"


def manual_gate_positions(
    amplitude_df: pd.DataFrame,
    ch1_col: str,
    ch2_col: str,
) -> tuple[float | None, float | None, str, str, list[str]]:
    label_cols = manual_label_columns(amplitude_df)
    if len(label_cols) < 2:
        return None, None, "manual_label_columns_missing", "manual_label_columns_missing", label_cols

    y_gate, y_method = manual_axis_gate(
        amplitude_df[ch1_col].to_numpy(dtype=float),
        amplitude_df[label_cols[0]],
    )
    x_gate, x_method = manual_axis_gate(
        amplitude_df[ch2_col].to_numpy(dtype=float),
        amplitude_df[label_cols[1]],
    )
    return x_gate, y_gate, x_method, y_method, label_cols


def summarize_manual_labels(
    amplitude_df: pd.DataFrame,
    ch1_col: str,
    ch2_col: str,
    label_cols: list[str],
) -> dict[str, object]:
    if len(label_cols) < 2:
        return {
            "target1": "",
            "target2": "",
            "quadrant": "",
            "count": "",
            "ch1_mean": "",
            "ch2_mean": "",
            "summary_rows": [],
        }

    summary_df = amplitude_df[[ch1_col, ch2_col, label_cols[0], label_cols[1]]].copy()
    summary_df["target1"] = summary_df[label_cols[0]].map(normalize_manual_label)
    summary_df["target2"] = summary_df[label_cols[1]].map(normalize_manual_label)
    summary_df = summary_df[summary_df["target1"].isin(["0", "1"]) & summary_df["target2"].isin(["0", "1"])]
    order = {"0": 0, "1": 1}
    rows = []
    for (target1, target2), group in summary_df.groupby(["target1", "target2"], sort=False):
        rows.append({
            "target1": target1,
            "target2": target2,
            "quadrant": quadrant_label(target1, target2),
            "count": int(len(group)),
            "ch1_mean": float(group[ch1_col].mean()),
            "ch2_mean": float(group[ch2_col].mean()),
        })
    rows.sort(key=lambda row: (order[row["target1"]], order[row["target2"]]))
    target1_values = sorted(set(summary_df["target1"]), key=lambda value: order[value])
    target2_values = sorted(set(summary_df["target2"]), key=lambda value: order[value])
    return {
        "target1": "|".join(target1_values),
        "target2": "|".join(target2_values),
        "quadrant": "|".join(row["quadrant"] for row in rows),
        "count": "|".join(str(row["count"]) for row in rows),
        "ch1_mean": "|".join(str(row["ch1_mean"]) for row in rows),
        "ch2_mean": "|".join(str(row["ch2_mean"]) for row in rows),
        "summary_rows": rows,
    }


def has_usable_amplitude_points(amplitude_path: str) -> bool:
    try:
        chunks = pd.read_csv(
            amplitude_path,
            skiprows=3,
            chunksize=100,
            low_memory=False,
        )
        for chunk in chunks:
            ch1_col = next((column for column in chunk.columns if "Ch1" in str(column)), None)
            ch2_col = next((column for column in chunk.columns if "Ch2" in str(column)), None)
            if ch1_col is None or ch2_col is None:
                return False
            if not chunk[[ch1_col, ch2_col]].dropna().empty:
                return True
    except pd.errors.EmptyDataError:
        return False
    return False


def precheck_manual_gate_availability(amplitude_path: str) -> tuple[bool, bool, bool, list[str]]:
    has_points = False
    label_cols: list[str] = []
    target1_states: set[str] = set()
    target2_states: set[str] = set()
    try:
        chunks = pd.read_csv(
            amplitude_path,
            skiprows=3,
            chunksize=1000,
            low_memory=False,
        )
        for chunk in chunks:
            ch1_col = next((column for column in chunk.columns if "Ch1" in str(column)), None)
            ch2_col = next((column for column in chunk.columns if "Ch2" in str(column)), None)
            if ch1_col is None or ch2_col is None:
                return False, False, False, []
            if not chunk[[ch1_col, ch2_col]].dropna().empty:
                has_points = True
            if not label_cols:
                label_cols = manual_label_columns(chunk)
            if len(label_cols) >= 2:
                target1_states.update(
                    label for label in chunk[label_cols[0]].map(normalize_manual_label).unique()
                    if label in {"0", "1"}
                )
                target2_states.update(
                    label for label in chunk[label_cols[1]].map(normalize_manual_label).unique()
                    if label in {"0", "1"}
                )
            if has_points and target1_states and target2_states:
                break
    except pd.errors.EmptyDataError:
        return False, False, False, []
    return has_points, bool(target2_states), bool(target1_states), label_cols


def precheck_gate_data(
    dataset_dirs: list[str],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    missing_rows = []
    skipped_rows = []
    for data_dir in dataset_dirs:
        parent_dataset = os.path.basename(os.path.abspath(data_dir))
        amplitude_csvs = sorted(glob.glob(os.path.join(data_dir, "*_Amplitude.csv")))
        for amp_csv in amplitude_csvs:
            dataset = os.path.splitext(os.path.basename(amp_csv))[0]
            well = parse_well_name(amp_csv)
            has_points, x_estimate_available, y_estimate_available, label_cols = precheck_manual_gate_availability(amp_csv)
            if not has_points:
                skipped_rows.append({
                    "dataset": dataset,
                    "parent_dataset": parent_dataset,
                    "well": well,
                    "source_amplitude_csv": os.path.basename(amp_csv),
                    "missing_x_gate": True,
                    "missing_y_gate": True,
                    "processing_status": "not_processed",
                    "reason": "No usable Ch1/Ch2 observations",
                })
                continue

            if x_estimate_available and y_estimate_available:
                continue

            reasons = []
            if len(label_cols) < 2:
                reasons.append("Manual amplitude classification columns missing")
            if not x_estimate_available:
                reasons.append("Missing usable manual Target 2 labels for x gate")
            if not y_estimate_available:
                reasons.append("Missing usable manual Target 1 labels for y gate")
            missing_rows.append({
                "dataset": dataset,
                "parent_dataset": parent_dataset,
                "well": well,
                "source_amplitude_csv": os.path.basename(amp_csv),
                "missing_x_gate": not x_estimate_available,
                "missing_y_gate": not y_estimate_available,
                "processing_status": "processable",
                "reason": "; ".join(reasons),
            })
    return missing_rows, skipped_rows


def mip_output_dir(data_dir: str) -> str:
    current_dir = os.path.abspath(data_dir)
    while True:
        if os.path.basename(current_dir).lower().startswith("mip_"):
            return os.path.join(current_dir, "Biorad_manual_gating_output")
        parent_dir = os.path.dirname(current_dir)
        if parent_dir == current_dir:
            return os.path.join(data_dir, "Biorad_manual_gating_output")
        current_dir = parent_dir


def quadrant_label(target1: str, target2: str) -> str:
    mapping = {
        ("1", "1"): "Q1",
        ("1", "0"): "Q2",
        ("0", "0"): "Q3",
        ("0", "1"): "Q4",
    }
    return mapping.get((target1.strip(), target2.strip()), "")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create Bio-Rad gate summaries, diagnostic plots, and a full feature dataset."
    )
    parser.add_argument(
        "dataset_path",
        help="Dataset folder or root containing amplitude CSV files; ClusterData CSV files are optional.",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Reprocess all wells instead of updating the existing dataset.",
    )
    parser.add_argument(
        "--refresh-existing",
        action="store_true",
        help="Replace existing rows for the supplied dataset folders while retaining all other datasets.",
    )
    parser.add_argument(
        "--precheck-only",
        action="store_true",
        help="Report amplitude files without complete gate data, then exit without processing.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Build CSV outputs without regenerating per-well diagnostic PNG plots.",
    )
    args = parser.parse_args()

    root_dir = os.path.abspath(args.dataset_path)
    dataset_dirs = find_dataset_dirs(root_dir)
    if not dataset_dirs:
        print(f"No dataset folders containing amplitude CSVs found in: {root_dir}")
        sys.exit(1)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    project_output_dir = os.path.join(project_dir, "data_output")
    gate_output_dir = project_output_dir
    os.makedirs(project_output_dir, exist_ok=True)
    all_data_csv = os.path.join(gate_output_dir, "biorad_cluster_gate_data.csv")
    output_path = os.path.join(project_output_dir, "full_biorad_cluster_dataset.csv")
    precheck_path = os.path.join(project_output_dir, "precheck_missing_gate_data.csv")
    precheck_rows, skipped_rows = precheck_gate_data(dataset_dirs)
    precheck_report_rows = precheck_rows + skipped_rows
    scoped_parents = {os.path.basename(os.path.abspath(data_dir)) for data_dir in dataset_dirs}
    if not args.rebuild and os.path.exists(precheck_path):
        previous_precheck = pd.read_csv(precheck_path)
        if "parent_dataset" in previous_precheck.columns:
            retained_precheck = previous_precheck[
                ~previous_precheck["parent_dataset"].astype(str).isin(scoped_parents)
            ].to_dict(orient="records")
            precheck_report_rows = retained_precheck + precheck_report_rows
    precheck_columns = [
        "dataset", "parent_dataset", "well", "source_amplitude_csv",
        "missing_x_gate", "missing_y_gate", "processing_status", "reason",
    ]
    pd.DataFrame(precheck_report_rows, columns=precheck_columns).to_csv(precheck_path, index=False)
    total_amplitude_files = sum(
        len(glob.glob(os.path.join(data_dir, "*_Amplitude.csv")))
        for data_dir in dataset_dirs
    )
    processable_amplitude_files = total_amplitude_files - len(skipped_rows)
    print(
        f"Precheck: {len(precheck_rows)} of {processable_amplitude_files} processable amplitude files "
        f"are missing at least one gate axis. Report: {precheck_path}"
    )
    print(f"Amplitude files not processed (no usable Ch1/Ch2 observations): {len(skipped_rows)}")
    for row in precheck_report_rows[:10]:
        print(f"  {row['source_amplitude_csv']} ({row['parent_dataset']}): {row['reason']}")
    if len(precheck_report_rows) > 10:
        print(f"  ... and {len(precheck_report_rows) - 10} more; see the precheck report.")
    if args.precheck_only:
        return

    csv_fields = [
        "dataset", "parent_dataset", "well", "source_amplitude_csv", "x_gate", "y_gate",
        "x_gate_method", "y_gate_method",
        "x_min_ch2", "x_max_ch2", "y_min_ch1", "y_max_ch1", "target1(MUT)",
        "target2(WT)", "quadrant", "count", "ch1_mean", "ch2_mean",
    ]

    gate_rows = []
    feature_rows = []
    existing_dataset = None
    processed_keys: set[tuple[str, str]] = set()
    if not args.rebuild and os.path.exists(output_path):
        existing_dataset = pd.read_csv(output_path)
        required_columns = {
            "dataset",
            "parent_dataset",
            *[
                column
                for column in csv_fields
                if column not in {
                    "source_amplitude_csv", "dataset", "parent_dataset",
                    "x_gate_method", "y_gate_method",
                }
            ],
        }
        missing_columns = sorted(required_columns - set(existing_dataset.columns))
        if missing_columns:
            raise ValueError(
                f"Existing dataset is missing required columns: {', '.join(missing_columns)}. "
                "Run with --rebuild to regenerate it."
            )
        for column in ["x_gate_method", "y_gate_method"]:
            if column not in existing_dataset.columns:
                existing_dataset[column] = "legacy_unrecorded"
        processed_keys = set(
            zip(
                existing_dataset["parent_dataset"].astype(str),
                existing_dataset["dataset"].astype(str),
            )
        )
        if args.refresh_existing:
            processed_keys.clear()
        existing_gate_rows = existing_dataset[
            [column for column in csv_fields if column != "source_amplitude_csv"]
        ].to_dict(orient="records")
        for row in existing_gate_rows:
            row["source_amplitude_csv"] = f"{row['dataset']}.csv"
        gate_rows.extend(existing_gate_rows)
        print(f"Loaded existing wells: {len(existing_dataset)}")

    for index, data_dir in enumerate(dataset_dirs, start=1):
        print(f"Processing dataset folder {index}/{len(dataset_dirs)}: {data_dir}", flush=True)
        process_dataset(data_dir, gate_rows, feature_rows, processed_keys, skip_plots=args.no_plots)

    if not gate_rows:
        raise ValueError("No wells with readable amplitude data were found.")

    gate_well_count = 0
    calculated_well_count = 0
    if feature_rows:
        gate = add_gate_features(pd.DataFrame(gate_rows[-len(feature_rows):]))
        features = pd.DataFrame(feature_rows)
        gate_well_count = len(gate)
        calculated_well_count = len(features)
        new_rows = gate.merge(features, on="dataset", how="left", validate="one_to_one")
        new_rows = new_rows.drop(columns=["source_amplitude_csv"], errors="ignore")
        new_rows = new_rows.rename(columns={
            "gate_n_cluster_rows": "n_populated_quadrants",
            "gate_ch1_mean_weighted": "weighted_mean_ch1_MUT",
            "gate_ch2_mean_weighted": "weighted_mean_ch2_WT",
            "gate_ch1_range": "amplitude_range_MUT",
            "gate_ch2_range": "amplitude_range_WT",
        })
        if args.refresh_existing and existing_dataset is not None:
            refreshed_keys = set(
                zip(
                    new_rows["parent_dataset"].astype(str),
                    new_rows["dataset"].astype(str),
                )
            )
            existing_keys = list(
                zip(
                    existing_dataset["parent_dataset"].astype(str),
                    existing_dataset["dataset"].astype(str),
                )
            )
            existing_dataset = existing_dataset.loc[
                [key not in refreshed_keys for key in existing_keys]
            ].copy()
        merged = (
            pd.concat([existing_dataset, new_rows], ignore_index=True, sort=False)
            if existing_dataset is not None
            else new_rows
        )
    else:
        merged = existing_dataset
        if merged is None:
            raise ValueError("No new or previously processed wells are available.")

    merged.to_csv(output_path, index=False)
    new_gate_rows = gate_rows[-len(feature_rows):] if feature_rows else []
    if args.refresh_existing and new_gate_rows:
        refreshed_keys = {
            (str(row["parent_dataset"]), str(row["dataset"]))
            for row in new_gate_rows
        }
        gate_rows = [
            row for row in gate_rows
            if (str(row["parent_dataset"]), str(row["dataset"])) not in refreshed_keys
        ] + new_gate_rows
    with open(all_data_csv, "w", encoding="utf-8", newline="") as all_data_fh:
        all_data_writer = csv.DictWriter(all_data_fh, fieldnames=csv_fields)
        all_data_writer.writeheader()
        all_data_writer.writerows(gate_rows)

    all_skipped_rows = [
        row for row in precheck_report_rows
        if row.get("processing_status") == "not_processed"
    ]
    missing_gate_path, missing_gate_count, no_gate_count = write_missing_gate_report(
        merged,
        project_output_dir,
        all_skipped_rows,
    )
    print(f"New wells processed: {len(feature_rows)}")
    print(f"Previously processed wells retained: {len(existing_dataset) if existing_dataset is not None else 0}")
    print(f"All x/y data saved to: {all_data_csv}")
    print(f"Saved full dataset: {output_path}")
    print(f"Saved missing-gate wells: {missing_gate_path}")
    print(f"New gate-summary wells: {gate_well_count}")
    print(f"Newly calculated wells: {calculated_well_count}")
    print(f"Merged rows: {len(merged)}")
    print(f"Rows without features: {int(merged['n_droplets'].isna().sum())}")
    print(f"Processed wells missing at least one gate: {missing_gate_count}")
    print(f"Processed wells missing both gates: {no_gate_count}")
    print(f"Unprocessed wells included in missing-gate report: {len(skipped_rows)}")


def process_dataset(
    data_dir: str,
    gate_rows: list[dict[str, object]],
    feature_rows: list[dict[str, object]],
    processed_keys: set[tuple[str, str]],
    skip_plots: bool = False,
) -> None:
    amplitude_csvs = sorted(glob.glob(os.path.join(data_dir, "*_Amplitude.csv")))
    if not amplitude_csvs:
        print(f"No *_Amplitude.csv files found in: {data_dir}")
        sys.exit(1)

    out_dir = mip_output_dir(data_dir)
    if not skip_plots:
        os.makedirs(out_dir, exist_ok=True)

    dataset_name = os.path.basename(os.path.abspath(data_dir))

    for amp_csv in amplitude_csvs:
        dataset_id = os.path.splitext(os.path.basename(amp_csv))[0]
        if (dataset_name, dataset_id) in processed_keys:
            continue

        well = parse_well_name(amp_csv)
        amp_df = pd.read_csv(amp_csv, skiprows=3, low_memory=False)
        ch1_col = next((c for c in amp_df.columns if "Ch1" in str(c)), None)
        ch2_col = next((c for c in amp_df.columns if "Ch2" in str(c)), None)
        if ch1_col is None or ch2_col is None:
            print(f"Skipping {amp_csv}: amplitude columns not found.")
            continue

        label_cols = manual_label_columns(amp_df)
        retained_cols = [ch1_col, ch2_col] + label_cols
        amp_df = amp_df[retained_cols].dropna(subset=[ch1_col, ch2_col]).copy()
        if amp_df.empty:
            print(f"Skipping {amp_csv}: no amplitude points available.")
            continue

        x_gate, y_gate, x_gate_method, y_gate_method, label_cols = manual_gate_positions(amp_df, ch1_col, ch2_col)
        if x_gate is None and y_gate is None:
            print(f"Plotting {well} without gate lines: no valid manual labels found in amplitude CSV.")
        manual_summary = summarize_manual_labels(amp_df, ch1_col, ch2_col, label_cols)
        gate_row = {
            "dataset": dataset_id,
            "parent_dataset": dataset_name,
            "well": well,
            "source_amplitude_csv": os.path.basename(amp_csv),
            "x_gate": x_gate,
            "y_gate": y_gate,
            "x_gate_method": x_gate_method,
            "y_gate_method": y_gate_method,
            "x_min_ch2": amp_df[ch2_col].min(),
            "x_max_ch2": amp_df[ch2_col].max(),
            "y_min_ch1": amp_df[ch1_col].min(),
            "y_max_ch1": amp_df[ch1_col].max(),
            "target1(MUT)": manual_summary["target1"],
            "target2(WT)": manual_summary["target2"],
            "quadrant": manual_summary["quadrant"],
            "count": manual_summary["count"],
            "ch1_mean": manual_summary["ch1_mean"],
            "ch2_mean": manual_summary["ch2_mean"],
        }
        gate_rows.append(gate_row)
        feature_row: dict[str, object] = {"dataset": gate_row["dataset"]}
        feature_row.update(extract_features(amp_df[[ch1_col, ch2_col]].to_numpy(dtype=float)))
        feature_rows.append(feature_row)
        processed_keys.add((dataset_name, dataset_id))
        if skip_plots:
            continue

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

        # Show the mean of each manual quadrant as a marker, useful for validation.
        for row in manual_summary["summary_rows"]:
            try:
                x_mean = float(row["ch2_mean"])
                y_mean = float(row["ch1_mean"])
                ax.scatter([x_mean], [y_mean], s=12, edgecolors="black", linewidth=0.4, zorder=3)
            except (TypeError, ValueError):
                pass

        ax.set_xlabel("Ch2Amplitude")
        ax.set_ylabel("Ch1Amplitude")
        if any("single" in method for method in {x_gate_method, y_gate_method}):
            gate_title = "amplitude-bound gate estimate"
        else:
            gate_title = "gate from manual labels"
        ax.set_title(f"{well} {gate_title} (Ch2 x, Ch1 y)")
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

    if skip_plots:
        print(f"Done. Plot generation skipped for: {data_dir}")
    else:
        print(f"Done. PNG files saved under: {out_dir}")


if __name__ == "__main__":
    main()
