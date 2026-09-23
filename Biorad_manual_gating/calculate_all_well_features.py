import glob
import os
import warnings

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde, kurtosis, skew

warnings.filterwarnings("ignore", category=RuntimeWarning)


FEATURE_COLUMNS = [
    "n_droplets",
    "ch1_std", "ch1_iqr", "ch1_cv", "ch1_skew", "ch1_kurt", "ch1_bimodality",
    "ch1_log_std", "ch1_log_skew", "ch1_outlier_frac", "ch1_kde_peaks",
    "ch2_std", "ch2_iqr", "ch2_cv", "ch2_skew", "ch2_kurt", "ch2_bimodality",
    "ch2_log_std", "ch2_log_skew", "ch2_outlier_frac", "ch2_kde_peaks",
    "ch1_ch2_corr",
]


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
                "gate_n_cluster_rows": len(parse_pipe_numbers(row["quadrant"])),
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


def kde_peak_count(values: np.ndarray) -> int:
    if len(values) < 3 or np.ptp(values) == 0:
        return 1
    try:
        density = gaussian_kde(values, bw_method=0.15)(
            np.linspace(values.min(), values.max(), 500)
        )
        peaks, _ = find_peaks(density, prominence=density.max() * 0.05)
        return int(len(peaks))
    except (np.linalg.LinAlgError, ValueError):
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


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(here, "output")
    gate_path = os.path.join(output_dir, "biorad_cluster_gate_data.csv")
    gate = add_gate_features(pd.read_csv(gate_path))
    amplitude_paths = {}
    for path in glob.glob(os.path.join(os.path.dirname(here), "ddPCR_data", "**", "*_Amplitude.csv"), recursive=True):
        amplitude_paths.setdefault(os.path.basename(path), path)

    missing_sources = gate["source_amplitude_csv"].astype(str).tolist()
    rows = []
    for position, source in enumerate(missing_sources, start=1):
        amplitude_path = amplitude_paths[source]
        data = pd.read_csv(amplitude_path, skiprows=3, low_memory=False)[
            ["Ch1Amplitude", "Ch2Amplitude"]
        ].dropna().to_numpy(dtype=float)
        row = extract_features(data)
        row.update({
            "dataset": os.path.splitext(source)[0],
        })
        rows.append(row)
        if position % 250 == 0 or position == len(missing_sources):
            print(f"Calculated {position}/{len(missing_sources)} missing wells")

    features = pd.DataFrame(rows)
    gate["dataset"] = gate["source_amplitude_csv"].astype(str).str.replace(".csv", "", regex=False)
    merged = gate.merge(features, on="dataset", how="left", validate="one_to_one")
    merged = merged.drop(columns=["source_amplitude_csv"], errors="ignore")
    merged = merged.rename(columns={
        "gate_n_cluster_rows": "n_populated_quadrants",
        "gate_ch1_mean_weighted": "weighted_mean_ch1_MUT",
        "gate_ch2_mean_weighted": "weighted_mean_ch2_WT",
        "gate_ch1_range": "amplitude_range_MUT",
        "gate_ch2_range": "amplitude_range_WT",
    })
    output_path = os.path.join(output_dir, "full_biorad_cluster_dataset.csv")
    merged.to_csv(output_path, index=False)

    print(f"Saved full dataset: {output_path}")
    print(f"Gate wells: {len(gate)}")
    print(f"Calculated wells: {len(features)}")
    print(f"Merged rows: {len(merged)}")
    print(f"Rows without features: {int(merged['n_droplets'].isna().sum())}")


if __name__ == "__main__":
    main()