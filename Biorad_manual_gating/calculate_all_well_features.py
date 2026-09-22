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
    meta_path = os.path.join(output_dir, "meta_dataset.csv")
    gate_path = os.path.join(output_dir, "biorad_cluster_gate_data.csv")
    all_meta_path = os.path.join(output_dir, "meta_dataset_all_wells.csv")
    all_merged_path = os.path.join(output_dir, "meta_biorad_cluster_dataset_all_wells.csv")

    existing = pd.read_csv(meta_path)
    gate = pd.read_csv(gate_path)
    known_sources = set(existing["dataset"].astype(str) + ".csv")
    amplitude_paths = {}
    for path in glob.glob(os.path.join(os.path.dirname(here), "ddPCR_data", "**", "*_Amplitude.csv"), recursive=True):
        amplitude_paths.setdefault(os.path.basename(path), path)

    missing_sources = [source for source in gate["source_amplitude_csv"] if source not in known_sources]
    rows = []
    for position, source in enumerate(missing_sources, start=1):
        amplitude_path = amplitude_paths[source]
        data = pd.read_csv(amplitude_path, skiprows=3, low_memory=False)[
            ["Ch1Amplitude", "Ch2Amplitude"]
        ].dropna().to_numpy(dtype=float)
        row = extract_features(data)
        row.update({
            "dataset": os.path.splitext(source)[0],
            "best_n_clusters": np.nan,
            "best_xdim": np.nan,
            "best_ydim": np.nan,
            "best_rlen": np.nan,
            "clear_clustering": np.nan,
            "mut_found": np.nan,
            "metric": "biorad_cluster_gate",
        })
        rows.append(row)
        if position % 250 == 0 or position == len(missing_sources):
            print(f"Calculated {position}/{len(missing_sources)} missing wells")

    calculated = pd.DataFrame(rows)
    all_meta = pd.concat([existing, calculated], ignore_index=True)
    all_meta = all_meta[[column for column in existing.columns if column in all_meta.columns]]
    all_meta.to_csv(all_meta_path, index=False)

    gate["source_amplitude_csv"] = gate["source_amplitude_csv"].astype(str)
    all_meta["source_amplitude_csv"] = all_meta["dataset"].astype(str) + ".csv"
    merged = gate.merge(all_meta, on="source_amplitude_csv", how="left", suffixes=("", "_meta"), validate="one_to_one")
    merged.to_csv(all_merged_path, index=False)

    print(f"Saved all-well metadata: {all_meta_path}")
    print(f"Saved all-well merged data: {all_merged_path}")
    print(f"Gate wells: {len(gate)}")
    print(f"Calculated missing wells: {len(calculated)}")
    print(f"Merged rows: {len(merged)}")
    print(f"Rows without features: {int(merged['n_droplets'].isna().sum())}")


if __name__ == "__main__":
    main()