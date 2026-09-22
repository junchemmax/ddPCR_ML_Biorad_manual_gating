"""
Build a large cluster-count training dataset by pairing every well's raw
amplitude statistics with an objective `true_n_clusters` label derived from
that well's `*_ClusterData.csv` (QuantaSoft/QX Manager quadrant-gating export).

Unlike meta_dataset.csv (which only has rows for wells that were manually run
through optimize_flowsom_rs_params_single.py), this covers EVERY well in EVERY
MIP_XXX folder that has a ClusterData.csv next to its Amplitude CSVs — no
Optuna/FlowSOM run required, since ClusterData.csv already tells us the real
number of populations.

Note: this dataset only has features + best_n_clusters (no best_xdim/ydim/rlen,
since those require an actual FlowSOM/Optuna grid search). It is meant to
expand training data for the cluster-count classifier only.

Usage:
    python flowsom/build_full_cluster_dataset.py [--count-threshold 2] [--n-jobs -1] [--limit 200]
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import time
from collections import defaultdict

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde, kurtosis, skew

HERE      = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
DATA_ROOT = os.path.join(REPO_ROOT, "ddPCR_data")
OUT_CSV   = os.path.join(HERE, "output", "full_cluster_dataset.csv")

WELL_RE = re.compile(r"^[A-H](0[1-9]|1[0-2])$")


def parse_quadrant_counts(cluster_csv_path: str) -> dict:
    """Return {well: [count, count, ...]} for the quadrant-counts table only."""
    well_counts: dict = defaultdict(list)
    with open(cluster_csv_path, "r", encoding="utf-8-sig", errors="ignore") as fh:
        for line in fh:
            parts = [p.strip() for p in line.strip().split(",")]
            if len(parts) < 4:
                continue
            well = parts[0]
            if not WELL_RE.match(well):
                continue
            target1 = parts[1]
            # Quadrant-counts rows have Target1/Target2 in {0, 1, u}; the S-value
            # table's second column instead holds strings like "NN"/"PN"/"NP"/"PP".
            if target1 not in ("0", "1", "u"):
                continue
            try:
                count = int(float(parts[3]))
            except ValueError:
                continue
            well_counts[well].append(count)
    return dict(well_counts)


def well_from_stem(stem: str) -> str | None:
    tokens = stem.split("_")
    for tok in (tokens[-1], tokens[-2] if len(tokens) >= 2 else ""):
        if WELL_RE.match(tok):
            return tok
    return None


# ── Feature extraction (must match optimize_flowsom_rs_params_single.py) ─────────
def _kde_peak_count(vals: np.ndarray) -> int:
    kde = gaussian_kde(vals, bw_method=0.15)
    xs = np.linspace(vals.min(), vals.max(), 500)
    density = kde(xs)
    peaks, _ = find_peaks(density, prominence=density.max() * 0.05)
    return len(peaks)


def extract_features(X: np.ndarray) -> dict:
    feat: dict = {}
    feat["n_droplets"] = len(X)
    for i, ch in enumerate(["ch1", "ch2"]):
        vals = X[:, i]
        mu    = float(np.mean(vals))
        sigma = float(np.std(vals))
        p25   = float(np.percentile(vals, 25))
        p75   = float(np.percentile(vals, 75))
        feat[f"{ch}_std"]    = sigma
        feat[f"{ch}_iqr"]    = p75 - p25
        feat[f"{ch}_cv"]     = sigma / (mu + 1e-9)
        feat[f"{ch}_skew"]   = float(skew(vals))
        feat[f"{ch}_kurt"]   = float(kurtosis(vals))
        feat[f"{ch}_bimodality"] = (feat[f"{ch}_skew"] ** 2 + 1) / (feat[f"{ch}_kurt"] + 3 + 1e-9)
        log_vals = np.log1p(np.clip(vals, 0, None))
        feat[f"{ch}_log_std"]  = float(np.std(log_vals))
        feat[f"{ch}_log_skew"] = float(skew(log_vals))
        feat[f"{ch}_outlier_frac"] = float(np.mean((vals < mu - 3 * sigma) | (vals > mu + 3 * sigma)))
        feat[f"{ch}_kde_peaks"] = _kde_peak_count(vals)
    feat["ch1_ch2_corr"] = float(np.corrcoef(X[:, 0], X[:, 1])[0, 1])
    return feat


def process_well(amp_path: str, true_n_clusters: int) -> dict | None:
    try:
        df = pd.read_csv(amp_path, skiprows=3)[["Ch1Amplitude", "Ch2Amplitude"]].dropna()
    except Exception:
        return None
    data = df.values.astype(np.float64)
    if len(data) < 10:
        return None
    features = extract_features(data)
    features["dataset"]         = os.path.splitext(os.path.basename(amp_path))[0]
    features["best_n_clusters"] = true_n_clusters
    return features


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count-threshold", type=int, default=2,
                         help="Minimum droplet count for a quadrant to count as a real population (default: 2).")
    parser.add_argument("--n-jobs", type=int, default=-1, help="Parallel jobs (default: -1, all cores).")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N wells (for testing).")
    args = parser.parse_args()

    cluster_csvs = sorted(glob.glob(os.path.join(DATA_ROOT, "**", "*_ClusterData.csv"), recursive=True))
    print(f"Found {len(cluster_csvs)} ClusterData.csv file(s).")

    tasks = []  # (amp_path, true_n_clusters)
    for cluster_csv in cluster_csvs:
        well_counts = parse_quadrant_counts(cluster_csv)
        dir_ = os.path.dirname(cluster_csv)
        amp_files = glob.glob(os.path.join(dir_, "*_Amplitude.csv"))
        for amp_path in amp_files:
            stem = os.path.splitext(os.path.basename(amp_path))[0]
            well = well_from_stem(stem)
            if well is None:
                continue
            counts = well_counts.get(well)
            if counts is None:
                continue
            true_n_clusters = sum(1 for c in counts if c >= args.count_threshold)
            tasks.append((amp_path, true_n_clusters))

    if args.limit:
        tasks = tasks[: args.limit]
    print(f"Prepared {len(tasks)} well(s) to process (n_jobs={args.n_jobs}) …")

    t0 = time.time()
    results = Parallel(n_jobs=args.n_jobs, verbose=5)(
        delayed(process_well)(amp_path, tnc) for amp_path, tnc in tasks
    )
    rows = [r for r in results if r is not None]
    print(f"Processed {len(rows)}/{len(tasks)} wells in {time.time() - t0:.1f}s.")

    out_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    out_df.to_csv(OUT_CSV, index=False)
    print(f"Saved -> {OUT_CSV}  ({len(out_df)} rows, {out_df.shape[1]} columns)")
    print("\nbest_n_clusters distribution:")
    print(out_df["best_n_clusters"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
