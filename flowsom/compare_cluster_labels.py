"""
Compare the manually-entered `best_n_clusters` label in meta_dataset.csv against
an objective ground truth derived from each well's `*_ClusterData.csv` file
(QuantaSoft/QX Manager's own quadrant-gating export).

This is a read-only diagnostic script: it does not modify meta_dataset.csv.
It prints an agreement report so we can decide whether to replace/supplement
the manual labels with the ClusterData-derived ones.

Usage:
    python flowsom/compare_cluster_labels.py [--count-threshold 10]
"""
from __future__ import annotations

import argparse
import glob
import os
import re
from collections import defaultdict

import pandas as pd

HERE        = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT   = os.path.dirname(HERE)
DATA_ROOT   = os.path.join(REPO_ROOT, "ddPCR_data")
META_CSV    = os.path.join(HERE, "output", "meta_dataset.csv")

WELL_RE = re.compile(r"^[A-H](0[1-9]|1[0-2])$")


def build_amplitude_index(data_root: str) -> dict:
    """Map csv stem (without extension) -> full path, for every *_Amplitude.csv."""
    index = {}
    for path in glob.glob(os.path.join(data_root, "**", "*_Amplitude.csv"), recursive=True):
        stem = os.path.splitext(os.path.basename(path))[0]
        index[stem] = path
    return index


def find_cluster_data_csv(amplitude_csv_path: str) -> str | None:
    dir_ = os.path.dirname(amplitude_csv_path)
    matches = glob.glob(os.path.join(dir_, "*_ClusterData.csv"))
    return matches[0] if matches else None


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--count-threshold", type=int, default=10,
        help="Minimum droplet count for a quadrant to be treated as a real "
             "population rather than noise (default: 10).",
    )
    args = parser.parse_args()

    if not os.path.exists(META_CSV):
        raise FileNotFoundError(f"{META_CSV} not found.")

    df = pd.read_csv(META_CSV)
    print(f"Loaded meta_dataset.csv: {len(df)} rows.")

    print("Indexing *_Amplitude.csv files under ddPCR_data/ …")
    amplitude_index = build_amplitude_index(DATA_ROOT)
    print(f"  Indexed {len(amplitude_index)} amplitude files.")

    cluster_data_cache: dict = {}
    rows = []

    for _, row in df.iterrows():
        dataset = row["dataset"]
        tokens = dataset.split("_")
        well = tokens[-1] if WELL_RE.match(tokens[-1]) else (
            tokens[-2] if len(tokens) >= 2 and WELL_RE.match(tokens[-2]) else None
        )
        if well is None:
            rows.append({"dataset": dataset, "well": None, "status": "well_not_parsed"})
            continue

        amp_path = amplitude_index.get(dataset)
        if amp_path is None:
            rows.append({"dataset": dataset, "well": well, "status": "amplitude_csv_not_found"})
            continue

        cluster_csv = find_cluster_data_csv(amp_path)
        if cluster_csv is None:
            rows.append({"dataset": dataset, "well": well, "status": "no_cluster_data_csv"})
            continue

        if cluster_csv not in cluster_data_cache:
            cluster_data_cache[cluster_csv] = parse_quadrant_counts(cluster_csv)
        well_counts = cluster_data_cache[cluster_csv]

        counts = well_counts.get(well)
        if counts is None:
            rows.append({"dataset": dataset, "well": well, "status": "well_not_in_cluster_data"})
            continue

        true_n_clusters = sum(1 for c in counts if c >= args.count_threshold)
        rows.append({
            "dataset": dataset,
            "well": well,
            "status": "ok",
            "manual_best_n_clusters": row["best_n_clusters"],
            "true_n_clusters": true_n_clusters,
            "quadrant_counts": counts,
        })

    result = pd.DataFrame(rows)
    status_counts = result["status"].value_counts()
    print("\n=== Match status ===")
    print(status_counts.to_string())

    ok = result[result["status"] == "ok"].copy()
    if ok.empty:
        print("\nNo rows could be matched to a ClusterData.csv well entry.")
        return

    ok["agree"] = ok["manual_best_n_clusters"] == ok["true_n_clusters"]
    agreement_rate = ok["agree"].mean()
    print(f"\n=== Agreement (n={len(ok)}) ===")
    print(f"manual_best_n_clusters == true_n_clusters : {agreement_rate:.3f}")

    print("\nConfusion (rows = manual, cols = true):")
    confusion = pd.crosstab(ok["manual_best_n_clusters"], ok["true_n_clusters"])
    print(confusion.to_string())

    disagreements = ok[~ok["agree"]].sort_values("dataset")
    print(f"\n{len(disagreements)} disagreement(s). First 20:")
    print(
        disagreements[["dataset", "well", "manual_best_n_clusters", "true_n_clusters"]]
        .head(20)
        .to_string(index=False)
    )

    out_csv = os.path.join(HERE, "output", "cluster_label_comparison.csv")
    ok.drop(columns=["quadrant_counts"]).to_csv(out_csv, index=False)
    print(f"\nFull comparison saved -> {out_csv}")


if __name__ == "__main__":
    main()
