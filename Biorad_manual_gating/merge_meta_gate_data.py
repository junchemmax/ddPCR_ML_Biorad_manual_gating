import argparse
import os

import numpy as np
import pandas as pd


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
    derived = gate_df.apply(
        lambda row: pd.Series(
            {
                "gate_has_x": int(pd.notna(row["x_gate"])),
                "gate_has_y": int(pd.notna(row["y_gate"])),
                "gate_n_cluster_rows": len(parse_pipe_numbers(row["cluster_id"])),
                "gate_total_cluster_count": sum(parse_pipe_numbers(row["count"])),
                "gate_ch1_mean_weighted": weighted_mean(
                    parse_pipe_numbers(row["ch1_mean"]),
                    parse_pipe_numbers(row["count"]),
                ),
                "gate_ch2_mean_weighted": weighted_mean(
                    parse_pipe_numbers(row["ch2_mean"]),
                    parse_pipe_numbers(row["count"]),
                ),
                "gate_ch2_range": float(row["x_max_ch2"]) - float(row["x_min_ch2"]),
                "gate_ch1_range": float(row["y_max_ch1"]) - float(row["y_min_ch1"]),
            }
        ),
        axis=1,
    )
    gate_df = gate_df.copy()
    for column in [
        "x_gate", "y_gate", "x_min_ch2", "x_max_ch2", "y_min_ch1", "y_max_ch1"
    ]:
        gate_df[column] = pd.to_numeric(gate_df[column], errors="coerce")
    derived["gate_ch2_range"] = gate_df["x_max_ch2"] - gate_df["x_min_ch2"]
    derived["gate_ch1_range"] = gate_df["y_max_ch1"] - gate_df["y_min_ch1"]
    return pd.concat([gate_df, derived], axis=1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge metadata features with Bio-Rad gate summaries.")
    parser.add_argument("--meta", default=os.path.join("output", "meta_dataset.csv"))
    parser.add_argument("--gate", default=os.path.join("output", "biorad_cluster_gate_data.csv"))
    parser.add_argument("--out", default=os.path.join("output", "meta_biorad_cluster_dataset.csv"))
    args = parser.parse_args()

    meta_df = pd.read_csv(args.meta)
    gate_df = pd.read_csv(args.gate)
    gate_df = add_gate_features(gate_df)

    meta_df["source_amplitude_csv"] = meta_df["dataset"].astype(str) + ".csv"
    merged = meta_df.merge(
        gate_df,
        on="source_amplitude_csv",
        how="left",
        suffixes=("", "_gate"),
        validate="one_to_one",
    )
    merged.insert(0, "merge_status", np.where(merged["well"].notna(), "matched", "unmatched"))
    merged.to_csv(args.out, index=False)

    print(f"Saved merged dataset: {args.out}")
    print(f"Metadata rows: {len(meta_df)}")
    print(f"Matched rows: {int((merged['merge_status'] == 'matched').sum())}")
    print(f"Output columns: {len(merged.columns)}")


if __name__ == "__main__":
    main()