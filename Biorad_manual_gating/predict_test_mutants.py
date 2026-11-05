"""Compare saved-model mutant rankings with predicted gates and existing manual labels."""
import argparse
import hashlib
import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from build_biorad_gate_dataset import extract_features, manual_label_columns, normalize_manual_label
from mutant_well_analysis import features_from_gate_data
from train_and_predict_gate_model import assign_quadrants
from train_mutant_candidate_model import FEATURE_COLUMNS, add_model_features

PROJECT_DIR = Path(__file__).resolve().parents[1]
QUADRANTS = ["Q1", "Q2", "Q3", "Q4"]
IDENTIFIERS = ["dataset", "parent_dataset", "well"]


def plate_identity(path: Path) -> tuple[str, str]:
    match = re.fullmatch(r"(.+)_([A-H](?:0[1-9]|1[0-2]))_Amplitude", path.stem)
    if match is None:
        raise ValueError(f"Cannot identify plate and well from {path}")
    return match.group(1), match.group(2)


def read_amplitudes(path: Path) -> tuple[pd.DataFrame, np.ndarray, int]:
    frame = pd.read_csv(path, skiprows=3, low_memory=False)
    amplitude = frame[["Ch1Amplitude", "Ch2Amplitude"]].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(dtype=float)
    valid = np.isfinite(amplitude).all(axis=1)
    return frame.loc[valid].copy(), amplitude[valid], int((~valid).sum())


def manual_quadrants(frame: pd.DataFrame) -> tuple[np.ndarray, int]:
    columns = manual_label_columns(frame)
    if len(columns) != 2:
        raise ValueError("Expected two manual classification columns")
    mut = frame[columns[0]].map(normalize_manual_label).to_numpy()
    wt = frame[columns[1]].map(normalize_manual_label).to_numpy()
    valid = np.isin(mut, ["0", "1"]) & np.isin(wt, ["0", "1"])
    quadrants = np.where(mut[valid] == "1", np.where(wt[valid] == "1", 1, 2),
                         np.where(wt[valid] == "1", 4, 3))
    return quadrants, int((~valid).sum())


def count_summary(quadrants: np.ndarray) -> dict[str, str]:
    counts = np.bincount(quadrants, minlength=5)[1:5]
    return {"quadrant": "|".join(QUADRANTS), "count": "|".join(map(str, counts))}


def amplitude_features(amplitude: np.ndarray) -> dict[str, float]:
    ch1, ch2 = amplitude[:, 0], amplitude[:, 1]
    return {
        **extract_features(amplitude),
        "x_min_ch2": float(ch2.min()), "x_max_ch2": float(ch2.max()),
        "y_min_ch1": float(ch1.min()), "y_max_ch1": float(ch1.max()),
        "amplitude_range_WT": float(np.ptp(ch2)),
        "amplitude_range_MUT": float(np.ptp(ch1)),
    }


def rank_wells(summary: pd.DataFrame, model: Pipeline) -> pd.DataFrame:
    wells = add_model_features(features_from_gate_data(summary))
    scores = model.decision_function(wells[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan))
    if not np.isfinite(scores).all():
        raise ValueError("Mutant model returned nonfinite scores")
    wells["model_score"] = scores
    wells["model_score_for_ranking"] = wells["model_score"].mask(wells["poisson_red_flag"], -1e6)
    wells["candidate_rank"] = wells.groupby("parent_dataset")["model_score_for_ranking"].rank(
        ascending=False, method="min"
    )
    wells["no_mutant_positive_droplets"] = wells["mut_pos"] == 0
    wells["no_wt_positive_droplets"] = wells["wt_pos"] == 0
    metadata = summary.drop(columns=["quadrant", "count"])
    wells = wells.merge(metadata, on=IDENTIFIERS, validate="one_to_one")
    return wells.sort_values(["project", "parent_dataset", "candidate_rank", "well"])


def top_candidates(wells: pd.DataFrame) -> pd.DataFrame:
    return wells[(wells["candidate_rank"] <= 3) & ~wells["poisson_red_flag"]].copy()


def compare_plates(first: pd.DataFrame, second: pd.DataFrame) -> pd.DataFrame:
    comparisons = []
    for (project, plate), a in first.groupby(["project", "parent_dataset"]):
        b = second[second["parent_dataset"] == plate]
        a_top = set(a.loc[(a["candidate_rank"] == 1) & ~a["poisson_red_flag"], "well"])
        b_top = set(b.loc[(b["candidate_rank"] == 1) & ~b["poisson_red_flag"], "well"])
        comparisons.append({
            "project": project, "parent_dataset": plate,
            "trial1_evaluated_wells": len(a), "trial2_evaluated_wells": len(b),
            "trial1_top_wells": "|".join(sorted(a_top)),
            "trial2_top_wells": "|".join(sorted(b_top)),
            "top_candidate_agreement": bool(a_top and a_top == b_top),
            "trial1_flagged_wells": int(a["poisson_red_flag"].sum()),
            "trial2_flagged_wells": int(b["poisson_red_flag"].sum()),
        })
    return pd.DataFrame(comparisons)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", nargs="?", type=Path, default=PROJECT_DIR / "test_data")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "test_mutant_prediction_output")
    args = parser.parse_args()
    gate_path = PROJECT_DIR / "gating_model_output" / "biorad_gate_model.joblib"
    gate_metadata_path = gate_path.with_name("biorad_gate_model_metrics.json")
    mutant_path = PROJECT_DIR / "mut_prediction_model_output" / "mutant_candidate_ranker.joblib"
    gate_model = joblib.load(gate_path)
    mutant_model = joblib.load(mutant_path)
    gate_columns = json.loads(gate_metadata_path.read_text(encoding="utf-8"))["feature_columns"]
    if list(gate_model.feature_names_in_) != gate_columns:
        raise ValueError("Gate model and feature metadata disagree")
    if list(mutant_model.feature_names_in_) != FEATURE_COLUMNS:
        raise ValueError("Mutant model and feature definitions disagree")
    paths = sorted(args.input_dir.rglob("*_Amplitude.csv"))
    if not paths:
        raise ValueError(f"No amplitude CSVs in {args.input_dir}")
    records, feature_rows, failures = [], [], []
    for index, path in enumerate(paths, 1):
        plate, well = plate_identity(path)
        project = next((p.name for p in path.parents if re.fullmatch(r"MIP_\d+", p.name)), None)
        if project is None:
            raise ValueError(f"Cannot identify MIP project for {path}")
        frame, amplitude, excluded = read_amplitudes(path)
        identity = {"project": project, "dataset": path.stem, "parent_dataset": plate, "well": well}
        if not len(amplitude):
            failures.append({**identity, "trial": "both", "reason": "No finite amplitude observations"})
            continue
        quadrants, unclassified = manual_quadrants(frame)
        records.append({
            **identity, "source_amplitude_csv": str(path.resolve()),
            "n_droplets": len(amplitude), "excluded_amplitude_rows": excluded,
            "unclassified_manual_rows": unclassified, **count_summary(quadrants),
        })
        feature_rows.append(amplitude_features(amplitude))
        if index % 500 == 0:
            print(f"Extracted {index}/{len(paths)} files", flush=True)
    summary = pd.DataFrame(records)
    if summary.empty:
        raise ValueError("No usable amplitude files")
    if summary.duplicated(["parent_dataset", "well"]).any():
        raise ValueError("Duplicate plate/well identities in test data")
    features = pd.DataFrame(feature_rows)
    predictions = gate_model.predict(features[gate_columns])
    if not np.isfinite(predictions).all():
        raise ValueError("Gate model returned nonfinite coordinates")
    predicted_summary = summary.copy()
    predicted_summary["predicted_x_gate"] = predictions[:, 0]
    predicted_summary["predicted_y_gate"] = predictions[:, 1]
    agreements = []
    for index, row in predicted_summary.iterrows():
        frame, amplitude, _ = read_amplitudes(Path(row["source_amplitude_csv"]))
        predicted = assign_quadrants(amplitude, row["predicted_x_gate"], row["predicted_y_gate"])
        counts = count_summary(predicted)
        for column, value in counts.items():
            predicted_summary.loc[index, column] = value
        columns = manual_label_columns(frame)
        labeled = frame[columns].apply(lambda col: col.map(normalize_manual_label))
        valid = labeled.isin(["0", "1"]).all(axis=1).to_numpy()
        reference, _ = manual_quadrants(frame)
        agreements.append(float(100 * np.mean(predicted[valid] == reference)) if len(reference) else np.nan)
        if (index + 1) % 500 == 0:
            print(f"Gated {index + 1}/{len(summary)} wells", flush=True)
    predicted_summary["quadrant_agreement_with_manual_pct"] = agreements
    eligible_manual = (summary["n_droplets"] - summary["unclassified_manual_rows"]) > 0
    for row in summary.loc[~eligible_manual].to_dict("records"):
        failures.append({**{key: row[key] for key in ["project", *IDENTIFIERS]},
                         "trial": "trial2", "reason": "No droplets with both manual labels classified"})
    first = rank_wells(predicted_summary, mutant_model)
    second = rank_wells(summary.loc[eligible_manual].copy(), mutant_model)
    comparison = compare_plates(first, second)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, result in [("trial1_predicted_gates", first), ("trial2_existing_labels", second)]:
        result.to_csv(args.output_dir / f"{name}_all_wells.csv", index=False)
        top_candidates(result).to_csv(args.output_dir / f"{name}_top_candidates.csv", index=False)
    comparison.to_csv(args.output_dir / "plate_comparison.csv", index=False)
    pd.DataFrame(failures, columns=["project", *IDENTIFIERS, "trial", "reason"]).to_csv(
        args.output_dir / "excluded_wells.csv", index=False)
    project_summaries = []
    for project, a in first.groupby("project"):
        b = second[second["project"] == project]
        plates = comparison[comparison["project"] == project]
        project_summaries.append({
            "project": project, "plates": len(plates),
            "trial1_wells": len(a), "trial2_wells": len(b),
            "same_top_candidate_plates": int(plates["top_candidate_agreement"].sum()),
            "mean_well_quadrant_agreement_with_manual_pct": float(
                a["quadrant_agreement_with_manual_pct"].mean()),
            "trial1_mut_positive_droplets": int(a["mut_pos"].sum()),
            "trial2_mut_positive_droplets": int(b["mut_pos"].sum()),
            "trial1_poisson_flagged_wells": int(a["poisson_red_flag"].sum()),
            "trial2_poisson_flagged_wells": int(b["poisson_red_flag"].sum()),
        })
    pd.DataFrame(project_summaries).to_csv(args.output_dir / "project_comparison.csv", index=False)
    training_ids = set(pd.read_csv(
        PROJECT_DIR / "data_output" / "full_biorad_cluster_dataset.csv", usecols=["dataset"]
    )["dataset"])
    metadata = {
        "input_directory": str(args.input_dir.resolve()), "input_csvs": len(paths),
        "trial1_wells": len(first), "trial2_wells": len(second),
        "plates": len(comparison), "excluded_records": failures,
        "plates_with_same_top_candidates": int(comparison["top_candidate_agreement"].sum()),
        "projects": project_summaries,
        "test_well_ids_also_in_training": len(set(summary["dataset"]) & training_ids),
        "mean_well_quadrant_agreement_with_manual_pct": float(np.nanmean(agreements)),
        "model_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in [gate_path, mutant_path]},
        "notes": [
            "Saved models used for inference only; no training or retraining.",
            "Trial 1 uses only amplitudes to predict gates and quadrant counts.",
            "Trial 2 uses the existing manual labels directly; u/unknown labels are excluded.",
            "Same plate-background features and red-flag suppression as mutant model training.",
            "Ranks are not probabilities or confirmed mutant calls; no test ground truth supplied.",
            "All rank ties are preserved. Top candidates means unflagged rank <= 3.",
            "mutant_call is the existing heuristic call, not the trained model prediction.",
            "A rank-1 well can have no MUT or no WT signal: inspect both no_*_positive_droplets columns.",
            "High overall quadrant agreement can conceal large changes in rare MUT-positive counts.",
        ],
    }
    (args.output_dir / "run_summary.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(comparison.to_string(index=False))
    print(f"Saved results to {args.output_dir}", flush=True)
    print(f"Excluded records: {len(failures)}")


if __name__ == "__main__":
    main()
