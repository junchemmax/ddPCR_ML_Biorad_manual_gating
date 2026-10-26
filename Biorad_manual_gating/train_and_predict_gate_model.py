"""Train, predict, and evaluate the Bio-Rad gate model.

Metrics come from a model trained on a plate-grouped 80/20 split; the saved model is then
refit on all eligible wells. Reference labels are reconstructed gates, not original saved
Bio-Rad UI coordinates.
All-well predictions include training wells and are not independent test results.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_DIR / "data_output" / "full_biorad_cluster_dataset.csv"
OUTPUT_DIR = PROJECT_DIR / "gating_model_output"
MODEL_PATH = OUTPUT_DIR / "biorad_gate_model.joblib"
METRICS_PATH = OUTPUT_DIR / "biorad_gate_model_metrics.json"
TEST_PREDICTIONS_PATH = OUTPUT_DIR / "biorad_gate_model_predictions.csv"
IMPORTANCE_PATH = OUTPUT_DIR / "biorad_gate_feature_importance.csv"
PREDICTION_OUTPUT_PATH = OUTPUT_DIR / "biorad_gate_predictions_all_wells.csv"
TEST_SIZE = 0.2
RANDOM_STATE = 42

TARGET_COLUMNS = ["x_gate", "y_gate"]
GROUP_COLUMN = "parent_dataset"
IDENTIFIER_COLUMNS = ["dataset", "parent_dataset", "well"]
LEAKED_COLUMNS = {
    "x_gate", "y_gate", "target1(MUT)", "target2(WT)", "quadrant",
    "count", "ch1_mean", "ch2_mean", "gate_has_x", "gate_has_y",
    "gate_n_cluster_rows", "gate_total_cluster_count",
    "gate_ch1_mean_weighted", "gate_ch2_mean_weighted",
    "n_populated_quadrants", "weighted_mean_ch1_MUT", "weighted_mean_ch2_WT",
}


def show(frame: pd.DataFrame) -> None:
    print(frame.to_string(), end="\n\n")


def make_model():
    return make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestRegressor(
            n_estimators=400, min_samples_leaf=3, max_features=0.8,
            random_state=RANDOM_STATE, n_jobs=-1,
        ),
    )


def select_feature_columns(data: pd.DataFrame) -> list[str]:
    excluded = LEAKED_COLUMNS | set(IDENTIFIER_COLUMNS)
    return [column for column in data.columns
            if column not in excluded and pd.api.types.is_numeric_dtype(data[column])]


def load_training_data() -> pd.DataFrame:
    data = pd.read_csv(DATA_PATH)
    missing_columns = [column for column in TARGET_COLUMNS + IDENTIFIER_COLUMNS if column not in data]
    if missing_columns:
        raise ValueError(f"Required columns are missing: {missing_columns}")
    input_rows = len(data)
    data[TARGET_COLUMNS] = data[TARGET_COLUMNS].apply(pd.to_numeric, errors="coerce")
    data = data.dropna(subset=TARGET_COLUMNS + [GROUP_COLUMN]).copy()
    print(f"Input wells: {input_rows}; eligible wells: {len(data)}; dropped: {input_rows - len(data)}")
    for column in ["x_gate_method", "y_gate_method"]:
        if column in data:
            show(data[column].value_counts(dropna=False).rename_axis(column).to_frame("wells"))
    return data


def train_and_evaluate(data: pd.DataFrame) -> list[str]:
    feature_columns = select_feature_columns(data)
    if not feature_columns:
        raise ValueError("No numeric, non-leaking feature columns were found.")
    print(f"Selected {len(feature_columns)} features: {feature_columns}")

    groups = data[GROUP_COLUMN].astype(str)
    if groups.nunique() < 2:
        raise ValueError("At least two parent_dataset groups are required for evaluation.")
    splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_idx, test_idx = next(splitter.split(data[feature_columns], data[TARGET_COLUMNS], groups))
    assert set(groups.iloc[train_idx]).isdisjoint(set(groups.iloc[test_idx]))
    show(pd.DataFrame({
        "split": ["train", "test"],
        "rows": [len(train_idx), len(test_idx)],
        "groups": [groups.iloc[train_idx].nunique(), groups.iloc[test_idx].nunique()],
    }))

    model = make_model()
    model.fit(data.iloc[train_idx][feature_columns], data.iloc[train_idx][TARGET_COLUMNS])
    test_predictions = model.predict(data.iloc[test_idx][feature_columns])

    test_targets = data.iloc[test_idx][TARGET_COLUMNS].to_numpy()
    baseline = np.tile(data.iloc[train_idx][TARGET_COLUMNS].median().to_numpy(), (len(test_idx), 1))
    metrics = {
        "data_path": str(DATA_PATH), "feature_columns": feature_columns,
        "target_columns": TARGET_COLUMNS, "group_column": GROUP_COLUMN,
        "train_rows": int(len(train_idx)), "test_rows": int(len(test_idx)),
        "train_groups": int(groups.iloc[train_idx].nunique()),
        "test_groups": int(groups.iloc[test_idx].nunique()),
    }
    for index, target in enumerate(TARGET_COLUMNS):
        metrics[f"model_mae_{target}"] = float(mean_absolute_error(test_targets[:, index], test_predictions[:, index]))
        metrics[f"model_rmse_{target}"] = float(mean_squared_error(test_targets[:, index], test_predictions[:, index]) ** 0.5)
        metrics[f"baseline_mae_{target}"] = float(mean_absolute_error(test_targets[:, index], baseline[:, index]))

    predictions_df = data.iloc[test_idx][IDENTIFIER_COLUMNS + TARGET_COLUMNS].copy()
    predictions_df["predicted_x_gate"] = test_predictions[:, 0]
    predictions_df["predicted_y_gate"] = test_predictions[:, 1]
    predictions_df["x_gate_error"] = predictions_df["predicted_x_gate"] - predictions_df["x_gate"]
    predictions_df["y_gate_error"] = predictions_df["predicted_y_gate"] - predictions_df["y_gate"]

    final_model = make_model()
    final_model.fit(data[feature_columns], data[TARGET_COLUMNS])
    metrics["final_model_train_rows"] = int(len(data))
    metrics["final_model_train_groups"] = int(groups.nunique())
    print(f"Final model refit on all {len(data)} wells from {groups.nunique()} plates")
    forest = final_model.named_steps["randomforestregressor"]
    importance_df = pd.DataFrame({"feature": feature_columns, "importance": forest.feature_importances_}).sort_values("importance", ascending=False)
    show(importance_df.head(15))

    joblib.dump(final_model, MODEL_PATH)
    with METRICS_PATH.open("w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=2)
    predictions_df.to_csv(TEST_PREDICTIONS_PATH, index=False)
    importance_df.to_csv(IMPORTANCE_PATH, index=False)
    for path in [MODEL_PATH, METRICS_PATH, TEST_PREDICTIONS_PATH, IMPORTANCE_PATH]:
        print(f"Saved: {path}")
    for target in TARGET_COLUMNS:
        print(f"{target} MAE: {metrics[f'model_mae_{target}']:.3f}; baseline: {metrics[f'baseline_mae_{target}']:.3f}")
    return feature_columns


def predict_all_wells() -> None:
    inference_data = pd.read_csv(DATA_PATH)
    with METRICS_PATH.open("r", encoding="utf-8") as metrics_file:
        inference_features = json.load(metrics_file)["feature_columns"]
    missing_columns = [column for column in inference_features if column not in inference_data]
    if missing_columns:
        raise ValueError(f"Required feature columns are missing: {missing_columns}")
    batch_predictions = joblib.load(MODEL_PATH).predict(inference_data[inference_features])
    assert np.isfinite(batch_predictions).all()
    result = inference_data.copy()
    result["predicted_x_gate"] = batch_predictions[:, 0]
    result["predicted_y_gate"] = batch_predictions[:, 1]
    result.to_csv(PREDICTION_OUTPUT_PATH, index=False)
    print(f"Saved predictions: {PREDICTION_OUTPUT_PATH}; predicted wells: {len(result)}")


def assign_quadrants(amplitude: np.ndarray, x_gate: float, y_gate: float) -> np.ndarray:
    mut_positive = amplitude[:, 0] > y_gate
    wt_positive = amplitude[:, 1] > x_gate
    return np.where(mut_positive, np.where(wt_positive, 1, 2), np.where(wt_positive, 4, 3))


def compare_droplets(amplitude: np.ndarray, reference_x: float, reference_y: float,
                     predicted_x: float, predicted_y: float) -> tuple[dict, np.ndarray]:
    reference = assign_quadrants(amplitude, reference_x, reference_y)
    predicted = assign_quadrants(amplitude, predicted_x, predicted_y)
    confusion = np.zeros((4, 4), dtype=np.int64)
    np.add.at(confusion, (reference - 1, predicted - 1), 1)
    summary = {
        "n_droplets": len(amplitude),
        "changed_droplets": int(np.sum(reference != predicted)),
        "quadrant_agreement_pct": float(100 * np.mean(reference == predicted)),
    }
    for quadrant in range(1, 5):
        reference_count = int(np.sum(reference == quadrant))
        predicted_count = int(np.sum(predicted == quadrant))
        summary[f"reference_Q{quadrant}_count"] = reference_count
        summary[f"predicted_Q{quadrant}_count"] = predicted_count
        summary[f"Q{quadrant}_count_change"] = predicted_count - reference_count
    for target, quadrants in [("MUT", [1, 2]), ("WT", [1, 4])]:
        reference_positive = np.isin(reference, quadrants)
        predicted_positive = np.isin(predicted, quadrants)
        reference_fraction = float(reference_positive.mean())
        predicted_fraction = float(predicted_positive.mean())
        summary[f"reference_{target}_positive_count"] = int(reference_positive.sum())
        summary[f"predicted_{target}_positive_count"] = int(predicted_positive.sum())
        summary[f"reference_{target}_positive_fraction"] = reference_fraction
        summary[f"predicted_{target}_positive_fraction"] = predicted_fraction
        summary[f"{target}_positive_shift_pp"] = 100 * (predicted_fraction - reference_fraction)
        summary[f"{target}_changed_droplets"] = int(np.sum(reference_positive != predicted_positive))
        # Poisson occupancy is unbounded when every droplet is positive.
        reference_occupancy = float(-np.log1p(-reference_fraction)) if reference_fraction < 1 else np.nan
        predicted_occupancy = float(-np.log1p(-predicted_fraction)) if predicted_fraction < 1 else np.nan
        summary[f"reference_{target}_copies_per_droplet"] = reference_occupancy
        summary[f"predicted_{target}_copies_per_droplet"] = predicted_occupancy
        summary[f"{target}_copies_per_droplet_change"] = predicted_occupancy - reference_occupancy
    return summary, confusion


def self_test() -> None:
    synthetic_amplitude = np.array([[20., 20.], [20., 0.], [0., 0.], [0., 20.], [10., 10.]])
    assert assign_quadrants(synthetic_amplitude, 10., 10.).tolist() == [1, 2, 3, 4, 3]
    same, _ = compare_droplets(synthetic_amplitude, 10., 10., 10., 10.)
    assert same["changed_droplets"] == 0 and same["quadrant_agreement_pct"] == 100
    changed, changed_confusion = compare_droplets(synthetic_amplitude, 10., 10., 30., 10.)
    assert changed["changed_droplets"] == 2
    assert changed["MUT_positive_shift_pp"] == 0 and np.isclose(changed["WT_positive_shift_pp"], -40)
    assert changed_confusion[0, 1] == 1 and changed_confusion[3, 2] == 1


def evaluate_heldout_droplets() -> None:
    heldout = pd.read_csv(TEST_PREDICTIONS_PATH)
    reference_data = pd.read_csv(DATA_PATH)
    label_columns = IDENTIFIER_COLUMNS + ["x_gate_method", "y_gate_method"]
    heldout = heldout.merge(reference_data[label_columns], on=IDENTIFIER_COLUMNS, how="left", validate="one_to_one")
    amplitude_index: dict[tuple[str, str], list[Path]] = {}
    for amplitude_path in (PROJECT_DIR / "ddPCR_data").rglob("*_Amplitude.csv"):
        amplitude_index.setdefault((amplitude_path.parent.name, amplitude_path.stem), []).append(amplitude_path)

    well_rows, failed_rows = [], []
    pooled_confusion = np.zeros((4, 4), dtype=np.int64)
    for position, row in enumerate(heldout.to_dict("records"), start=1):
        identity = {column: row[column] for column in IDENTIFIER_COLUMNS}
        candidates = amplitude_index.get((str(row["parent_dataset"]), str(row["dataset"])), [])
        if len(candidates) != 1:
            failed_rows.append({**identity, "reason": f"Expected one amplitude CSV, found {len(candidates)}"})
            continue
        try:
            gates = np.array([row[column] for column in ["x_gate", "y_gate", "predicted_x_gate", "predicted_y_gate"]], dtype=float)
            if not np.isfinite(gates).all():
                raise ValueError("Nonfinite gate coordinates")
            frame = pd.read_csv(candidates[0], skiprows=3, low_memory=False)
            amplitude = np.asarray(frame[["Ch1Amplitude", "Ch2Amplitude"]].apply(pd.to_numeric, errors="coerce"), dtype=float)
            valid = np.isfinite(amplitude).all(axis=1)
            amplitude = amplitude[valid]
            if not len(amplitude):
                raise ValueError("No finite amplitude observations")
            summary, confusion = compare_droplets(amplitude, *gates)
        except (OSError, ValueError, KeyError, pd.errors.ParserError) as error:
            failed_rows.append({**identity, "reason": str(error)})
            continue
        well_rows.append({
            **identity, "x_gate_method": row["x_gate_method"], "y_gate_method": row["y_gate_method"],
            "x_gate": gates[0], "y_gate": gates[1], "predicted_x_gate": gates[2], "predicted_y_gate": gates[3],
            "excluded_amplitude_rows": int((~valid).sum()), **summary,
        })
        pooled_confusion += confusion
        if position % 250 == 0:
            print(f"Evaluated {position}/{len(heldout)} held-out wells")

    well_comparison = pd.DataFrame(well_rows)
    if well_comparison.empty:
        raise ValueError("No held-out wells could be evaluated; inspect failures CSV.")
    well_comparison["changed_fraction"] = well_comparison["changed_droplets"] / well_comparison["n_droplets"]
    well_comparison["reference_gate_type"] = np.where(
        (well_comparison[["x_gate_method", "y_gate_method"]] == "cluster_means").all(axis=1),
        "both_cluster_means", "includes_estimate_or_unknown",
    )
    experiment_rows = []
    for experiment, wells in well_comparison.groupby("parent_dataset"):
        droplets = int(wells["n_droplets"].sum())
        summary = {
            "parent_dataset": experiment, "evaluated_wells": len(wells), "n_droplets": droplets,
            "changed_droplets": int(wells["changed_droplets"].sum()),
            "pooled_quadrant_agreement_pct": 100 * (1 - wells["changed_droplets"].sum() / droplets),
            "mean_well_quadrant_agreement_pct": wells["quadrant_agreement_pct"].mean(),
            "worst_well_quadrant_agreement_pct": wells["quadrant_agreement_pct"].min(),
        }
        for target in ["MUT", "WT"]:
            summary[f"mean_abs_{target}_positive_shift_pp"] = wells[f"{target}_positive_shift_pp"].abs().mean()
            summary[f"pooled_{target}_positive_shift_pp"] = 100 * (
                wells[f"predicted_{target}_positive_count"].sum() - wells[f"reference_{target}_positive_count"].sum()
            ) / droplets
            summary[f"mean_abs_{target}_copies_per_droplet_change"] = wells[f"{target}_copies_per_droplet_change"].abs().mean()
        experiment_rows.append(summary)
    experiment_comparison = pd.DataFrame(experiment_rows).sort_values("pooled_quadrant_agreement_pct")
    confusion_df = pd.DataFrame(pooled_confusion, index=["Q1", "Q2", "Q3", "Q4"], columns=["Q1", "Q2", "Q3", "Q4"])
    confusion_df.index.name = "reference_quadrant"
    confusion_df.columns.name = "predicted_quadrant"
    well_comparison.to_csv(OUTPUT_DIR / "heldout_droplet_comparison.csv", index=False)
    experiment_comparison.to_csv(OUTPUT_DIR / "heldout_experiment_comparison.csv", index=False)
    confusion_df.to_csv(OUTPUT_DIR / "heldout_quadrant_confusion.csv")
    pd.DataFrame(failed_rows, columns=IDENTIFIER_COLUMNS + ["reason"]).to_csv(
        OUTPUT_DIR / "heldout_droplet_evaluation_failures.csv", index=False)
    assert len(well_comparison) + len(failed_rows) == len(heldout)
    assert pooled_confusion.sum() == well_comparison["n_droplets"].sum()

    total_droplets = int(well_comparison["n_droplets"].sum())
    evaluation_summary = {
        "evaluated_wells": len(well_comparison), "failed_wells": len(failed_rows),
        "evaluated_droplets": total_droplets,
        "pooled_quadrant_agreement_pct": float(100 * np.trace(pooled_confusion) / total_droplets),
        "median_well_quadrant_agreement_pct": float(well_comparison["quadrant_agreement_pct"].median()),
        "p10_well_quadrant_agreement_pct": float(well_comparison["quadrant_agreement_pct"].quantile(0.1)),
        "wells_at_least_99pct_agreement": int((well_comparison["quadrant_agreement_pct"] >= 99).sum()),
    }
    for target in ["MUT", "WT"]:
        shifts = well_comparison[f"{target}_positive_shift_pp"].abs()
        evaluation_summary[f"mean_abs_{target}_positive_shift_pp"] = float(shifts.mean())
        evaluation_summary[f"p95_abs_{target}_positive_shift_pp"] = float(shifts.quantile(0.95))
        evaluation_summary[f"max_abs_{target}_positive_shift_pp"] = float(shifts.max())
        evaluation_summary[f"{target}_wells_unbounded_occupancy"] = int(
            ((well_comparison[f"reference_{target}_positive_fraction"] == 1)
             | (well_comparison[f"predicted_{target}_positive_fraction"] == 1)).sum()
        )
    with (OUTPUT_DIR / "heldout_droplet_evaluation_summary.json").open("w", encoding="utf-8") as summary_file:
        json.dump(evaluation_summary, summary_file, indent=2)
    print(json.dumps(evaluation_summary, indent=2))
    show(confusion_df)
    show(well_comparison.groupby("reference_gate_type")["quadrant_agreement_pct"].agg(["count", "mean", "median", "min"]))
    worst_columns = IDENTIFIER_COLUMNS + ["quadrant_agreement_pct", "MUT_positive_shift_pp", "WT_positive_shift_pp"]
    show(well_comparison.sort_values("quadrant_agreement_pct")[worst_columns].head(10))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Project: {PROJECT_DIR}\nOutput: {OUTPUT_DIR}")
    self_test()
    train_and_evaluate(load_training_data())
    predict_all_wells()
    evaluate_heldout_droplets()


if __name__ == "__main__":
    main()
