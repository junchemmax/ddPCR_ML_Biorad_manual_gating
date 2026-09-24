import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline


TARGET_COLUMNS = ["x_gate", "y_gate"]
GROUP_COLUMN = "parent_dataset"
IDENTIFIER_COLUMNS = ["dataset", "parent_dataset", "well"]
LEAKED_COLUMNS = {
    "x_gate",
    "y_gate",
    "target1(MUT)",
    "target2(WT)",
    "quadrant",
    "count",
    "ch1_mean",
    "ch2_mean",
    "gate_has_x",
    "gate_has_y",
    "gate_n_cluster_rows",
    "gate_total_cluster_count",
    "gate_ch1_mean_weighted",
    "gate_ch2_mean_weighted",
}


def default_data_path() -> str:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    candidates = [
        os.path.join(project_dir, "output", "full_biorad_cluster_dataset.csv"),
        os.path.join(script_dir, "output", "full_biorad_cluster_dataset.csv"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[0]


def select_feature_columns(data: pd.DataFrame) -> list[str]:
    excluded = LEAKED_COLUMNS | set(IDENTIFIER_COLUMNS)
    return [
        column
        for column in data.columns
        if column not in excluded and pd.api.types.is_numeric_dtype(data[column])
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a leakage-resistant model to predict Bio-Rad x/y gates."
    )
    parser.add_argument("--data-path", default=default_data_path())
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    data_path = os.path.abspath(args.data_path)
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Feature dataset not found: {data_path}")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    output_dir = os.path.abspath(args.output_dir or os.path.join(project_dir, "output"))
    os.makedirs(output_dir, exist_ok=True)

    data = pd.read_csv(data_path)
    missing_columns = [column for column in TARGET_COLUMNS + [GROUP_COLUMN] if column not in data]
    if missing_columns:
        raise ValueError(f"Required columns are missing: {', '.join(missing_columns)}")

    data[TARGET_COLUMNS] = data[TARGET_COLUMNS].apply(pd.to_numeric, errors="coerce")
    data = data.dropna(subset=TARGET_COLUMNS + [GROUP_COLUMN]).copy()
    feature_columns = select_feature_columns(data)
    if not feature_columns:
        raise ValueError("No numeric, non-leaking feature columns were found.")

    groups = data[GROUP_COLUMN].astype(str)
    if groups.nunique() < 2:
        raise ValueError("At least two parent_dataset groups are required for evaluation.")

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    train_idx, test_idx = next(splitter.split(data[feature_columns], data[TARGET_COLUMNS], groups))

    model = make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestRegressor(
            n_estimators=400,
            min_samples_leaf=3,
            max_features=0.8,
            random_state=args.random_state,
            n_jobs=-1,
        ),
    )
    model.fit(data.iloc[train_idx][feature_columns], data.iloc[train_idx][TARGET_COLUMNS])
    predictions = model.predict(data.iloc[test_idx][feature_columns])

    test_targets = data.iloc[test_idx][TARGET_COLUMNS].to_numpy()
    baseline = np.tile(data.iloc[train_idx][TARGET_COLUMNS].median().to_numpy(), (len(test_idx), 1))
    metrics = {
        "data_path": data_path,
        "feature_columns": feature_columns,
        "target_columns": TARGET_COLUMNS,
        "group_column": GROUP_COLUMN,
        "train_rows": int(len(train_idx)),
        "test_rows": int(len(test_idx)),
        "train_groups": int(groups.iloc[train_idx].nunique()),
        "test_groups": int(groups.iloc[test_idx].nunique()),
        "model_mae_x_gate": float(mean_absolute_error(test_targets[:, 0], predictions[:, 0])),
        "model_mae_y_gate": float(mean_absolute_error(test_targets[:, 1], predictions[:, 1])),
        "model_rmse_x_gate": float(mean_squared_error(test_targets[:, 0], predictions[:, 0]) ** 0.5),
        "model_rmse_y_gate": float(mean_squared_error(test_targets[:, 1], predictions[:, 1]) ** 0.5),
        "baseline_mae_x_gate": float(mean_absolute_error(test_targets[:, 0], baseline[:, 0])),
        "baseline_mae_y_gate": float(mean_absolute_error(test_targets[:, 1], baseline[:, 1])),
    }

    predictions_df = data.iloc[test_idx][IDENTIFIER_COLUMNS + TARGET_COLUMNS].copy()
    predictions_df["predicted_x_gate"] = predictions[:, 0]
    predictions_df["predicted_y_gate"] = predictions[:, 1]
    predictions_df["x_gate_error"] = predictions_df["predicted_x_gate"] - predictions_df["x_gate"]
    predictions_df["y_gate_error"] = predictions_df["predicted_y_gate"] - predictions_df["y_gate"]

    forest = model.named_steps["randomforestregressor"]
    importance_df = pd.DataFrame({
        "feature": feature_columns,
        "importance": forest.feature_importances_,
    }).sort_values("importance", ascending=False)

    model_path = os.path.join(output_dir, "biorad_gate_model.joblib")
    metrics_path = os.path.join(output_dir, "biorad_gate_model_metrics.json")
    predictions_path = os.path.join(output_dir, "biorad_gate_model_predictions.csv")
    importance_path = os.path.join(output_dir, "biorad_gate_feature_importance.csv")

    joblib.dump(model, model_path)
    with open(metrics_path, "w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=2)
    predictions_df.to_csv(predictions_path, index=False)
    importance_df.to_csv(importance_path, index=False)

    print(f"Saved model: {model_path}")
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved test predictions: {predictions_path}")
    print(f"Saved feature importance: {importance_path}")
    print(f"x-gate MAE: {metrics['model_mae_x_gate']:.3f} (baseline: {metrics['baseline_mae_x_gate']:.3f})")
    print(f"y-gate MAE: {metrics['model_mae_y_gate']:.3f} (baseline: {metrics['baseline_mae_y_gate']:.3f})")


if __name__ == "__main__":
    main()