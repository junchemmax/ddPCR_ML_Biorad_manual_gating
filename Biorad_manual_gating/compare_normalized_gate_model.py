"""Compare raw-amplitude gating with a per-well normalized gate model.

This experiment uses the existing plate-grouped holdout but writes only to
``gating_normalization_experiment``. It does not replace the production model.
"""
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupShuffleSplit

from build_biorad_gate_dataset import extract_features
from train_and_predict_gate_model import (
    DATA_PATH,
    GROUP_COLUMN,
    IDENTIFIER_COLUMNS,
    PROJECT_DIR,
    RANDOM_STATE,
    TARGET_COLUMNS,
    TEST_SIZE,
    assign_quadrants,
    load_training_data,
    make_model,
    select_feature_columns,
)

AMPLITUDE_ROOT = PROJECT_DIR / "ddPCR_data"
OUTPUT_DIR = PROJECT_DIR / "gating_normalization_experiment"
NORMALIZATION = "per-well channel median and IQR"
NORMALIZED_EXCLUDED_FEATURES = {
    "ch1_cv",
    "ch1_log_std",
    "ch1_log_skew",
    "ch2_cv",
    "ch2_log_std",
    "ch2_log_skew",
}


def build_amplitude_index() -> dict[tuple[str, str], Path]:
    candidates: dict[tuple[str, str], list[Path]] = defaultdict(list)
    for path in AMPLITUDE_ROOT.rglob("*_Amplitude.csv"):
        candidates[(path.parent.name, path.stem)].append(path)
    duplicates = {key: paths for key, paths in candidates.items() if len(paths) != 1}
    if duplicates:
        examples = {str(key): [str(path) for path in paths] for key, paths in list(duplicates.items())[:5]}
        raise ValueError(f"Amplitude source keys must be unique; duplicate examples: {examples}")
    return {key: paths[0] for key, paths in candidates.items()}


def robust_center_scale(values: np.ndarray) -> tuple[float, float, str]:
    center = float(np.median(values))
    scale = float(np.percentile(values, 75) - np.percentile(values, 25))
    method = "iqr"
    if not np.isfinite(scale) or scale <= 0:
        scale = float(np.percentile(values, 99) - np.percentile(values, 1))
        method = "p99-p01"
    if not np.isfinite(scale) or scale <= 0:
        scale = float(np.std(values))
        method = "std"
    if not np.isfinite(center) or not np.isfinite(scale) or scale <= 0:
        raise ValueError("A channel has no finite, positive robust scale.")
    return center, scale, method


def normalized_features(
    data: pd.DataFrame,
    amplitude_index: dict[tuple[str, str], Path],
) -> tuple[pd.DataFrame, pd.DataFrame, Counter[str]]:
    feature_rows: list[dict[str, float]] = []
    transform_rows: list[dict[str, float | str]] = []
    fallbacks: Counter[str] = Counter()
    for position, row in enumerate(data.itertuples(index=False), start=1):
        identity = (str(row.parent_dataset), str(row.dataset))
        path = amplitude_index.get(identity)
        if path is None:
            raise FileNotFoundError(f"No amplitude CSV found for {identity}")
        frame = pd.read_csv(
            path,
            skiprows=3,
            low_memory=False,
            usecols=["Ch1Amplitude", "Ch2Amplitude"],
        )
        amplitudes = frame.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        amplitudes = amplitudes[np.isfinite(amplitudes).all(axis=1)]
        if not len(amplitudes):
            raise ValueError(f"No finite amplitude observations in {path}")

        centers = np.empty(2, dtype=float)
        scales = np.empty(2, dtype=float)
        methods = []
        for channel in range(2):
            centers[channel], scales[channel], method = robust_center_scale(amplitudes[:, channel])
            methods.append(method)
            fallbacks[method] += 1

        normalized = (amplitudes - centers) / scales
        features = extract_features(normalized)
        ch1_min, ch2_min = normalized.min(axis=0)
        ch1_max, ch2_max = normalized.max(axis=0)
        features.update({
            "x_min_ch2": float(ch2_min),
            "x_max_ch2": float(ch2_max),
            "y_min_ch1": float(ch1_min),
            "y_max_ch1": float(ch1_max),
            "amplitude_range_WT": float(ch2_max - ch2_min),
            "amplitude_range_MUT": float(ch1_max - ch1_min),
        })
        feature_rows.append(features)
        transform_rows.append({
            "ch1_median": centers[0],
            "ch1_scale": scales[0],
            "ch2_median": centers[1],
            "ch2_scale": scales[1],
            "ch1_scale_method": methods[0],
            "ch2_scale_method": methods[1],
        })
        if position % 1000 == 0:
            print(f"Normalized {position}/{len(data)} wells")

    return pd.DataFrame(feature_rows), pd.DataFrame(transform_rows), fallbacks


def gate_metrics(targets: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    return {
        "mae_x_gate": float(mean_absolute_error(targets[:, 0], predictions[:, 0])),
        "rmse_x_gate": float(mean_squared_error(targets[:, 0], predictions[:, 0]) ** 0.5),
        "mae_y_gate": float(mean_absolute_error(targets[:, 1], predictions[:, 1])),
        "rmse_y_gate": float(mean_squared_error(targets[:, 1], predictions[:, 1]) ** 0.5),
    }


def evaluate_droplets(
    heldout: pd.DataFrame,
    baseline_predictions: np.ndarray,
    normalized_predictions: np.ndarray,
    amplitude_index: dict[tuple[str, str], Path],
) -> pd.DataFrame:
    rows = []
    for position, (row, baseline_gate, normalized_gate) in enumerate(
        zip(heldout.to_dict("records"), baseline_predictions, normalized_predictions),
        start=1,
    ):
        identity = (str(row["parent_dataset"]), str(row["dataset"]))
        path = amplitude_index[identity]
        frame = pd.read_csv(
            path,
            skiprows=3,
            low_memory=False,
            usecols=["Ch1Amplitude", "Ch2Amplitude"],
        )
        amplitude = frame.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        amplitude = amplitude[np.isfinite(amplitude).all(axis=1)]
        reference_gate = np.array([row["x_gate"], row["y_gate"]], dtype=float)
        reference_quadrants = assign_quadrants(amplitude, *reference_gate)
        baseline_quadrants = assign_quadrants(amplitude, *baseline_gate)
        normalized_quadrants = assign_quadrants(amplitude, *normalized_gate)
        row_result = {
            **{column: row[column] for column in IDENTIFIER_COLUMNS},
            "n_droplets": len(amplitude),
            "x_gate": reference_gate[0],
            "y_gate": reference_gate[1],
            "baseline_predicted_x_gate": baseline_gate[0],
            "baseline_predicted_y_gate": baseline_gate[1],
            "normalized_predicted_x_gate": normalized_gate[0],
            "normalized_predicted_y_gate": normalized_gate[1],
            "baseline_quadrant_agreement_pct": float(
                100 * np.mean(baseline_quadrants == reference_quadrants)
            ),
            "normalized_quadrant_agreement_pct": float(
                100 * np.mean(normalized_quadrants == reference_quadrants)
            ),
            "baseline_mut_shift_pp": float(
                100 * np.mean(baseline_quadrants <= 2) - 100 * np.mean(reference_quadrants <= 2)
            ),
            "normalized_mut_shift_pp": float(
                100 * np.mean(normalized_quadrants <= 2) - 100 * np.mean(reference_quadrants <= 2)
            ),
            "baseline_wt_shift_pp": float(
                100 * np.mean(np.isin(baseline_quadrants, [1, 4]))
                - 100 * np.mean(np.isin(reference_quadrants, [1, 4]))
            ),
            "normalized_wt_shift_pp": float(
                100 * np.mean(np.isin(normalized_quadrants, [1, 4]))
                - 100 * np.mean(np.isin(reference_quadrants, [1, 4]))
            ),
        }
        rows.append(row_result)
        if position % 500 == 0:
            print(f"Evaluated droplets for {position}/{len(heldout)} held-out wells")
    return pd.DataFrame(rows)


def summarize_droplets(wells: pd.DataFrame) -> dict[str, float | int]:
    total_droplets = int(wells["n_droplets"].sum())
    summary: dict[str, float | int] = {
        "heldout_wells": int(len(wells)),
        "heldout_droplets": total_droplets,
    }
    for model_name in ["baseline", "normalized"]:
        agreements = wells[f"{model_name}_quadrant_agreement_pct"]
        shifts_mut = wells[f"{model_name}_mut_shift_pp"].abs()
        shifts_wt = wells[f"{model_name}_wt_shift_pp"].abs()
        summary.update({
            f"{model_name}_mean_well_quadrant_agreement_pct": float(agreements.mean()),
            f"{model_name}_median_well_quadrant_agreement_pct": float(agreements.median()),
            f"{model_name}_p10_well_quadrant_agreement_pct": float(agreements.quantile(0.1)),
            f"{model_name}_wells_at_least_99pct_agreement": int((agreements >= 99).sum()),
            f"{model_name}_mean_abs_MUT_positive_shift_pp": float(shifts_mut.mean()),
            f"{model_name}_p95_abs_MUT_positive_shift_pp": float(shifts_mut.quantile(0.95)),
            f"{model_name}_max_abs_MUT_positive_shift_pp": float(shifts_mut.max()),
            f"{model_name}_mean_abs_WT_positive_shift_pp": float(shifts_wt.mean()),
            f"{model_name}_p95_abs_WT_positive_shift_pp": float(shifts_wt.quantile(0.95)),
            f"{model_name}_max_abs_WT_positive_shift_pp": float(shifts_wt.max()),
        })
        pooled_agreement = np.average(
            agreements,
            weights=wells["n_droplets"],
        )
        summary[f"{model_name}_pooled_quadrant_agreement_pct"] = float(pooled_agreement)
    return summary


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_training_data().reset_index(drop=True)
    feature_columns = select_feature_columns(data)
    amplitude_index = build_amplitude_index()
    missing = [
        (str(row.parent_dataset), str(row.dataset))
        for row in data.itertuples(index=False)
        if (str(row.parent_dataset), str(row.dataset)) not in amplitude_index
    ]
    if missing:
        raise FileNotFoundError(f"Missing amplitude CSVs for {len(missing)} wells; examples: {missing[:5]}")

    groups = data[GROUP_COLUMN].astype(str)
    splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_idx, test_idx = next(splitter.split(data[feature_columns], data[TARGET_COLUMNS], groups))
    if not set(groups.iloc[train_idx]).isdisjoint(set(groups.iloc[test_idx])):
        raise AssertionError("A plate appears in both train and test splits.")

    normalized_data, transforms, scale_methods = normalized_features(data, amplitude_index)
    normalized_feature_columns = [
        column for column in feature_columns
        if column not in NORMALIZED_EXCLUDED_FEATURES
    ]
    normalized_data = normalized_data.drop(columns=list(NORMALIZED_EXCLUDED_FEATURES))
    if set(normalized_data.columns) != set(normalized_feature_columns):
        missing_features = sorted(set(normalized_feature_columns) - set(normalized_data.columns))
        unexpected_features = sorted(set(normalized_data.columns) - set(normalized_feature_columns))
        raise ValueError(
            "Normalized feature schema does not match the experiment schema: "
            f"missing={missing_features}, unexpected={unexpected_features}"
        )
    normalized_data = normalized_data.reindex(columns=normalized_feature_columns)

    raw_targets = data[TARGET_COLUMNS].to_numpy(dtype=float)
    normalized_targets = np.column_stack([
        (raw_targets[:, 0] - transforms["ch2_median"]) / transforms["ch2_scale"],
        (raw_targets[:, 1] - transforms["ch1_median"]) / transforms["ch1_scale"],
    ])
    baseline_model = make_model()
    baseline_model.fit(data.iloc[train_idx][feature_columns], raw_targets[train_idx])
    baseline_predictions = baseline_model.predict(data.iloc[test_idx][feature_columns])

    normalized_model = make_model()
    normalized_model.fit(
        normalized_data.iloc[train_idx][normalized_feature_columns],
        normalized_targets[train_idx],
    )
    normalized_gate_coordinates = normalized_model.predict(
        normalized_data.iloc[test_idx][normalized_feature_columns]
    )
    heldout_transforms = transforms.iloc[test_idx]
    normalized_predictions = np.column_stack([
        heldout_transforms["ch2_median"].to_numpy()
        + normalized_gate_coordinates[:, 0] * heldout_transforms["ch2_scale"].to_numpy(),
        heldout_transforms["ch1_median"].to_numpy()
        + normalized_gate_coordinates[:, 1] * heldout_transforms["ch1_scale"].to_numpy(),
    ])
    if not np.isfinite(baseline_predictions).all() or not np.isfinite(normalized_predictions).all():
        raise ValueError("The experiment produced non-finite gate predictions.")

    heldout_targets = raw_targets[test_idx]
    baseline_metrics = gate_metrics(heldout_targets, baseline_predictions)
    normalized_metrics = gate_metrics(heldout_targets, normalized_predictions)
    heldout_rows = data.iloc[test_idx][IDENTIFIER_COLUMNS + TARGET_COLUMNS].copy().reset_index(drop=True)
    heldout_rows["baseline_predicted_x_gate"] = baseline_predictions[:, 0]
    heldout_rows["baseline_predicted_y_gate"] = baseline_predictions[:, 1]
    heldout_rows["normalized_predicted_x_gate"] = normalized_predictions[:, 0]
    heldout_rows["normalized_predicted_y_gate"] = normalized_predictions[:, 1]
    heldout_rows["normalized_gate_x"] = normalized_gate_coordinates[:, 0]
    heldout_rows["normalized_gate_y"] = normalized_gate_coordinates[:, 1]

    print("Evaluating held-out droplet classifications...")
    well_comparison = evaluate_droplets(
        heldout_rows,
        baseline_predictions,
        normalized_predictions,
        amplitude_index,
    )
    plate_comparison = (
        well_comparison.groupby("parent_dataset", as_index=False)
        .agg(
            wells=("dataset", "count"),
            droplets=("n_droplets", "sum"),
            baseline_mean_agreement_pct=("baseline_quadrant_agreement_pct", "mean"),
            normalized_mean_agreement_pct=("normalized_quadrant_agreement_pct", "mean"),
            baseline_mut_abs_shift_pp=("baseline_mut_shift_pp", lambda values: values.abs().mean()),
            normalized_mut_abs_shift_pp=("normalized_mut_shift_pp", lambda values: values.abs().mean()),
            baseline_wt_abs_shift_pp=("baseline_wt_shift_pp", lambda values: values.abs().mean()),
            normalized_wt_abs_shift_pp=("normalized_wt_shift_pp", lambda values: values.abs().mean()),
        )
    )
    heldout_rows = heldout_rows.merge(
        well_comparison[
            IDENTIFIER_COLUMNS + [
                "n_droplets",
                "baseline_quadrant_agreement_pct",
                "normalized_quadrant_agreement_pct",
                "baseline_mut_shift_pp",
                "normalized_mut_shift_pp",
                "baseline_wt_shift_pp",
                "normalized_wt_shift_pp",
            ]
        ],
        on=IDENTIFIER_COLUMNS,
        how="left",
        validate="one_to_one",
    )
    heldout_rows.to_csv(OUTPUT_DIR / "heldout_well_comparison.csv", index=False)
    plate_comparison.to_csv(OUTPUT_DIR / "heldout_plate_comparison.csv", index=False)

    median_baseline = np.tile(np.median(raw_targets[train_idx], axis=0), (len(test_idx), 1))
    normalized_median_coordinates = np.tile(
        np.median(normalized_targets[train_idx], axis=0),
        (len(test_idx), 1),
    )
    normalized_median_baseline = np.column_stack([
        heldout_transforms["ch2_median"].to_numpy()
        + normalized_median_coordinates[:, 0] * heldout_transforms["ch2_scale"].to_numpy(),
        heldout_transforms["ch1_median"].to_numpy()
        + normalized_median_coordinates[:, 1] * heldout_transforms["ch1_scale"].to_numpy(),
    ])
    summary = {
        "normalization": NORMALIZATION,
        "normalized_channels": "each well and channel is centered by its amplitude median and divided by its IQR",
        "normalized_target": "(manual_gate - channel_median) / channel_IQR",
        "raw_gate_reconstruction": "channel_median + predicted_normalized_gate * channel_IQR",
        "normalized_feature_columns": normalized_feature_columns,
        "excluded_features": {
            feature: (
                "unstable because normalized amplitudes are median-centered"
                if feature.endswith("_cv")
                else "excluded because normalized amplitudes can be negative"
            )
            for feature in sorted(NORMALIZED_EXCLUDED_FEATURES)
        },
        "input_dataset": str(DATA_PATH),
        "amplitude_root": str(AMPLITUDE_ROOT),
        "split": {
            "method": "GroupShuffleSplit",
            "group_column": GROUP_COLUMN,
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "train_wells": int(len(train_idx)),
            "test_wells": int(len(test_idx)),
            "train_plates": int(groups.iloc[train_idx].nunique()),
            "test_plates": int(groups.iloc[test_idx].nunique()),
        },
        "scale_method_counts_across_channels": dict(scale_methods),
        "baseline_gate_metrics": baseline_metrics,
        "normalized_gate_metrics": normalized_metrics,
        "baseline_training_median_gate_mae": gate_metrics(heldout_targets, median_baseline),
        "normalized_target_median_gate_mae": gate_metrics(heldout_targets, normalized_median_baseline),
        "droplet_metrics": summarize_droplets(well_comparison),
        "limitations": [
            "This is one plate-grouped 80/20 split, not cross-validation.",
            "Manual gate targets are reconstructed from the existing droplet labels.",
            "Per-well normalization uses all unlabeled amplitudes in that well; it is valid only when the complete well's amplitudes are available at prediction time.",
            "The normalized model is experimental and was not saved over the production model.",
        ],
    }
    with (OUTPUT_DIR / "summary.json").open("w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"Saved experiment outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
