"""Train and evaluate a plate-aware mutant candidate ranker."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_DIR / "data_output" / "mutant_well_features.csv"
OUTPUT_DIR = PROJECT_DIR / "mut_prediction_model_output"
MODEL_PATH = OUTPUT_DIR / "mutant_candidate_ranker.joblib"
METRICS_PATH = OUTPUT_DIR / "mutant_candidate_model_metrics.json"
CV_PREDICTIONS_PATH = OUTPUT_DIR / "mutant_candidate_cv_predictions.csv"
ALL_CANDIDATES_PATH = OUTPUT_DIR / "mutant_candidate_predictions.csv"
TOP_CANDIDATES_PATH = OUTPUT_DIR / "mutant_candidate_top_per_plate.csv"
N_SPLITS = 5

FEATURE_COLUMNS = [
    "log_mut_pos",
    "log_q4",
    "log_ratio_q12_q4",
    "q2_fraction",
    "fractional_abundance",
    "poisson_neglog10_p",
    "q2_neglog10_p",
    "q1_neglog10_p",
    "combined_score",
    "poisson_fit_p",
]
IDENTIFIER_COLUMNS = ["dataset", "parent_dataset", "well"]


def add_model_features(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    mut_pos = pd.to_numeric(data["Q1"], errors="coerce") + pd.to_numeric(data["Q2"], errors="coerce")
    q4 = pd.to_numeric(data["Q4"], errors="coerce")
    data["log_mut_pos"] = np.log1p(mut_pos.clip(lower=0))
    data["log_q4"] = np.log1p(q4.clip(lower=0))
    data["log_ratio_q12_q4"] = np.log10((mut_pos + 0.5) / (q4 + 0.5))
    data["q2_fraction"] = (pd.to_numeric(data["Q2"], errors="coerce") + 0.5) / (mut_pos + 1)
    return data


def make_model() -> object:
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000),
    )


def percentile_rank_by_plate(values: pd.Series, groups: pd.Series) -> pd.Series:
    return values.groupby(groups).rank(method="average", pct=True, ascending=True)


def ranking_metrics(data: pd.DataFrame, score: pd.Series) -> dict[str, float | int]:
    ranked = data[IDENTIFIER_COLUMNS + ["is_mutant"]].copy()
    ranked["score"] = np.nan_to_num(score.to_numpy(dtype=float), nan=0.0, posinf=1e6, neginf=-1e6)
    if "poisson_red_flag" in data:
        ranked.loc[data["poisson_red_flag"].astype(bool).to_numpy(), "score"] = -1e6
    ranked["rank"] = ranked.groupby("parent_dataset")["score"].rank(
        method="min", ascending=False
    )
    positive_ranks = ranked.loc[ranked["is_mutant"] == 1, "rank"]
    return {
        "top1_rate": float((positive_ranks <= 1).mean()),
        "top3_rate": float((positive_ranks <= 3).mean()),
        "median_positive_rank": float(positive_ranks.median()),
        "positive_plates": int(data.loc[data["is_mutant"] == 1, "parent_dataset"].nunique()),
        "average_precision": float(
            average_precision_score(
                data["is_mutant"].astype(int), ranked["score"]
            )
        ),
        "roc_auc": float(roc_auc_score(data["is_mutant"].astype(int), ranked["score"])),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = add_model_features(pd.read_csv(DATA_PATH))
    training_data = data.copy().reset_index(drop=True)
    if training_data["is_mutant"].isna().any():
        raise ValueError("Every well must have a known positive or negative label.")
    if training_data["parent_dataset"].nunique() < N_SPLITS:
        raise ValueError(f"At least {N_SPLITS} labeled plates are required.")
    if training_data["is_mutant"].sum() != training_data.loc[
        training_data["is_mutant"] == 1, "parent_dataset"
    ].nunique():
        raise ValueError("Expected exactly one known mutant well per labeled plate.")

    x = training_data[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)
    y = training_data["is_mutant"].astype(int).to_numpy()
    groups = training_data["parent_dataset"].astype(str)
    cv_model_scores = np.full(len(training_data), np.nan)
    fold_ids = np.full(len(training_data), -1, dtype=int)

    splitter = GroupKFold(n_splits=N_SPLITS)
    for fold, (train_indices, test_indices) in enumerate(
        splitter.split(x, y, groups), start=1
    ):
        model = make_model()
        model.fit(x.iloc[train_indices], y[train_indices])
        cv_model_scores[test_indices] = model.decision_function(x.iloc[test_indices])
        fold_ids[test_indices] = fold

    cv_predictions = training_data[
        IDENTIFIER_COLUMNS + [
            "is_mutant", "Q1", "Q2", "Q3", "Q4", "n_called", "mut_pos",
            "fractional_abundance", "combined_score", "ratio_q12_q4", "poisson_red_flag",
            "mut_wt_independence_p", "enrichment_reliability_warning",
        ]
    ].copy()
    cv_predictions["plate_has_known_mutant"] = cv_predictions.groupby("parent_dataset")[
        "is_mutant"
    ].transform("max").astype(bool)
    cv_predictions["fold"] = fold_ids
    cv_predictions["model_score"] = cv_model_scores
    cv_groups = cv_predictions["parent_dataset"]
    model_score_for_ranking = cv_predictions["model_score"].mask(
        cv_predictions["poisson_red_flag"].astype(bool), -1e6
    )
    heuristic_score_for_ranking = cv_predictions["combined_score"].mask(
        cv_predictions["poisson_red_flag"].astype(bool), -1e6
    )
    cv_predictions["model_rank_percentile"] = percentile_rank_by_plate(
        model_score_for_ranking, cv_groups
    )
    cv_predictions["heuristic_rank_percentile"] = percentile_rank_by_plate(
        heuristic_score_for_ranking, cv_groups
    )
    cv_predictions["candidate_score"] = cv_predictions["model_rank_percentile"]
    cv_predictions["candidate_rank"] = cv_predictions.groupby("parent_dataset")[
        "candidate_score"
    ].rank(method="min", ascending=False)
    cv_predictions["model_score_for_ranking"] = model_score_for_ranking
    cv_predictions["heuristic_score_for_ranking"] = heuristic_score_for_ranking
    cv_predictions.to_csv(CV_PREDICTIONS_PATH, index=False)

    metrics = {
        "data_path": str(DATA_PATH),
        "plates": int(groups.nunique()),
        "positive_plates": int(training_data.loc[training_data["is_mutant"] == 1, "parent_dataset"].nunique()),
        "confirmed_negative_plates": int(
            groups.nunique()
            - training_data.loc[training_data["is_mutant"] == 1, "parent_dataset"].nunique()
        ),
        "known_mutant_wells": int(y.sum()),
        "known_negative_wells": int(len(y) - y.sum()),
        "cv_method": f"GroupKFold({N_SPLITS}) by parent_dataset",
        "feature_columns": FEATURE_COLUMNS,
        "scores_are_calibrated_probabilities": False,
        "occupancy_warning_p_threshold": 0.001,
        "occupancy_warning_is_used_for_ranking": False,
        "limitations": [
            "Only 66 mutant-positive plates are available.",
            "All wells on plates absent from mut_info.csv are treated as confirmed negatives.",
            "Validation metrics are exploratory and use plate-level held-out folds.",
        ],
        "cross_validation": {
            "model_only": ranking_metrics(cv_predictions, cv_predictions["model_score"]),
            "heuristic_only": ranking_metrics(cv_predictions, cv_predictions["combined_score"]),
        },
    }
    plate_scores = cv_predictions.groupby("parent_dataset").agg(
        plate_has_known_mutant=("is_mutant", "max"),
        model_top_score=("model_score_for_ranking", "max"),
        heuristic_top_score=("heuristic_score_for_ranking", "max"),
    )
    metrics["plate_detection_roc_auc"] = {
        "model": float(roc_auc_score(plate_scores["plate_has_known_mutant"], plate_scores["model_top_score"])),
        "heuristic": float(roc_auc_score(plate_scores["plate_has_known_mutant"], plate_scores["heuristic_top_score"])),
    }

    final_model = make_model()
    final_model.fit(x, y)
    joblib.dump(final_model, MODEL_PATH)

    scored_data = cv_predictions.copy()
    output_columns = IDENTIFIER_COLUMNS + [
        "is_mutant", "plate_has_known_mutant", "Q1", "Q2", "Q3", "Q4",
        "n_called", "mut_pos", "ratio_q12_q4", "fractional_abundance",
        "combined_score", "poisson_red_flag", "mut_wt_independence_p",
        "enrichment_reliability_warning",
        "model_score", "model_rank_percentile", "heuristic_rank_percentile",
        "candidate_score", "candidate_rank",
    ]
    scored_data[output_columns].sort_values(
        ["parent_dataset", "candidate_rank", "candidate_score"],
        ascending=[True, True, False],
    ).to_csv(ALL_CANDIDATES_PATH, index=False)
    top_by_plate = (
        scored_data.sort_values(
            ["parent_dataset", "candidate_score", "model_score", "combined_score"],
            ascending=[True, False, False, False],
        )
        .groupby("parent_dataset", as_index=False)
        .head(1)
    )
    top_by_plate[output_columns].sort_values(
        "model_score", ascending=False
    ).to_csv(TOP_CANDIDATES_PATH, index=False)

    with METRICS_PATH.open("w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=2)

    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved CV predictions: {CV_PREDICTIONS_PATH}")
    print(f"Saved out-of-fold well rankings: {ALL_CANDIDATES_PATH}")
    print(f"Saved top candidate per plate: {TOP_CANDIDATES_PATH}")
    print(json.dumps(metrics["cross_validation"], indent=2))
    print("\nTop-ranked well per plate, sorted by model score:")
    print(
        top_by_plate[
            ["parent_dataset", "well", "is_mutant", "Q1", "Q2", "Q4", "ratio_q12_q4", "model_score", "candidate_score"]
        ].sort_values("model_score", ascending=False).head(15).to_string(index=False)
    )


if __name__ == "__main__":
    main()