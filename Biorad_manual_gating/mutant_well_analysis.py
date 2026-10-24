import os

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, poisson

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_DIR, "data_output", "full_biorad_cluster_dataset.csv")
MUT_PATH = os.path.join(PROJECT_DIR, "data_output", "mut_info.csv")
OUTPUT_PATH = os.path.join(PROJECT_DIR, "data_output", "mutant_well_features.csv")
MAX_MUTANTS_PER_PLATE = 1
MIN_CALL_SCORE = 5.0
POISSON_FLAG_P = 1e-3
OCCUPANCY_WARNING_P = 1e-3


def quadrant_counts(row: pd.Series) -> dict[str, int]:
    counts = {"Q1": 0, "Q2": 0, "Q3": 0, "Q4": 0}
    if isinstance(row["quadrant"], str) and isinstance(row["count"], str):
        for quadrant, count in zip(row["quadrant"].split("|"), row["count"].split("|")):
            counts[quadrant] = int(float(count))
    return counts


def poisson_lambda(positive: np.ndarray, total: np.ndarray) -> np.ndarray:
    fraction = np.clip(positive / total, 0, 1 - 1e-9)
    return -np.log1p(-fraction)


def mut_wt_independence_p(row: pd.Series) -> float:
    table = [
        [int(row["Q1"]), int(row["Q2"])],
        [int(row["Q4"]), int(row["Q3"])],
    ]
    return float(fisher_exact(table, alternative="two-sided").pvalue)


def background_neglog10_p(observed: pd.Series, ratio: pd.Series, base: pd.Series, plates: pd.Series) -> pd.Series:
    plate_ratio = ratio.groupby(plates).transform("median").fillna(0)
    exposure = base.clip(lower=1)
    expected = np.maximum(plate_ratio, 0.5 / exposure) * exposure
    return -np.log10(np.clip(poisson.sf(observed - 1, expected), 1e-300, 1))


def call_mutants(wells: pd.DataFrame, min_score: float) -> pd.Series:
    eligible = wells[~wells["poisson_red_flag"] & (wells["combined_score"] >= min_score)]
    top = eligible.sort_values("combined_score", ascending=False).groupby("parent_dataset").head(MAX_MUTANTS_PER_PLATE)
    return wells.index.isin(top.index)


def build_features() -> pd.DataFrame:
    data = pd.read_csv(DATA_PATH)
    mutants = pd.read_csv(MUT_PATH)
    counts = pd.DataFrame([quadrant_counts(row) for _, row in data.iterrows()])
    wells = pd.concat([data[["dataset", "parent_dataset", "well"]], counts], axis=1)

    wells["n_called"] = wells[["Q1", "Q2", "Q3", "Q4"]].sum(axis=1)
    wells["mut_pos"] = wells["Q1"] + wells["Q2"]
    wells["wt_pos"] = wells["Q1"] + wells["Q4"]
    wells["ratio_q12_q4"] = wells["mut_pos"] / wells["Q4"].replace(0, np.nan)
    wells["lambda_mut"] = poisson_lambda(wells["mut_pos"], wells["n_called"])
    wells["lambda_wt"] = poisson_lambda(wells["wt_pos"], wells["n_called"])
    wells["fractional_abundance"] = wells["lambda_mut"] / (wells["lambda_mut"] + wells["lambda_wt"])

    # Independent Poisson partitioning predicts Q1 = N * p_mut * p_wt.
    p_mut = wells["mut_pos"] / wells["n_called"]
    p_wt = wells["wt_pos"] / wells["n_called"]
    wells["expected_q1"] = wells["n_called"] * p_mut * p_wt
    wells["expected_q2"] = wells["n_called"] * p_mut * (1 - p_wt)
    wells["q1_obs_exp"] = (wells["Q1"] + 0.5) / (wells["expected_q1"] + 0.5)
    wells["q2_obs_exp"] = (wells["Q2"] + 0.5) / (wells["expected_q2"] + 0.5)

    # Background is the median mutant-positive fraction among wells on the same plate.
    plate_background = wells.groupby("parent_dataset")["mut_pos"].transform("sum")
    plate_droplets = wells.groupby("parent_dataset")["n_called"].transform("sum")
    median_rate = (wells["mut_pos"] / wells["n_called"]).groupby(wells["parent_dataset"]).transform("median")
    wells["plate_bg_rate"] = np.maximum(median_rate, 0.5 / wells["n_called"])
    wells["plate_mean_rate"] = plate_background / plate_droplets
    wells["expected_mut_bg"] = wells["plate_bg_rate"] * wells["n_called"]
    wells["poisson_neglog10_p"] = -np.log10(
        np.clip(poisson.sf(wells["mut_pos"] - 1, wells["expected_mut_bg"]), 1e-300, 1)
    )
    wells["fold_over_plate_bg"] = (wells["mut_pos"] + 0.5) / (wells["expected_mut_bg"] + 0.5)

    log_fa = np.log10(wells["fractional_abundance"] + 1e-6)
    plate_median = log_fa.groupby(wells["parent_dataset"]).transform("median")
    plate_mad = log_fa.groupby(wells["parent_dataset"]).transform(
        lambda values: np.median(np.abs(values - np.median(values)))
    )
    wells["fa_robust_z"] = (log_fa - plate_median) / (1.4826 * plate_mad + 1e-6)

    # Non-mutant Q1 rises with WT load, so score Q2 among WT-negative droplets and Q1 among WT-positive droplets.
    wt_neg = wells["Q2"] + wells["Q3"]
    q2_rate = (wells["Q2"] / wt_neg).groupby(wells["parent_dataset"]).transform("median")
    expected_q2 = np.maximum(q2_rate, 0.5 / wt_neg) * wt_neg
    wells["q2_neglog10_p"] = -np.log10(np.clip(poisson.sf(wells["Q2"] - 1, expected_q2), 1e-300, 1))
    q1_rate = (wells["Q1"] / wells["wt_pos"]).groupby(wells["parent_dataset"]).transform("median")
    expected_q1 = np.maximum(q1_rate, 0.5 / wells["wt_pos"]) * wells["wt_pos"]
    wells["q1_neglog10_p"] = -np.log10(np.clip(poisson.sf(wells["Q1"] - 1, expected_q1), 1e-300, 1))
    wells["mutant_score"] = wells["q2_neglog10_p"] + wells["q1_neglog10_p"]

    plates = wells["parent_dataset"]
    q4 = wells["Q4"].replace(0, np.nan)
    wells["ratio_q2_q4"] = wells["Q2"] / q4
    wells["q2_q4_neglog10_p"] = background_neglog10_p(wells["Q2"], wells["ratio_q2_q4"], wells["Q4"], plates)
    wells["q12_q4_neglog10_p"] = background_neglog10_p(wells["mut_pos"], wells["ratio_q12_q4"], wells["Q4"], plates)
    wells["combined_score"] = wells["q2_q4_neglog10_p"] + wells["q12_q4_neglog10_p"]

    # Two-sided test that Q1 matches independent Poisson partitioning of MUT and WT.
    expected_q1 = wells["expected_q1"].clip(lower=1e-9)
    lower = poisson.cdf(wells["Q1"], expected_q1)
    upper = poisson.sf(wells["Q1"] - 1, expected_q1)
    wells["poisson_fit_p"] = np.clip(2 * np.minimum(lower, upper), 0, 1)
    wells["poisson_red_flag"] = wells["poisson_fit_p"] < POISSON_FLAG_P
    wells["mut_wt_independence_p"] = wells.apply(mut_wt_independence_p, axis=1)
    wells["enrichment_reliability_warning"] = wells["mut_wt_independence_p"] < OCCUPANCY_WARNING_P
    wells["mutant_call"] = call_mutants(wells, MIN_CALL_SCORE)

    mutant_ids = set(mutants["dataset"])
    labeled_plates = set(wells.loc[wells["dataset"].isin(mutant_ids), "parent_dataset"])
    wells["plate_labeled"] = wells["parent_dataset"].isin(labeled_plates)
    wells["is_mutant"] = wells["dataset"].isin(mutant_ids).astype(int)
    return wells


def rank_summary(wells: pd.DataFrame, feature: str) -> dict[str, object]:
    labeled = wells[wells["plate_labeled"]].copy()
    labeled["rank"] = labeled.groupby("parent_dataset")[feature].rank(ascending=False, method="min")
    ranks = labeled.loc[labeled["is_mutant"] == 1, "rank"]
    return {
        "feature": feature,
        "top1_pct": 100 * (ranks <= 1).mean(),
        "top3_pct": 100 * (ranks <= 3).mean(),
        "median_rank": ranks.median(),
        "worst_rank": ranks.max(),
    }


def main() -> None:
    wells = build_features()
    wells.to_csv(OUTPUT_PATH, index=False)
    features = [
        "mut_pos", "ratio_q12_q4", "fractional_abundance", "fa_robust_z",
        "poisson_neglog10_p", "fold_over_plate_bg", "q1_obs_exp", "q2_obs_exp",
        "q2_neglog10_p", "q1_neglog10_p", "mutant_score",
        "q2_q4_neglog10_p", "q12_q4_neglog10_p", "combined_score",
    ]
    print(f"Saved: {OUTPUT_PATH}")
    positive_plate_count = wells.loc[wells["plate_labeled"], "parent_dataset"].nunique()
    negative_label_plate_count = wells.loc[~wells["plate_labeled"], "parent_dataset"].nunique()
    mutant_count = int(wells["is_mutant"].sum())
    print(f"Plates: {wells['parent_dataset'].nunique()} ({positive_plate_count} with mutant labels, "
          f"{negative_label_plate_count} without mutant labels)")
    print("\nWithin-plate rank of the known mutant well:")
    print(pd.DataFrame([rank_summary(wells, feature) for feature in features]).round(2).to_string(index=False))

    labeled = wells[wells["plate_labeled"]]
    columns = ["mut_pos", "Q1", "Q2", "lambda_wt", "fractional_abundance", "combined_score", "poisson_fit_p"]
    print("\nMedian by class on labeled plates:")
    print(labeled.groupby("is_mutant")[columns].median().round(5).T.to_string())

    flagged = labeled["poisson_red_flag"]
    print(f"\nPoisson red flags (p < {POISSON_FLAG_P:g}): mutant "
          f"{flagged[labeled['is_mutant'] == 1].sum()}/{mutant_count}, "
          f"non-mutant {flagged[labeled['is_mutant'] == 0].sum()}/{(labeled['is_mutant'] == 0).sum()}, "
          f"unlabeled-plate wells {wells.loc[~wells['plate_labeled'], 'poisson_red_flag'].sum()}")
    reliability_warnings = wells["enrichment_reliability_warning"]
    print(
        f"\nMUT/WT occupancy reliability warnings (Fisher p < {OCCUPANCY_WARNING_P:g}): mutant "
        f"{reliability_warnings[wells['is_mutant'] == 1].sum()}/{mutant_count}, "
        f"negative {reliability_warnings[wells['is_mutant'] == 0].sum()}/{(wells['is_mutant'] == 0).sum()}"
    )

    print(f"\nCalls (max {MAX_MUTANTS_PER_PLATE}/plate, red-flagged wells excluded):")
    rows = []
    for min_score in [0, 3, 5, 10, 15, 20]:
        calls = call_mutants(wells, min_score)
        lab = wells["plate_labeled"]
        hits = calls & (wells["is_mutant"] == 1)
        rows.append({
            "min_score": min_score,
            "mutant_found_pct": 100 * hits.sum() / mutant_count if mutant_count else 0,
            "false_calls_labeled": (calls & lab & (wells["is_mutant"] == 0)).sum(),
            "labeled_plates_no_call": positive_plate_count - wells.loc[calls & lab, "parent_dataset"].nunique(),
            "unlabeled_calls": (calls & ~lab).sum(),
            "unlabeled_plates_called": wells.loc[calls & ~lab, "parent_dataset"].nunique(),
            "unlabeled_plate_call_pct": (
                100 * wells.loc[calls & ~lab, "parent_dataset"].nunique() / negative_label_plate_count
                if negative_label_plate_count else 0
            ),
        })
    print(pd.DataFrame(rows).round(1).to_string(index=False))


if __name__ == "__main__":
    main()
