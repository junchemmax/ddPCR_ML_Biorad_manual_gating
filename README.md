# ddPCR_ML_Biorad_manual_gating

Build a well-level ddPCR feature dataset from Bio-Rad amplitude exports, train a model that predicts manual gate positions, and score wells for likely mutants.

## Repository Structure

- `Biorad_manual_gating/build_biorad_gate_dataset.py` - builds manual-label gate summaries, diagnostic plots, and the feature dataset.
- `Biorad_manual_gating/train_and_predict_gate_model.py` - trains the gate model, predicts all wells, and evaluates held-out droplet classification.
- `Biorad_manual_gating/mutant_well_analysis.py` - scores each well for mutant signal and calls at most one mutant per plate.
- `ddPCR_data/MIP_XXX/` - raw experiments with `*_Amplitude.csv` exports; plots go to `Biorad_manual_gating_output/`.
- `data_output/` - generated gate summaries, feature dataset, mutant labels (`mut_info.csv`), and mutant-well scores.
- `gating_model_output/` - trained model, predictions, metrics, and held-out evaluation reports.
- `aws_s3_cmd.txt` - example S3 sync commands.

## Requirements

Python 3.10 or newer:

```bash
pip install pandas numpy scipy scikit-learn matplotlib joblib
```

## 1. Manual Gate Targets

Amplitude exports have three header rows, then Ch1/Ch2 amplitudes and two manual classification columns: Target 1 / MUT (Ch1, y-axis) and Target 2 / WT (Ch2, x-axis). Labels `0` and `1` are used; `u` is excluded.

For each axis, the gate is placed between the negative and positive droplets. If they overlap, the threshold with the fewest misclassified droplets is used; if only one class exists, the gate is placed just beyond the observed amplitude range. The method is stored in `x_gate_method` / `y_gate_method`.

These gates are reconstructed from manual droplet labels, not exported Bio-Rad UI coordinates. `ClusterData.csv` is not used.

Quadrants:

| Quadrant | MUT | WT |
|---|---|---|
| Q1 | + | + |
| Q2 | + | - |
| Q3 | - | - |
| Q4 | - | + |

## 2. Build the Dataset

```bash
python Biorad_manual_gating/build_biorad_gate_dataset.py ddPCR_data
```

Existing wells are kept and only new wells are processed. Outputs:

- `data_output/full_biorad_cluster_dataset.csv` - one row per well: gates, quadrant counts, and amplitude features.
- `data_output/biorad_cluster_gate_data.csv` - gate values, methods, and quadrant summaries.
- `data_output/precheck_missing_gate_data.csv`, `data_output/wells_with_missing_gates.csv` - wells or files with missing gate information.
- `ddPCR_data/MIP_XXX/Biorad_manual_gating_output/*_gated_cluster_lines.png` - per-well scatter plots with the manual gates.

Options:

```bash
python Biorad_manual_gating/build_biorad_gate_dataset.py ddPCR_data --precheck-only     # check labels only
python Biorad_manual_gating/build_biorad_gate_dataset.py ddPCR_data --rebuild           # recalculate all wells
python Biorad_manual_gating/build_biorad_gate_dataset.py ddPCR_data/MIP_043/210802_MIP-043_TRT63_TA171 --refresh-existing
python Biorad_manual_gating/build_biorad_gate_dataset.py ddPCR_data --refresh-existing --no-plots
```

## 3. Gate Model

```bash
python Biorad_manual_gating/train_and_predict_gate_model.py
```

### How it is trained

1. **Input:** `data_output/full_biorad_cluster_dataset.csv`; wells without both `x_gate` and `y_gate` are dropped.
2. **Targets:** `x_gate` (Ch2 / WT) and `y_gate` (Ch1 / MUT), predicted together as a multi-output regression.
3. **Features:** all numeric columns except identifiers and anything derived from the manual labels or gates (`x_gate`, `y_gate`, `quadrant`, `count`, cluster means, `gate_*`, `weighted_mean_*`, `n_populated_quadrants`, target names). This leaves 28 label-free amplitude features:
   - amplitude bounds: `x_min_ch2`, `x_max_ch2`, `y_min_ch1`, `y_max_ch1`, `amplitude_range_WT`, `amplitude_range_MUT`
   - droplet count: `n_droplets`
   - per-channel distribution shape (`ch1_*`, `ch2_*`): `std`, `iqr`, `cv`, `skew`, `kurt`, `bimodality`, `log_std`, `log_skew`, `outlier_frac`, `kde_peaks`
   - channel correlation: `ch1_ch2_corr`
4. **Split:** `GroupShuffleSplit` by `parent_dataset` (80% train / 20% test, `random_state=42`), so all wells from one plate are in either train or test, never both.
5. **Model:** scikit-learn pipeline of `SimpleImputer(strategy="median")` and `RandomForestRegressor(n_estimators=400, min_samples_leaf=3, max_features=0.8, random_state=42)`.
6. **Baseline:** the training-set median gate, used to judge whether the model adds value.
7. **Droplet check:** for each held-out well, droplets are assigned to quadrants with both the reference and predicted gates (positive = amplitude strictly above the gate) and compared.
8. **Final model:** after evaluation, the same pipeline is refit on all eligible wells (14,212 wells, 150 plates). This full-data model is the one saved and used for all-well predictions; the reported metrics come from the 80/20 split model.

Change paths, `TEST_SIZE`, or `RANDOM_STATE` at the top of the script.

### Current results

| | Train | Test |
|---|---|---|
| Wells | 11,380 | 2,832 |
| Plates | 120 | 30 |

| Target | Model MAE | Model RMSE | Median baseline MAE |
|---|---|---|---|
| `x_gate` (WT) | 257 | 330 | 465 |
| `y_gate` (MUT) | 291 | 427 | 1,231 |

Held-out droplet quadrant agreement: 97.1% across 2,832 wells.

### Outputs (`gating_model_output/`)

- `biorad_gate_model.joblib` - pipeline refit on all eligible wells.
- `biorad_gate_model_metrics.json`, `biorad_gate_model_predictions.csv` - held-out metrics and predictions from the 80/20 split model.
- `biorad_gate_predictions_all_wells.csv` - full-data model predictions for every well (all wells were used in training, so not an independent test).
- `biorad_gate_feature_importance.csv` - Random Forest feature importance of the full-data model.
- `heldout_droplet_comparison.csv`, `heldout_experiment_comparison.csv`, `heldout_quadrant_confusion.csv`, `heldout_droplet_evaluation_summary.json`, `heldout_droplet_evaluation_failures.csv` - droplet-level evaluation.

## 4. Mutant-Well Analysis

```bash
python Biorad_manual_gating/mutant_well_analysis.py
```

Uses the quadrant counts in the feature dataset and the known mutant wells in `data_output/mut_info.csv` (columns `dataset`, `well`). Writes `data_output/mutant_well_features.csv`.

- **Score:** each well's Q2 count and Q1+Q2 count are tested against the plate's median Q2/Q4 and (Q1+Q2)/Q4 ratio with a one-sided Poisson test. `combined_score` is the sum of the two -log10 p-values.
- **Poisson red flag:** Q1 is compared with the count expected if MUT and WT partition independently (two-sided Poisson test). Wells with p < 0.001 are flagged and never called.
- **Call:** at most one well per plate (`MAX_MUTANTS_PER_PLATE`), the highest-scoring non-flagged well with `combined_score >= MIN_CALL_SCORE` (default 5).

On the 66 labelled plates, the default setting calls the labelled mutant on 71% of plates, with 16 wrong calls and 3 plates without a call. Low-signal mutants (no Q2 droplets or fewer than ~10 MUT-positive droplets) cannot be separated from background with count features alone.

### Mutant Candidate Ranking Model

```bash
python Biorad_manual_gating/train_mutant_candidate_model.py
```

This trains a regularized logistic ranker. All 150 plates are used as labeled data: the 66 plates listed in `mut_info.csv` have one known mutant well, and the remaining plates are treated as confirmed mutant-free, as annotated for this dataset. Validation uses five-fold `GroupKFold` splits by plate, so wells from one plate are never split between train and test. The model achieved 76% top-1 and 97% top-3 recovery on the 66 positive plates, versus 74% and 95% for `combined_score` alone. Scores are rankings, not calibrated probabilities; results are exploratory because there are only 66 positive plates.

`mutant_well_features.csv` and the ranking exports include `mut_wt_independence_p` (a two-sided Fisher exact test of the quadrant table) and `enrichment_reliability_warning` (`p < 0.001`). This is a warning about consistency with independent MUT/WT occupancy, not a mutant call or artifact diagnosis; it does not change model rankings.

Outputs are written to `mut_prediction_model_output/`: the final fitted ranker, out-of-fold predictions and metrics for all wells, and the top-ranked well for each of the 150 plates. Out-of-fold scores are used for the current dataset exports; the full-data fit is saved for future inference.

## Data Sync

`aws_s3_cmd.txt` contains example commands for syncing raw data and dataset CSVs to S3. Update paths and bucket before use.

## Troubleshooting

- No datasets found: check the path contains `*_Amplitude.csv` files.
- Missing columns: the export needs Ch1/Ch2 amplitudes and two manual classification columns after three header rows.
- Missing gates: see `precheck_missing_gate_data.csv`; files with no usable amplitudes are skipped and axes without manual labels stay blank.
