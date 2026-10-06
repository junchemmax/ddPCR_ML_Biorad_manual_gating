# ddPCR_ML_Biorad_manual_gating

Bio-Rad manual quadrant-gating summaries and amplitude-derived features for ddPCR data.

This repository contains:
- Raw ddPCR datasets organized by assay/batch in `ddPCR_data/`
- Bio-Rad `ClusterData.csv` quadrant-gating exports
- A canonical pipeline that creates `data_output/full_biorad_cluster_dataset.csv`

## Repository Structure

- `Biorad_manual_gating/`
  - `build_biorad_gate_dataset.py` - reads amplitude and `ClusterData.csv` files, creates gate summaries and plots, and builds the feature dataset
- `ddPCR_data/`
  - `MIP_XXX/`
    - `*_Amplitude.csv` files (per-well amplitude exports)
    - `*_ClusterData.csv` - Bio-Rad QuantaSoft/QX Manager quadrant-gating export (present for most, not all, `MIP_XXX` folders)
    - `Biorad_manual_gating_output/` - per-dataset generated plots
- `data_output/` - generated dataset, gate summaries, and reports
- `aws_s3_cmd.txt` - utility command notes

## Bio-Rad Pipeline

1. `build_biorad_gate_dataset.py` reads `Ch1Amplitude`, `Ch2Amplitude`, and Bio-Rad `ClusterData.csv` files.
2. It calculates Ch2 X-axis and Ch1 Y-axis gate positions and assigns quadrants using:
  - Q1: Ch1 positive, Ch2 positive
  - Q2: Ch1 positive, Ch2 negative
  - Q3: Ch1 negative, Ch2 negative
  - Q4: Ch1 negative, Ch2 positive
3. The script calculates amplitude distribution features and gate summaries in the same per-well pass.
4. The final file is saved as `data_output/full_biorad_cluster_dataset.csv`.

The final dataset contains one row per well. `dataset` identifies the amplitude file,
`parent_dataset` identifies the experiment, and `well` identifies the plate position.
`quadrant`, `count`, `ch1_mean`, and `ch2_mean` are pipe-separated lists aligned by position.
The Ch1 channel represents MUT and Ch2 represents WT.

## Requirements

Recommended:
- Python 3.10+
- Virtual environment

Install dependencies:

```bash
pip install scipy scikit-learn matplotlib pandas
```

If you use the notebook, install Jupyter as needed:

```bash
pip install jupyter
```

## Data Expectations

Input CSVs are expected to:
- Match Bio-Rad style amplitude exports
- Have 3 non-data header rows (scripts use `skiprows=3`)
- Include columns:
  - `Ch1Amplitude`
  - `Ch2Amplitude`

Folder-level optimization expects multiple `*_Amplitude.csv` files inside one folder.

`ClusterData.csv` (when present next to a well's Amplitude CSVs) is Bio-Rad's own quadrant-gating
export. It has two sections: a per-quadrant droplet-count table (negative/positive combinations per
target) and a pairwise cluster-separation table (`S Value`). The cluster-labeling tools in this
branch use the droplet-count table only.

## Quick Start

### 1) Generate gates, plots, and the full feature dataset

Run from the repository root:

```bash
python Biorad_manual_gating/build_biorad_gate_dataset.py ddPCR_data
```

This creates the gate-summary CSV, per-dataset gate plots, and:

```text
data_output/biorad_cluster_gate_data.csv
data_output/full_biorad_cluster_dataset.csv
data_output/wells_with_missing_gates.csv
data_output/precheck_missing_gate_data.csv
```

Both gate reports include wells without usable amplitude observations. They are marked `not_processed` with a reason; processable wells with unavailable gate axes are also included and marked separately. The full feature dataset contains only wells with usable amplitude observations.
Folders containing amplitude CSVs but no `ClusterData.csv` are also processed as ungated wells; no plots are generated for those folders.

The script runs a gate-data precheck before feature extraction and plotting. It excludes amplitude files with no complete Ch1/Ch2 observations, since those files are skipped during processing. To run only the precheck, without processing wells, use:

```bash
python Biorad_manual_gating/build_biorad_gate_dataset.py ddPCR_data --precheck-only
```

The precheck treats an axis as estimable when its target has only one observed polarity. The script places that gate just outside the observed amplitude range to keep the observed droplets in the assigned class. This also handles multi-quadrant wells such as Q1+Q4, where Target 2 is positive in both quadrants. Axes without enough target information to estimate remain blank and are listed in the missing-coordinate report.

These are containment estimates rather than biologically inferred thresholds; the dataset marks them as `single_observed_class_amplitude_bound` in `x_gate_method` or `y_gate_method`.

Rerunning the command updates the existing dataset: wells already present are retained, and only new wells are processed. To recalculate every well, run:

```bash
python Biorad_manual_gating/build_biorad_gate_dataset.py ddPCR_data --rebuild
```

When ClusterData changes for an existing experiment, refresh only that experiment's rows and plots while retaining all other projects:

```powershell
python .\Biorad_manual_gating\build_biorad_gate_dataset.py .\ddPCR_data\MIP_043\210802_MIP-043_TRT63_TA171 --refresh-existing
```

The output includes the cleaned names `target1(MUT)`, `target2(WT)`, `quadrant`,
`n_populated_quadrants`, `weighted_mean_ch1_MUT`, `weighted_mean_ch2_WT`,
`amplitude_range_MUT`, and `amplitude_range_WT`.

### 2) Train an x/y gate prediction model

For an interactive training and prediction workflow, open
[`Biorad_manual_gating/train_and_predict_gate_model.ipynb`](Biorad_manual_gating/train_and_predict_gate_model.ipynb),
select the `learn_python` kernel, and run the cells in order. Edit the paths and
split settings in the first two code cells to use different inputs. The notebook
saves execution results and writes all-well predictions to
`model_output/biorad_gate_predictions_all_wells.csv` in addition to the artifacts below.
Training and inference default to `data_output/full_biorad_cluster_dataset.csv`;
model artifacts are saved under `model_output/`.
All-well predictions include training wells; use the grouped held-out metrics for
evaluation. Reference gates include amplitude-bound estimates, not just
ClusterData-mean midpoints, and are not original saved manual gate coordinates.

The notebook uses numeric amplitude-derived features and excludes manual gate
labels and gate-derived summaries to prevent target leakage. It holds out
complete `parent_dataset` groups rather than randomly splitting wells. The
model predicts `x_gate` (Ch2 axis) and `y_gate` (Ch1 axis) together and compares
its mean absolute error with a training-set median baseline.

The following model files are written to `model_output/`:

- `biorad_gate_model.joblib` - fitted scikit-learn model
- `biorad_gate_model_metrics.json` - held-out metrics and split details
- `biorad_gate_model_predictions.csv` - held-out wells and predictions
- `biorad_gate_feature_importance.csv` - random-forest feature importance

Configure `DATA_PATH`, `OUTPUT_DIR`, `TEST_SIZE`, and `RANDOM_STATE` in the notebook.
To score another feature CSV with the saved model, set `INFERENCE_DATA_PATH` and
`PREDICTION_OUTPUT_PATH`, then run the inference cells after the path configuration
and imports. Retraining is not required for inference with existing artifacts.

The scoring CSV must contain the same numeric feature columns used during
training, but it does not need to contain `x_gate` or `y_gate`.

### 3) Evaluate Droplet Classification

The final notebook section compares reference and predicted quadrant assignments
on raw amplitudes for held-out wells only. It can run with the notebook's import
and path configuration cells without retraining. Reports are saved in `model_output/`:

- `heldout_droplet_comparison.csv` - per-well quadrant counts, agreement, MUT/WT-positive shifts, and Poisson occupancy changes
- `heldout_experiment_comparison.csv` - per-experiment agreement and target-fraction shifts
- `heldout_quadrant_confusion.csv` - pooled reference-versus-predicted quadrant counts
- `heldout_droplet_evaluation_failures.csv` - files that could not be evaluated
- `heldout_droplet_evaluation_summary.json` - overall evaluation statistics

Reference assignments use reconstructed gates, not original Bio-Rad per-droplet
labels. Positive-fraction shifts are in percentage points. Poisson occupancy is
copies per droplet, not concentration; all-positive wells have unbounded
occupancy and are excluded from finite occupancy comparisons. No biological
acceptance threshold is assumed.

## Typical Workflow

1. Run `build_biorad_gate_dataset.py` on `ddPCR_data/` to create the full dataset.
2. Inspect or model `data_output/full_biorad_cluster_dataset.csv`.

## S3 Sync

Run these commands from the repository root. The first sync uploads this repository's `ddPCR_data/` folder; the second uploads the four selected output CSVs:

```bash
aws s3 sync "ddPCR_data" s3://ddpcr-ml-data/ddPCR_data
aws s3 sync "data_output" s3://ddpcr-ml-data/Biorad_manual_gating --exclude "*" --include "biorad_cluster_gate_data.csv" --include "full_biorad_cluster_dataset.csv" --include "mut_info.csv" --include "wells_with_missing_gates.csv"
```

Use the `ddPCR_data` folder in this repository. The older `...\github_projects\ddPCR_ML\ddPCR_data` path points to a different project and does not exist in this workspace.

## Troubleshooting

- No CSV files found:
  - Confirm path points to folder containing amplitude CSV files.
- Missing columns:
  - Verify CSV contains `Ch1Amplitude` and `Ch2Amplitude`.
- Quality concerns:
  - Add more diverse wells and experiments to the feature dataset.
  - Check gate placement and quadrant assignments in `ClusterData.csv`.
- Memory/runtime pressure during optimization:
  - Reduce `--trials`.
  - Set `N_JOBS` to `1` in script if needed.
- Large dataset runtime:
  - Feature extraction processes every well with a `ClusterData.csv` and can take time because KDE-based
    peak detection scales with droplet count.

## Notes

- `build_biorad_gate_dataset.py` generates gate summaries, plots, and the full feature dataset in one run.
- Not every `MIP_XXX` folder has a `ClusterData.csv` (a handful of folders are missing it, and one
  folder is effectively empty/placeholder data).

## License

Add your preferred license file (for example `LICENSE`) if this project will be shared publicly.
