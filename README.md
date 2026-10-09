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
# ddPCR_ML_Biorad_manual_gating

Build a well-level ddPCR feature dataset and train a model to predict reconstructed Bio-Rad gate positions from amplitude data. Gate targets come from the original per-droplet manual labels in the amplitude CSVs; `ClusterData.csv` is not used to estimate them.

## Repository Structure

- `Biorad_manual_gating/build_biorad_gate_dataset.py` - builds manual-label gate summaries, diagnostic plots, and the feature dataset.
- `Biorad_manual_gating/train_and_predict_gate_model.ipynb` - trains, predicts, and evaluates gate positions and droplet classifications.
- `ddPCR_data/` - raw experiments organized in `MIP_XXX/` folders, containing `*_Amplitude.csv` exports and sometimes Bio-Rad `*_ClusterData.csv` files.
- `data_output/` - generated gate summaries, features, and data-quality reports.
- `model_output/` - trained model, predictions, metrics, and held-out evaluation reports.

## Manual Gate Targets

Amplitude exports are expected to have three non-data header rows, followed by Ch1/Ch2 amplitudes and two manual classification columns. The first classification column is Target 1 / MUT (Ch1, y-axis); the second is Target 2 / WT (Ch2, x-axis). Labels `0` and `1` are used for gate estimation and quadrant summaries; `u` labels are excluded from those calculations.

For each axis, the builder places a gate between negative and positive classes when they are separated. If the classes overlap, it chooses the threshold with the fewest classification errors. If only one polarity is present, it places the gate just beyond the observed amplitude range. The method is recorded in `x_gate_method` and `y_gate_method`.

These positions are reconstructed from manual droplet labels. They are not exact Bio-Rad UI gate coordinates, which are not exported in the available files. `ClusterData.csv` may exist for QA, but it is not the source of the gate targets.

Quadrant mapping:

- Q1: MUT positive, WT positive (`1,1`)
- Q2: MUT positive, WT negative (`1,0`)
- Q3: MUT negative, WT negative (`0,0`)
- Q4: MUT negative, WT positive (`0,1`)

## Requirements

Use Python 3.10 or newer. Install the pipeline and notebook dependencies:

```bash
pip install pandas numpy scipy scikit-learn matplotlib jupyter
```

## Build the Dataset

Run from the repository root:

```bash
python Biorad_manual_gating/build_biorad_gate_dataset.py ddPCR_data
```

The builder discovers amplitude CSVs recursively. By default, it retains existing wells and processes new wells. It writes:

- `data_output/biorad_cluster_gate_data.csv` - gate values, estimation methods, and manual-label quadrant summaries.
- `data_output/full_biorad_cluster_dataset.csv` - one row per processed well, with gate labels and amplitude features.
- `data_output/precheck_missing_gate_data.csv` - missing gate axes and files without usable amplitude observations.
- `data_output/wells_with_missing_gates.csv` - processed or skipped wells with missing gate information.
- `ddPCR_data/MIP_XXX/Biorad_manual_gating_output/` - diagnostic plots, unless plots are disabled.

Useful options:

```bash
# Check gate-label availability without building outputs
python Biorad_manual_gating/build_biorad_gate_dataset.py ddPCR_data --precheck-only

# Recalculate all wells
python Biorad_manual_gating/build_biorad_gate_dataset.py ddPCR_data --rebuild

# Refresh one experiment while retaining all other rows
python Biorad_manual_gating/build_biorad_gate_dataset.py ddPCR_data/MIP_043/210802_MIP-043_TRT63_TA171 --refresh-existing

# Refresh without regenerating diagnostic plots
python Biorad_manual_gating/build_biorad_gate_dataset.py ddPCR_data --refresh-existing --no-plots
```

## Train and Evaluate

Open [`Biorad_manual_gating/train_and_predict_gate_model.ipynb`](Biorad_manual_gating/train_and_predict_gate_model.ipynb), choose a Python kernel with the requirements installed, and run the notebook cells in order. Adjust its data, output, and split settings to change paths or evaluation parameters.

The notebook predicts `x_gate` (Ch2 / WT) and `y_gate` (Ch1 / MUT). It excludes manual labels and gate-derived summaries from model features and holds out complete `parent_dataset` groups for evaluation. Its reference gates are reconstructed estimates, not saved Bio-Rad UI coordinates.

Outputs are written under `model_output/`, including:

- `biorad_gate_model.joblib` - fitted model.
- `biorad_gate_model_metrics.json` and `biorad_gate_model_predictions.csv` - held-out gate prediction results.
- `biorad_gate_predictions_all_wells.csv` - predictions for the full input dataset.
- `biorad_gate_feature_importance.csv` - feature importance.
- `heldout_droplet_comparison.csv`, `heldout_experiment_comparison.csv`, and `heldout_quadrant_confusion.csv` - held-out classification comparisons.
- `heldout_droplet_evaluation_summary.json` and `heldout_droplet_evaluation_failures.csv` - evaluation summary and failures.

Droplet-classification agreement compares predicted gates with manual-label reference assignments for held-out wells. Positive-fraction shifts are percentage points. Poisson occupancy is copies per droplet, not concentration; all-positive wells have unbounded occupancy and are excluded from finite occupancy comparisons.

## Data Sync

From the repository root, `aws_s3_cmd.txt` contains example commands for syncing raw data and selected dataset CSVs to S3. Update the local paths and bucket as appropriate before running them.

## Troubleshooting

- If no dataset folders are found, check that the supplied path contains `*_Amplitude.csv` files.
- If columns are missing, verify the export has Ch1 and Ch2 amplitudes and two manual classification columns after the three header rows.
- If gate axes are unavailable, review `precheck_missing_gate_data.csv`; files with no usable amplitude observations are skipped, and axes without usable manual labels remain blank.
`amplitude_range_MUT`, and `amplitude_range_WT`.
