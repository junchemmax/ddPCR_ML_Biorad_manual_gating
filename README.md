# ddPCR_ML_Biorad_manual_gating

Bio-Rad manual quadrant-gating summaries and amplitude-derived features for ddPCR data.

This repository contains:
- Raw ddPCR datasets organized by assay/batch in `ddPCR_data/`
- Bio-Rad `ClusterData.csv` quadrant-gating exports
- A canonical pipeline that creates `output/full_biorad_cluster_dataset.csv`

## Repository Structure

- `Biorad_manual_gating/`
  - `draw_cluster_gates.py` - reads amplitude and `ClusterData.csv` files and creates gate summaries
  - `calculate_all_well_features.py` - adds amplitude features and derived gate summaries
  - `output/` - generated CSVs and reports
- `ddPCR_data/`
  - `MIP_XXX/`
    - `*_Amplitude.csv` files (per-well amplitude exports)
    - `*_ClusterData.csv` - Bio-Rad QuantaSoft/QX Manager quadrant-gating export (present for most, not all, `MIP_XXX` folders)
    - `Biorad_manual_gating_output/` - per-dataset generated plots
- `aws_s3_cmd.txt` - utility command notes

## Bio-Rad Pipeline

1. `draw_cluster_gates.py` reads `Ch1Amplitude`, `Ch2Amplitude`, and Bio-Rad `ClusterData.csv` files.
2. It calculates Ch2 X-axis and Ch1 Y-axis gate positions and assigns quadrants using:
  - Q1: Ch1 positive, Ch2 positive
  - Q2: Ch1 positive, Ch2 negative
  - Q3: Ch1 negative, Ch2 negative
  - Q4: Ch1 negative, Ch2 positive
3. `calculate_all_well_features.py` calculates amplitude distribution features and gate summaries.
4. The final file is saved as `output/full_biorad_cluster_dataset.csv`.

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

### 1) Generate Bio-Rad gate summaries

Run from the repository root:

```bash
python Biorad_manual_gating/draw_cluster_gates.py ddPCR_data
```

This creates `Biorad_manual_gating/output/biorad_cluster_gate_data.csv` and per-dataset gate plots.

### 2) Build the full feature dataset

```bash
python Biorad_manual_gating/calculate_all_well_features.py
```

This creates:

```text
output/full_biorad_cluster_dataset.csv
```

The output includes the cleaned names `target1(MUT)`, `target2(WT)`, `quadrant`,
`n_populated_quadrants`, `weighted_mean_ch1_MUT`, `weighted_mean_ch2_WT`,
`amplitude_range_MUT`, and `amplitude_range_WT`.

### 3) Train an x/y gate prediction model

```bash
python Biorad_manual_gating/train_gate_model.py
```

The trainer uses numeric amplitude-derived features and excludes manual gate
labels and gate-derived summaries to prevent target leakage. It holds out
complete `parent_dataset` groups rather than randomly splitting wells. The
model predicts `x_gate` (Ch2 axis) and `y_gate` (Ch1 axis) together and compares
its mean absolute error with a training-set median baseline.

The following files are written to `output/`:

- `biorad_gate_model.joblib` - fitted scikit-learn model
- `biorad_gate_model_metrics.json` - held-out metrics and split details
- `biorad_gate_model_predictions.csv` - held-out wells and predictions
- `biorad_gate_feature_importance.csv` - random-forest feature importance

Optional arguments include `--data-path`, `--output-dir`, `--test-size`, and
`--random-state`.

To score another feature CSV with the saved model, run:

```bash
python Biorad_manual_gating/predict_gate_model.py path/to/features.csv \
  --model-path output/biorad_gate_model.joblib \
  --metrics-path output/biorad_gate_model_metrics.json \
  --output-path output/predicted_gates.csv
```

The scoring CSV must contain the same numeric feature columns used during
training, but it does not need to contain `x_gate` or `y_gate`.

## Typical Workflow

1. Run `draw_cluster_gates.py` on `ddPCR_data/`.
2. Run `calculate_all_well_features.py` to create the full CSV.
3. Inspect or model `output/full_biorad_cluster_dataset.csv`.

## S3 Sync

The current `aws_s3_cmd.txt` syncs the two output files used by this workflow:

```bash
aws s3 sync "C:\Users\HQCHEJUN\Downloads\github_projects\ddPCR_ML_Biorad_manual_gating\output" s3://ddpcr-ml-data/Biorad_manual_gating --exclude "*" --include "full_biorad_cluster_dataset.csv" --include "mut_info.csv"
```

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

- The production flow consists of `draw_cluster_gates.py` followed by `calculate_all_well_features.py`.
- Not every `MIP_XXX` folder has a `ClusterData.csv` (a handful of folders are missing it, and one
  folder is effectively empty/placeholder data).

## License

Add your preferred license file (for example `LICENSE`) if this project will be shared publicly.
