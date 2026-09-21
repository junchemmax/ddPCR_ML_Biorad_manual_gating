# ddPCR_ML

Machine-learning-assisted FlowSOM parameter optimization for ddPCR amplitude data.

This repository contains:
- Raw ddPCR datasets organized by assay/batch in `ddPCR_data/`
- A FlowSOM optimization pipeline in `flowsom/`
- Scripts to build a meta-dataset, train a predictor, and estimate good FlowSOM parameters for new datasets
- Tools (this branch) that use each well's Bio-Rad `ClusterData.csv` quadrant-gating export as an
  objective, low-effort ground truth for cluster count, used to build a much larger training set for
  the cluster-count classifier

## Repository Structure

- `ddPCR_data/`
  - `MIP_XXX/`
    - `*_Amplitude.csv` files (per-well amplitude exports)
    - `*_ClusterData.csv` - Bio-Rad QuantaSoft/QX Manager quadrant-gating export (present for most, not all, `MIP_XXX` folders)
    - `output/` folder for generated results
- `flowsom/`
  - `optimize_flowsom_rs_params_single.py` - Optuna + FlowSOM optimization over CSV files in a folder; also appends rows to `output/meta_dataset.csv`
  - `feature_engineering_diagnostics.ipynb` - notebook for diagnostics/feature exploration
  - `train_param_predictor.py_` - trains two models from `output/meta_dataset.csv`: a classifier for `N_CLUSTERS` and a regressor for `XDIM`/`YDIM`/`RLEN`
  - `predict_params.py_` - predicts FlowSOM parameters per well for new data and saves a predicted-cluster scatter PNG per well
  - `quadrant_predictor_feature_generator.py_` - feature generation and manual quadrant assignment workflow
  - `compare_cluster_labels.py` - diagnostic: compares manually-entered `best_n_clusters` labels in `meta_dataset.csv` against an objective count derived from each well's `ClusterData.csv`
  - `build_full_cluster_dataset.py` - builds `output/full_cluster_dataset.csv`, pairing amplitude-derived features for every well that has a `ClusterData.csv` with an objective cluster-count label (no Optuna/FlowSOM run required)
  - `train_cluster_classifier_full.py` - trains/evaluates a cluster-count classifier on `output/full_cluster_dataset.csv` via cross-validation
  - `output/` - script-generated artifacts, including `meta_dataset.csv` and `full_cluster_dataset.csv`
- `aws_s3_cmd.txt` - utility command notes

## What the Pipeline Does

1. Reads ddPCR amplitude CSV data (`Ch1Amplitude`, `Ch2Amplitude`).
2. Runs FlowSOM over a parameter search space:
   - `XDIM`
   - `YDIM`
   - `RLEN`
   - optionally `n_clusters`
3. Scores trials with either:
   - Silhouette score (maximize), or
   - Davies-Bouldin index (minimize)
4. Saves per-trial plots and summary outputs.
5. Allows manual trial override before writing final metadata.

## Requirements

Recommended:
- Python 3.10+
- Virtual environment

Install dependencies:

```bash
pip install optuna scipy anndata flowsom-rs scikit-learn matplotlib pandas joblib
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

### 1) Optimize FlowSOM parameters for one dataset folder

Run on a folder containing amplitude CSVs (for example one `MIP_XXX` folder):

```bash
python flowsom/optimize_flowsom_rs_params_single.py ddPCR_data/MIP_064
```

Optional arguments:

```bash
python flowsom/optimize_flowsom_rs_params_single.py ddPCR_data/MIP_064 --trials 60 --metric silhouette
python flowsom/optimize_flowsom_rs_params_single.py ddPCR_data/MIP_064 --trials 50 --metric davies_bouldin --n_clusters 3
```

Outputs are written under that dataset folder, typically:
- `output/<csv_stem>/trials/` (trial plots)
- cluster count summaries
- optimized scatter plot images

Note: the optimization script includes an interactive prompt to accept auto-best trial, choose another trial, mark single-cluster, or skip metadata write.

### 2) Build training data and train the parameter predictor

After running optimization across several datasets and building `flowsom/output/meta_dataset.csv`:

```bash
python flowsom/train_param_predictor.py_
```

This produces:
- `flowsom/param_predictor.pkl` - bundles a cluster-count classifier and a grid (XDIM/YDIM/RLEN) regressor
- `flowsom/param_predictor_report.txt`

### 3) Predict parameters for a new dataset

```bash
python flowsom/predict_params.py_ ddPCR_data/MIP_070
```

For each `*_Amplitude.csv` file (well) in the folder, the script prints predicted values for:
- `N_CLUSTERS`
- `XDIM`
- `YDIM`
- `RLEN`

and saves a scatter plot (`<csv_stem>_clusters_pred_....png`) to that dataset's `output/` folder, using FlowSOM run once with the predicted parameters. Use the printed values as warm-start hints before a full Optuna search.

### 4) (Optional) Build a larger cluster-count dataset from ClusterData.csv

`meta_dataset.csv` only has rows for wells that were manually run through the Optuna optimizer, and
its `best_n_clusters` label was entered by hand during that process. For folders that have a
`ClusterData.csv`, you can instead derive an objective cluster count directly from the droplet
counts in each quadrant - no FlowSOM/Optuna run required:

```bash
python flowsom/compare_cluster_labels.py --count-threshold 2
```

Prints an agreement report between the existing manual `best_n_clusters` labels and the
`ClusterData.csv`-derived count (a quadrant with at least `--count-threshold` droplets counts as a
real population). To build a full training set covering every well with a `ClusterData.csv`:

```bash
python flowsom/build_full_cluster_dataset.py --count-threshold 2
```

This saves `flowsom/output/full_cluster_dataset.csv` (features + objective `best_n_clusters`, no
grid params). Evaluate a classifier trained on it with:

```bash
python flowsom/train_cluster_classifier_full.py
```

## Typical Workflow

1. Run `optimize_flowsom_rs_params_single.py` on multiple representative `MIP_XXX` folders.
2. Accumulate metadata/features in `flowsom/output/meta_dataset.csv`.
3. Train predictor with `train_param_predictor.py_`.
4. Use `predict_params.py_` for new datasets.
5. Optionally run full optimization again using predicted values as guidance.
6. Optionally expand and improve cluster-count labels using `ClusterData.csv` (steps above) for
   folders where it's available.

## Troubleshooting

- No CSV files found:
  - Confirm path points to folder containing amplitude CSV files.
- Missing columns:
  - Verify CSV contains `Ch1Amplitude` and `Ch2Amplitude`.
- Poor prediction quality:
  - Add more diverse training datasets to `flowsom/output/meta_dataset.csv`.
  - Re-train predictor.
  - Consider training the cluster-count classifier on `full_cluster_dataset.csv` instead of/alongside
    `meta_dataset.csv` (see step 4 above) - it is ~39x larger and uses objective labels.
- Memory/runtime pressure during optimization:
  - Reduce `--trials`.
  - Set `N_JOBS` to `1` in script if needed.
- `build_full_cluster_dataset.py` runtime:
  - Processes every well with a `ClusterData.csv` (~8,700 wells across this dataset) and can take
    30-50 minutes even with parallelism, since KDE-based peak detection scales with droplet count.
    Use `--limit N` to test on a small subset first.

## Notes

- Some scripts currently use a trailing underscore in filename (for example `predict_params.py_`). Keep commands consistent with actual file names in this repository.
- The notebook is useful for exploratory diagnostics but the main production flow is script-based.
- Not every `MIP_XXX` folder has a `ClusterData.csv` (a handful of folders are missing it, and one
  folder is effectively empty/placeholder data).

## License

Add your preferred license file (for example `LICENSE`) if this project will be shared publicly.
