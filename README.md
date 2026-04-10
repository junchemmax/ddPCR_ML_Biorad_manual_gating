# ddPCR_ML

Machine-learning pipeline for unsupervised clustering of droplet digital PCR (ddPCR) amplitude data using **FlowSOM**, with Bayesian hyperparameter optimisation and a supervised parameter predictor that improves over time.

---

## Overview

ddPCR produces two-channel amplitude readings (Ch1, Ch2) per droplet per well. This project:

1. **Clusters** droplets plate-wide using FlowSOM (self-organising map + hierarchical metaclustering).
2. **Optimises** the SOM grid size (`XDIM`, `YDIM`) and training iterations (`RLEN`) automatically via Bayesian search.
3. **Learns** from accumulated optimisation results to predict good parameters for future plates instantly, without re-running the full search.

---

## Repository structure

```
ddPCR_ML/
├── ddPCR_data/                              # raw per-well Amplitude CSV files (git-ignored)
│   └── MIP_100/
│       ├── 220901_MIP100_PS_trt42_.../      # one subfolder per plate run
│       └── ...
└── flowsom/
    ├── run_ddpcr_flowsom.py                 # FlowSOM clustering (Python flowsom package)
    ├── run_ddpcr_flowsom_flowsom_rs.py      # FlowSOM clustering (Rust backend, faster)
    ├── optimize_flowsom_rs_params.py        # Bayesian hyperparameter search – local/interactive
    ├── optimize_flowsom_rs_params_aws.py    # Bayesian hyperparameter search – AWS/EC2, non-interactive
    ├── train_param_predictor.py             # Train supervised predictor on accumulated data
    └── predict_params.py                    # Predict best params for a new plate instantly
```

Generated files (not committed):

```
ddPCR_data/<plate_folder>/output/           # per-plate outputs
    ├── trials/                              # one PNG per Optuna trial (sortable by silhouette)
    ├── ddPCR_clusters_opt_*.png             # final scatter plot
    ├── ddPCR_cluster_counts_opt_*.csv       # per-well droplet counts
    ├── optuna_param_importances.html        # interactive Optuna charts (optional)
    └── optuna_optimization_history.html

flowsom/output/                             # cross-plate accumulated files
    ├── meta_dataset.csv                    # one row per optimised plate (grows over time)
    ├── param_predictor.pkl                 # trained Random Forest predictor
    └── param_predictor_report.txt          # CV scores and feature importances
```

---

## Input data format

Each well is one CSV file with 3 header rows followed by droplet data:

```
Target Value of 0 = negative
Target Value of 1 = positive
Target Value of u = unclassified (Advanced Classification Mode)

Ch1Amplitude,Ch2Amplitude,MUT,WT,
1755.646,3960.354,0,1,
995.6978,1166.363,0,0,
...
```

Only `Ch1Amplitude` and `Ch2Amplitude` are used for clustering.

---

## Installation

```bash
pip install anndata flowsom flowsom-rs mudata numpy pandas matplotlib \
            scipy scikit-learn optuna joblib plotly
```

> `flowsom-rs` is the Rust-accelerated backend used by `run_ddpcr_flowsom_flowsom_rs.py` and the optimiser. `flowsom` (Python) is used by `run_ddpcr_flowsom.py`.

---

## Usage

### 1. Basic clustering (fixed parameters)

Edit `DATA_DIR` and the three parameters at the top of either script, then run:

```bash
# Python backend
python flowsom/run_ddpcr_flowsom.py

# Rust backend (faster, same algorithm)
python flowsom/run_ddpcr_flowsom_flowsom_rs.py
```

Key parameters:

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `XDIM` | 10 | SOM grid width |
| `YDIM` | 10 | SOM grid height |
| `RLEN` | 10 | Training iterations per data pass |
| `N_CLUSTERS` | 4 | Number of FlowSOM metaclusters |

**Outputs:** `ddPCR_clusters_xdim{X}_ydim{Y}_rlen{R}.png`, `ddPCR_cluster_counts_xdim{X}_ydim{Y}_rlen{R}.csv`

---

### 2. Automatic hyperparameter optimisation (Bayesian)

#### Local / interactive

```bash
python flowsom/optimize_flowsom_rs_params.py
```

- Runs **40 Optuna trials** (TPE sampler) to maximise the **Silhouette Score** of the metaclusters.
- Searches: `XDIM` ∈ {6, 8, …, 16}, `YDIM` ∈ {6, 8, …, 16}, `RLEN` ∈ [5, 100], `N_CLUSTERS` ∈ [2, 4].
- Saves one PNG per trial to `<plate_folder>/output/trials/` (filenames sorted by score).
- After trials finish, prompts you to review PNGs and either accept the automatic best or choose a specific trial number.
- Type a trial number to save to `meta_dataset.csv`, or press Enter / type `skip` to skip.

Configure at the top of the script:

```python
DATA_DIR      = "path/to/your/plate_folder"
N_CLUST_RANGE = (2, 4)   # range for n_clusters Optuna parameter
N_TRIALS      = 40       # increase for a more thorough search
```

#### AWS / EC2 — non-interactive

```bash
python flowsom/optimize_flowsom_rs_params_aws.py --data-dir /path/to/plate_folder --skip-meta
```

No interactive prompt. Use CLI flags to control behaviour:

| Flag | Behaviour |
|------|-----------|
| *(no flags)* | Auto best, skip meta-dataset |
| `--data-dir PATH` | Override the dataset folder |
| `--trial N` | Use trial number N for final clustering |
| `--skip-meta` | Do **not** save to `meta_dataset.csv` (default) |
| `--save-meta` | Save to `meta_dataset.csv` |

**Two-phase EC2 workflow:**

```bash
# Phase 1 – run optimisation, skip meta
python optimize_flowsom_rs_params_aws.py --data-dir ~/data/plate1 --skip-meta
# → download trials/ PNGs locally, pick best trial number

# Phase 2 – finalise with chosen trial, save meta
python optimize_flowsom_rs_params_aws.py --data-dir ~/data/plate1 --trial 7 --save-meta
```

---

### 3. Train the supervised predictor

Once you have optimised **5 or more** plates (more is better), train the predictor:

```bash
python flowsom/train_param_predictor.py
```

- Trains a **Random Forest** (multi-output regression) on 26 statistical features extracted from each plate.
- Uses **Leave-One-Out CV** when ≤ 20 samples, 5-fold otherwise.
- Reports per-target MAE and feature importances.
- Saves `param_predictor.pkl` and `param_predictor_report.txt`.

---

### 4. Predict parameters for a new plate (instant)

```bash
python flowsom/predict_params.py  ddPCR_data/new_plate_folder/
```

Output:

```
── Predicted FlowSOM parameters ──────────────────
  XDIM = 10
  YDIM = 12
  RLEN = 25
──────────────────────────────────────────────────
```

Use these values directly in the clustering scripts. If confidence is low (few training samples), the script warns you to verify with Optuna.

---

## Meta-learning workflow

```
Each new plate
      │
      ▼
optimize_flowsom_rs_params.py ──► meta_dataset.csv (grows)
  (or _aws.py for EC2)                   │
                         (≥5 plates)     ▼
                         train_param_predictor.py ──► param_predictor.pkl
                                         │
                         Next plate      ▼
                         predict_params.py  ──► XDIM / YDIM / RLEN / N_CLUSTERS (instant)
```

The predictor improves as more plates are added. Re-run `train_param_predictor.py` periodically to refresh the model.

---

## Features used by the predictor

26 features are extracted per plate:

| Group | Features |
|-------|---------|
| Size | `n_droplets`, `n_wells` |
| Ch1 / Ch2 amplitude | mean, std, p5, p25, median, p75, p95, skewness, kurtosis, bimodality coefficient, log-space std & skew, outlier fraction |
| Cross-channel | Pearson correlation (Ch1 vs Ch2) |
| Per-well variability | coefficient of variation of well means (Ch1, Ch2), CV of droplet counts |

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `anndata` | In-memory droplet data structure |
| `flowsom` | FlowSOM Python implementation |
| `flowsom-rs` | Rust-accelerated SOM training |
| `mudata` | Required by the Python flowsom backend |
| `numpy`, `pandas` | Numerics and data handling |
| `matplotlib` | Scatter plots |
| `scipy` | Distance metrics, summary statistics |
| `scikit-learn` | Agglomerative clustering, Random Forest, metrics |
| `optuna` | Bayesian hyperparameter optimisation |
| `joblib` | Model serialisation |
| `plotly` *(optional)* | Interactive Optuna visualisation charts |
