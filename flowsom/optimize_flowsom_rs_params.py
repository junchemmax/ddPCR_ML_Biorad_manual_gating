"""
Bayesian hyperparameter optimisation for FlowSOM XDIM / YDIM / RLEN.

Uses Optuna to maximise the Silhouette Score of the resulting metaclustering.
After the study finishes, the best parameters are printed and a final
clustering plot + counts CSV are saved with those parameters.

Each run also appends one row to  meta_dataset.csv  (next to this script),
accumulating (data-features → best params) pairs that can later be used to
train a supervised predictor (see train_param_predictor.py).

Install once:
    pip install optuna scipy
"""
from __future__ import annotations

import glob
import os
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import anndata as ad
import flowsom_rs
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import kurtosis, skew
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

# ── Config ─────────────────────────────────────────────────────────────────────
DATA_DIR   = os.path.join(os.path.dirname(__file__), "..", "ddPCR_data/MIP_100/220901_MIP100_PS_trt42_ta395_20220901_141940_944")
N_CLUST_RANGE = (2, 4)  # n_clusters is now optimised by Optuna
SEED          = 42      # reproducibility
N_TRIALS   = 40         # Optuna trials (≈2-5 min on a laptop)
EVAL_PTS   = 50_000      # subsample size for silhouette (keeps it fast)
N_JOBS = -1  # parallel jobs for silhouette_score (set to 1 if you get memory issues)

# Search space – feel free to widen
XDIM_RANGE = (6, 16)    # even integers only
YDIM_RANGE = (6, 16)
RLEN_RANGE = (5, 100)

# All outputs go inside the data folder being processed
OUT_DIR    = os.path.join(DATA_DIR, "output")
TRIALS_DIR = os.path.join(OUT_DIR, "trials")
os.makedirs(TRIALS_DIR, exist_ok=True)

# meta_dataset.csv lives next to this script so it accumulates across datasets
META_DIR   = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(META_DIR, exist_ok=True)

optuna.logging.set_verbosity(optuna.logging.WARNING)  # quiet inner logs

# ── 1. Load data once ──────────────────────────────────────────────────────────
csv_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
print(f"Found {len(csv_files)} CSV files in '{os.path.basename(DATA_DIR)}'.")

adatas = []
for f in csv_files:
    parts = os.path.basename(f).split("_")
    well  = parts[-2]
    df    = pd.read_csv(f, skiprows=3)[["Ch1Amplitude", "Ch2Amplitude"]].dropna()
    adata = ad.AnnData(df.values.astype(np.float32))
    adata.var_names  = ["Ch1Amplitude", "Ch2Amplitude"]
    adata.obs_names  = [f"{well}_{i}" for i in range(len(df))]
    adata.obs["well"] = well
    adatas.append(adata)

combined = ad.concat(adatas, merge="same")
data     = combined.X.astype(np.float64)
print(f"Combined: {combined.n_obs:,} droplets across {len(csv_files)} wells.\n")

# Pre-draw fixed subsample indices (shared across all trials for fair comparison)
rng       = np.random.default_rng(SEED)
eval_idx  = rng.choice(len(data), min(EVAL_PTS, len(data)), replace=False)
eval_data = data[eval_idx]

PLOT_PTS  = 50_000
plot_idx  = np.random.default_rng(SEED + 1).choice(len(data), min(PLOT_PTS, len(data)), replace=False)
plot_idx.sort()
plot_data = data[plot_idx]


# ── 2. Optuna objective ────────────────────────────────────────────────────────
def _run_flowsom(xdim: int, ydim: int, rlen: int, n_clusters: int, seed: int = SEED):
    """Train SOM + metacluster; return (codes, node_meta, bmu_idx)."""
    n_nodes    = xdim * ydim
    grid       = [(x, y) for x in range(xdim) for y in range(ydim)]
    nhbrdist   = squareform(pdist(grid, metric="chebyshev"))
    radii      = (np.quantile(nhbrdist, 0.67), 0.0)
    codes_init = data[np.random.default_rng(seed).choice(len(data), n_nodes, replace=False)]

    codes, bmu_idx, _ = flowsom_rs.train_batch_som(
        data, codes_init, nhbrdist, radii, rlen=rlen
    )
    node_meta = AgglomerativeClustering(
        n_clusters=n_clusters, linkage="average"
    ).fit_predict(codes)
    return codes, node_meta, bmu_idx


def _save_trial_png(trial_number: int, xdim: int, ydim: int, rlen: int, n_clusters: int,
                    node_meta: np.ndarray, bmu_idx: np.ndarray, score: float) -> None:
    """Save a scatter plot for one Optuna trial.

    All trials use the same fixed subsample (plot_idx) so cluster boundaries
    are directly comparable across PNGs.
    Filename encodes the score so files sort naturally from best to worst:
        trial_sil+0.4231_t003_xdim10_ydim12_rlen25_nc4.png
    """
    X_p  = plot_data
    mc_p = (node_meta[bmu_idx] + 1)[plot_idx]

    fig, ax = plt.subplots(figsize=(5, 4))
    for mc in sorted(np.unique(mc_p)):
        mask = mc_p == mc
        ax.scatter(X_p[mask, 1], X_p[mask, 0], s=0.4, alpha=0.3,
                   label=f"C{mc}", rasterized=True)
    ax.set_xlabel("Ch2Amplitude")
    ax.set_ylabel("Ch1Amplitude")
    ax.legend(markerscale=6, title="MC", fontsize=7)
    ax.set_title(
        f"Trial {trial_number:03d}  xdim={xdim} ydim={ydim} rlen={rlen} nc={n_clusters}  "
        f"sil={score:.4f}",
        fontsize=9,
    )
    plt.tight_layout()
    fname = (
        f"trial_sil{score:+.4f}_t{trial_number:03d}"
        f"_xdim{xdim}_ydim{ydim}_rlen{rlen}_nc{n_clusters}.png"
    )
    fig.savefig(os.path.join(TRIALS_DIR, fname), dpi=100)
    plt.close(fig)


def objective(trial: optuna.Trial) -> float:
    xdim       = trial.suggest_int("xdim", XDIM_RANGE[0], XDIM_RANGE[1], step=2)
    ydim       = trial.suggest_int("ydim", YDIM_RANGE[0], YDIM_RANGE[1], step=2)
    rlen       = trial.suggest_int("rlen", RLEN_RANGE[0], RLEN_RANGE[1], log=True)
    n_clusters = trial.suggest_int("n_clusters", N_CLUST_RANGE[0], N_CLUST_RANGE[1])

    _, node_meta, bmu_idx = _run_flowsom(xdim, ydim, rlen, n_clusters)

    labels_eval = node_meta[bmu_idx][eval_idx]

    # Guard: silhouette undefined if only 1 unique label appears in subsample
    if len(np.unique(labels_eval)) < 2:
        _save_trial_png(trial.number, xdim, ydim, rlen, n_clusters, node_meta, bmu_idx, score=-1.0)
        return -1.0

    score = silhouette_score(eval_data, labels_eval, sample_size=None)
    _save_trial_png(trial.number, xdim, ydim, rlen, n_clusters, node_meta, bmu_idx, score)
    return float(score)


# ── 3. Run the study ───────────────────────────────────────────────────────────
print(f"Running Optuna study ({N_TRIALS} trials) …")
sampler = optuna.samplers.TPESampler(seed=SEED)
study   = optuna.create_study(direction="maximize", sampler=sampler)
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True, n_jobs=N_JOBS)

print(f"Trial PNGs saved → {TRIALS_DIR}  (sort by filename to rank by silhouette score)")

auto_best = study.best_params
auto_score = study.best_value
print(f"\nBest parameters found (by silhouette score):")
print(f"  XDIM       = {auto_best['xdim']}")
print(f"  YDIM       = {auto_best['ydim']}")
print(f"  RLEN       = {auto_best['rlen']}")
print(f"  N_CLUSTERS = {auto_best['n_clusters']}")
print(f"  Silhouette Score = {auto_score:.4f}")

# ── 4. Top-5 trials summary ────────────────────────────────────────────────────
print("\nTop-5 trials:")
trials_df = study.trials_dataframe()[["number", "value", "params_xdim", "params_ydim", "params_rlen", "params_n_clusters"]]
trials_df = trials_df.rename(columns={"value": "silhouette"}).sort_values("silhouette", ascending=False)
print(trials_df.head(5).to_string(index=False))

# ── 4b. Manual override ────────────────────────────────────────────────────────
print(f"\nReview the PNGs in: {TRIALS_DIR}")
print("Press Enter to accept the automatic best, type a trial number to use instead, or 'skip' to skip saving to meta-dataset.")
_choice = input("Manual trial number / 'skip' (or Enter to accept best): ").strip()

skip_meta = False
if _choice == "":
    best       = auto_best
    best_score = auto_score
    print("Using automatic best.")
elif _choice.lower() == "skip":
    best       = auto_best
    best_score = auto_score
    skip_meta  = True
    print("Skipping meta-dataset entry.")
else:
    try:
        _trial_num = int(_choice)
        _trial     = study.trials[_trial_num]
        best       = _trial.params
        best_score = _trial.value if _trial.value is not None else float("nan")
        print(
            f"Using trial {_trial_num}: "
            f"xdim={best['xdim']}  ydim={best['ydim']}  rlen={best['rlen']}  "
            f"n_clusters={best['n_clusters']}  sil={best_score:.4f}"
        )
    except (ValueError, IndexError) as e:
        print(f"Invalid input ({e}). Falling back to automatic best.")
        best       = auto_best
        best_score = auto_score

# ── 5. Final clustering with best params ───────────────────────────────────────
XDIM, YDIM, RLEN, N_CLUSTERS = best["xdim"], best["ydim"], best["rlen"], best["n_clusters"]
print(f"\nRunning final clustering: xdim={XDIM}, ydim={YDIM}, rlen={RLEN}, n_clusters={N_CLUSTERS} …")

_, node_meta, bmu_idx = _run_flowsom(XDIM, YDIM, RLEN, N_CLUSTERS)
combined.obs["cluster"]     = bmu_idx + 1
combined.obs["metacluster"] = node_meta[bmu_idx] + 1

print("\nDroplets per metacluster (all wells):")
print(combined.obs["metacluster"].value_counts().sort_index())

# ── 6. Scatter plot ────────────────────────────────────────────────────────────
max_pts  = 200_000
plot_idx = np.random.default_rng(0).choice(combined.n_obs, min(max_pts, combined.n_obs), replace=False)
plot_idx.sort()
X        = combined.X[plot_idx]
mc_plot  = combined.obs["metacluster"].values[plot_idx]

fig, ax = plt.subplots(figsize=(6, 5))
for mc in sorted(np.unique(mc_plot)):
    mask = mc_plot == mc
    ax.scatter(X[mask, 1], X[mask, 0], s=0.5, alpha=0.3, label=f"Cluster {mc}", rasterized=True)
ax.set_xlabel("Ch2Amplitude")
ax.set_ylabel("Ch1Amplitude")
ax.legend(markerscale=8, title="Metacluster")
ax.set_title(f"FlowSOM (optimised) – {len(csv_files)} wells, {combined.n_obs:,} droplets")
param_text = (
    f"xdim={XDIM}  ydim={YDIM}  rlen={RLEN}  nc={N_CLUSTERS}"
    f"  sil={study.best_value:.3f}  trials={N_TRIALS}"
)
ax.text(0.01, 0.01, param_text, transform=ax.transAxes, fontsize=7,
        verticalalignment="bottom", color="gray", family="monospace")
plt.tight_layout()

tag     = f"_opt_xdim{XDIM}_ydim{YDIM}_rlen{RLEN}_nc{N_CLUSTERS}"
out_png = os.path.join(OUT_DIR, f"ddPCR_clusters{tag}.png")
plt.savefig(out_png, dpi=150)
print(f"\nSaved scatter plot → {out_png}")

# ── 7. Per-well counts table ───────────────────────────────────────────────────
counts = (
    combined.obs
    .groupby(["well", "metacluster"], observed=True)
    .size()
    .unstack(fill_value=0)
    .sort_index()
)
counts.columns = [f"cluster_{c}" for c in counts.columns]
counts["total"] = counts.sum(axis=1)
for col in [c for c in counts.columns if c.startswith("cluster_")]:
    counts[col.replace("cluster_", "pct_")] = (counts[col] / counts["total"] * 100).round(2)

out_csv = os.path.join(OUT_DIR, f"ddPCR_cluster_counts{tag}.csv")
counts.to_csv(out_csv)
print(f"Saved per-well counts  → {out_csv}")
print("\nPer-well summary (first 10 wells):")
print(counts.head(10).to_string())

# ── 8. Append to meta-dataset (for supervised predictor training) ───────────────
def extract_features(X: np.ndarray, n_wells: int, combined=None) -> dict:
    """Compute summary statistics from raw amplitude data for meta-learning."""
    feat: dict = {}
    feat["n_droplets"] = len(X)
    feat["n_wells"]    = n_wells
    for i, ch in enumerate(["ch1", "ch2"]):
        vals = X[:, i]
        feat[f"{ch}_mean"]   = float(np.mean(vals))
        feat[f"{ch}_std"]    = float(np.std(vals))
        feat[f"{ch}_p5"]     = float(np.percentile(vals, 5))
        feat[f"{ch}_p25"]    = float(np.percentile(vals, 25))
        feat[f"{ch}_median"] = float(np.median(vals))
        feat[f"{ch}_p75"]    = float(np.percentile(vals, 75))
        feat[f"{ch}_p95"]    = float(np.percentile(vals, 95))
        feat[f"{ch}_skew"]   = float(skew(vals))
        feat[f"{ch}_kurt"]   = float(kurtosis(vals))
        # Bimodality coefficient: >0.555 suggests bimodality
        feat[f"{ch}_bimodality"] = (feat[f"{ch}_skew"] ** 2 + 1) / (feat[f"{ch}_kurt"] + 3 + 1e-9)
        # Log-space shape (captures behaviour not visible in linear space)
        log_vals = np.log1p(np.clip(vals, 0, None))
        feat[f"{ch}_log_std"]  = float(np.std(log_vals))
        feat[f"{ch}_log_skew"] = float(skew(log_vals))
        # Outlier fraction (beyond ±3 std)
        mu, sigma = feat[f"{ch}_mean"], feat[f"{ch}_std"]
        feat[f"{ch}_outlier_frac"] = float(np.mean((vals < mu - 3 * sigma) | (vals > mu + 3 * sigma)))
    feat["ch1_ch2_corr"] = float(np.corrcoef(X[:, 0], X[:, 1])[0, 1])
    # Per-well variability (requires combined AnnData with obs["well"])
    if combined is not None:
        for i, ch in enumerate(["ch1", "ch2"]):
            well_means = (
                pd.DataFrame({"well": combined.obs["well"].values, "val": combined.X[:, i]})
                .groupby("well")["val"].mean()
            )
            feat[f"well_cv_{ch}"] = float(well_means.std() / (well_means.mean() + 1e-9))
        well_counts = pd.Series(combined.obs["well"].values).value_counts()
        feat["well_cv_total"] = float(well_counts.std() / (well_counts.mean() + 1e-9))
    return feat

features = extract_features(data, n_wells=len(csv_files), combined=combined)
features["dataset"]        = os.path.basename(DATA_DIR)
features["best_n_clusters"] = N_CLUSTERS
features["best_xdim"]      = XDIM
features["best_ydim"]      = YDIM
features["best_rlen"]      = RLEN
features["best_silhouette"] = round(best_score, 6)

if skip_meta:
    print("\nMeta-dataset entry skipped (user requested 'skip').")
else:
    meta_path = os.path.join(META_DIR, "meta_dataset.csv")
    meta_row  = pd.DataFrame([features])
    if os.path.exists(meta_path):
        existing = pd.read_csv(meta_path)
        # Replace row for same dataset if re-run, otherwise append
        existing = existing[existing["dataset"] != features["dataset"]]
        meta_df  = pd.concat([existing, meta_row], ignore_index=True)
    else:
        meta_df = meta_row
    meta_df.to_csv(meta_path, index=False)
    print(f"\nAppended features + best params → {meta_path}  ({len(meta_df)} total rows)")
    print("Run  train_param_predictor.py  once you have collected enough datasets.")

# ── 9. Optuna visualisation (optional, requires plotly) ────────────────────────
try:
    import plotly  # noqa: F401
    fig_imp = optuna.visualization.plot_param_importances(study)
    out_imp = os.path.join(OUT_DIR, "optuna_param_importances.html")
    fig_imp.write_html(out_imp)
    print(f"\nSaved param importance chart → {out_imp}")

    fig_his = optuna.visualization.plot_optimization_history(study)
    out_his = os.path.join(OUT_DIR, "optuna_optimization_history.html")
    fig_his.write_html(out_his)
    print(f"Saved optimisation history  → {out_his}")
except ImportError:
    print("\n(Install plotly for interactive Optuna visualisations: pip install plotly)")
