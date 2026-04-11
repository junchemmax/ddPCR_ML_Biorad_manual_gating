"""
Bayesian hyperparameter optimisation for FlowSOM XDIM / YDIM / RLEN.
Folder version – iterates over all Amplitude CSV files in CSV_FOLDER.

Usage
-----
Edit CSV_FOLDER below, then run:
    python optimize_flowsom_rs_params_single.py

Install once:
    pip install optuna scipy anndata flowsom-rs scikit-learn matplotlib pandas
"""
from __future__ import annotations

import argparse
import glob
import os
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import anndata as ad
import flowsom_rs
import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
plt.style.use("default")
import numpy as np
import optuna
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import kurtosis, skew
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

# ── Config ─────────────────────────────────────────────────────────────────────
_parser = argparse.ArgumentParser(description="FlowSOM Bayesian optimisation over a folder of Amplitude CSVs.")
_parser.add_argument(
    "folder",
    help="Path to folder containing *_Amplitude.csv files",
)
_parser.add_argument(
    "--trials",
    type=int,
    default=40,
    help="Number of Optuna trials (default: 40)",
)
_args = _parser.parse_args()
CSV_FOLDER = os.path.normpath(_args.folder)
N_CLUST_RANGE = (2, 4)   # n_clusters is optimised by Optuna
SEED          = 42       # reproducibility
N_TRIALS      = _args.trials  # Optuna trials (≈2-5 min on a laptop; override with --trials)
EVAL_PTS      = 50_000   # subsample size for silhouette (keeps it fast)
N_JOBS        = -1       # parallel jobs (set to 1 if you get memory issues)

# Search space – feel free to widen
XDIM_RANGE = (6, 16)     # even integers only
YDIM_RANGE = (6, 16)
RLEN_RANGE = (5, 100)

# meta_dataset.csv lives next to this script so it accumulates across datasets
META_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(META_DIR, exist_ok=True)

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── Module-level state updated per CSV file (used by objective / _save_trial_png)
data      = None
eval_idx  = None
eval_data = None
plot_idx  = None
plot_data = None
TRIALS_DIR = ""

# ── Helper: run FlowSOM ───────────────────────────────────────────────────────
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
    """Save a scatter plot for one Optuna trial (thread-safe: no pyplot globals)."""
    X_p  = plot_data
    mc_p = (node_meta[bmu_idx] + 1)[plot_idx]

    # Use Figure() directly — avoids the shared pyplot figure registry
    fig = Figure(figsize=(5, 4), facecolor="white")
    ax  = fig.add_subplot(111, facecolor="white")
    for mc in sorted(np.unique(mc_p)):
        mask = mc_p == mc
        ax.scatter(X_p[mask, 1], X_p[mask, 0], s=3, alpha=0.5, label=f"C{mc}")
    ax.set_xlabel("Ch2Amplitude")
    ax.set_ylabel("Ch1Amplitude")
    ax.legend(markerscale=6, title="MC", fontsize=7)
    ax.set_title(
        f"Trial {trial_number:03d}  xdim={xdim} ydim={ydim} rlen={rlen} nc={n_clusters}  "
        f"sil={score:.4f}",
        fontsize=9,
    )
    fig.tight_layout()
    fname = (
        f"trial_sil{score:+.4f}_t{trial_number:03d}"
        f"_xdim{xdim}_ydim{ydim}_rlen{rlen}_nc{n_clusters}.png"
    )
    fig.savefig(os.path.join(TRIALS_DIR, fname), dpi=100, facecolor="white")


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


# ── Per-file processing ────────────────────────────────────────────────────────
def process_csv(csv_file: str) -> None:
    """Run the full Optuna + FlowSOM pipeline for a single CSV file."""
    global data, eval_idx, eval_data, plot_idx, plot_data, TRIALS_DIR

    # ── 1. Load CSV ────────────────────────────────────────────────────────────
    _csv_stem  = os.path.splitext(os.path.basename(csv_file))[0]
    out_dir    = os.path.join(os.path.dirname(csv_file), "output", _csv_stem)
    TRIALS_DIR = os.path.join(out_dir, "trials")
    os.makedirs(TRIALS_DIR, exist_ok=True)

    parts = os.path.basename(csv_file).split("_")
    well  = parts[-2]

    df       = pd.read_csv(csv_file, skiprows=3)[["Ch1Amplitude", "Ch2Amplitude"]].dropna()
    adata    = ad.AnnData(df.values.astype(np.float32))
    adata.var_names = ["Ch1Amplitude", "Ch2Amplitude"]
    adata.obs_names = [f"{well}_{i}" for i in range(len(df))]
    adata.obs["well"] = well

    combined = adata
    data     = combined.X.astype(np.float64)
    print(f"Loaded {combined.n_obs:,} droplets from '{os.path.basename(csv_file)}' (well={well}).\n")

    # Pre-draw fixed subsample indices (shared across all trials for fair comparison)
    rng       = np.random.default_rng(SEED)
    eval_idx  = rng.choice(len(data), min(EVAL_PTS, len(data)), replace=False)
    eval_data = data[eval_idx]

    PLOT_PTS  = 50_000
    plot_idx  = np.random.default_rng(SEED + 1).choice(len(data), min(PLOT_PTS, len(data)), replace=False)
    plot_idx.sort()
    plot_data = data[plot_idx]

    # ── 3. Run the study ───────────────────────────────────────────────────────
    print(f"Running Optuna study ({N_TRIALS} trials) …")
    sampler = optuna.samplers.TPESampler(seed=SEED)
    study   = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True, n_jobs=N_JOBS)

    print(f"Trial PNGs saved → {TRIALS_DIR}  (sort by filename to rank by silhouette score)")

    auto_best  = study.best_params
    auto_score = study.best_value
    print(f"\nBest parameters found (by silhouette score):")
    print(f"  XDIM       = {auto_best['xdim']}")
    print(f"  YDIM       = {auto_best['ydim']}")
    print(f"  RLEN       = {auto_best['rlen']}")
    print(f"  N_CLUSTERS = {auto_best['n_clusters']}")
    print(f"  Silhouette Score = {auto_score:.4f}")

    # ── 4. Top-5 trials summary ────────────────────────────────────────────────
    print("\nTop-5 trials:")
    trials_df = study.trials_dataframe()[["number", "value", "params_xdim", "params_ydim", "params_rlen", "params_n_clusters"]]
    trials_df = trials_df.rename(columns={"value": "silhouette"}).sort_values("silhouette", ascending=False)
    print(trials_df.head(5).to_string(index=False))

    # ── 4b. Manual override ────────────────────────────────────────────────────
    print(f"\nReview the PNGs in: {TRIALS_DIR}")
    print(f"The automatic best is #{study.best_trial.number} with silhouette {auto_score:.4f}.")
    print("Press Enter to accept the automatic best, type a trial number to use instead, '1cluster' to mark as single-cluster, or 'skip' to skip saving to meta-dataset.")
    _choice = input("Manual trial number / '1cluster' / 'skip' (or Enter to accept best): ").strip()

    skip_meta      = False
    is_one_cluster = False
    if _choice == "":
        best       = auto_best
        best_score = auto_score
        print("Using automatic best.")
    elif _choice.lower() == "skip":
        best       = auto_best
        best_score = auto_score
        skip_meta  = True
        print("Skipping meta-dataset entry.")
    elif _choice.lower() == "1cluster":
        best           = {**auto_best, "n_clusters": 1}
        best_score     = float("nan")
        is_one_cluster = True
        print("Marking as single-cluster (all droplets → metacluster 1).")
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

    # ── 5. Final clustering with best params ───────────────────────────────────
    XDIM, YDIM, RLEN, N_CLUSTERS = best["xdim"], best["ydim"], best["rlen"], best["n_clusters"]
    print(f"\nRunning final clustering: xdim={XDIM}, ydim={YDIM}, rlen={RLEN}, n_clusters={N_CLUSTERS} …")

    if is_one_cluster:
        combined.obs["cluster"]     = 1
        combined.obs["metacluster"] = 1
    else:
        _, node_meta, bmu_idx = _run_flowsom(XDIM, YDIM, RLEN, N_CLUSTERS)
        combined.obs["cluster"]     = bmu_idx + 1
        combined.obs["metacluster"] = node_meta[bmu_idx] + 1

    print("\nDroplets per metacluster:")
    print(combined.obs["metacluster"].value_counts().sort_index())

    # ── 6. Per-droplet counts table + quadrant assignment ─────────────────────
    counts = combined.obs["metacluster"].value_counts().sort_index().rename("count").to_frame()
    counts["pct"] = (counts["count"] / counts["count"].sum() * 100).round(2)
    _X_all = np.asarray(combined.X, dtype=np.float64)
    _mc_all = np.asarray(combined.obs["metacluster"].values, dtype=int)
    for mc in sorted(np.unique(_mc_all)):
        mask = _mc_all == mc
        for i, ch in enumerate(["ch1", "ch2"]):
            vals = _X_all[mask, i]
            mean_v = float(np.mean(vals))
            std_v  = float(np.std(vals))
            counts.loc[mc, f"{ch}_centroid"] = round(mean_v, 2)
            counts.loc[mc, f"{ch}_std"]      = round(std_v, 2)
            counts.loc[mc, f"{ch}_cv_pct"]   = round(std_v / mean_v * 100, 2) if mean_v != 0 else float("nan")
            counts.loc[mc, f"{ch}_min"]      = round(float(np.min(vals)), 2)
            counts.loc[mc, f"{ch}_p25"]      = round(float(np.percentile(vals, 25)), 2)
            counts.loc[mc, f"{ch}_median"]   = round(float(np.median(vals)), 2)
            counts.loc[mc, f"{ch}_p75"]      = round(float(np.percentile(vals, 75)), 2)
            counts.loc[mc, f"{ch}_max"]      = round(float(np.max(vals)), 2)

    # ── Manual quadrant assignment ─────────────────────────────────────────────
    # Quadrant convention: 1=Ch1+Ch2+, 2=Ch1+Ch2-, 3=Ch1-Ch2-, 4=Ch1-Ch2+
    print("\nAssign quadrant (1\u20134) to each metacluster.")
    print("  1=Ch1+Ch2+  2=Ch1+Ch2-  3=Ch1-Ch2-  4=Ch1-Ch2+  (blank = NaN)")
    print(counts[["count", "pct", "ch1_centroid", "ch2_centroid"]].to_string())
    _mc_list = sorted(np.unique(_mc_all))
    _quads: dict = {}
    for mc in _mc_list:
        while True:
            _ans = input(f"  Metacluster {mc} quadrant [1/2/3/4 or Enter=NaN]: ").strip()
            if _ans == "":
                _quads[mc] = float("nan")
                break
            elif _ans in ("1", "2", "3", "4"):
                _quads[mc] = int(_ans)
                break
            else:
                print("    Please enter 1, 2, 3, 4, or press Enter for NaN.")
    counts["quadrant"] = [_quads[mc] for mc in counts.index]

    # ── 7. Scatter plot with quadrant annotations ──────────────────────────────
    max_pts   = 200_000
    _plot_idx = np.random.default_rng(0).choice(combined.n_obs, min(max_pts, combined.n_obs), replace=False)
    _plot_idx.sort()
    X       = np.asarray(combined.X[_plot_idx], dtype=np.float64)
    mc_plot = np.asarray(combined.obs["metacluster"].values[_plot_idx], dtype=int)

    plt.close("all")          # discard figures left over from parallel workers
    plt.style.use("default")  # restore clean rcParams after parallel corruption
    fig, ax = plt.subplots(figsize=(6, 5), facecolor="white")
    ax.set_facecolor("white")
    _quad_label = {1: "Q1(++)", 2: "Q2(+-)", 3: "Q3(--)", 4: "Q4(-+)"}
    for mc in sorted(np.unique(mc_plot)):
        mask = mc_plot == mc
        _q = _quads.get(mc)
        _qlabel = f" {_quad_label[_q]}" if isinstance(_q, int) else ""
        ax.scatter(X[mask, 1], X[mask, 0], s=5, alpha=0.6, label=f"MC{mc}{_qlabel}")
        _cx = float(counts.loc[mc, "ch2_centroid"])
        _cy = float(counts.loc[mc, "ch1_centroid"])
        _tag = f"Q{_q}" if isinstance(_q, int) else "?"
        ax.annotate(
            f"MC{mc}\n{_tag}",
            xy=(_cx, _cy), fontsize=8, fontweight="bold", ha="center", va="center",
            color="black",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.7),
        )
    ax.set_xlabel("Ch2Amplitude")
    ax.set_ylabel("Ch1Amplitude")
    ax.legend(markerscale=8, title="Metacluster")
    ax.set_title(f"FlowSOM (optimised) \u2013 {os.path.basename(csv_file)}  {combined.n_obs:,} droplets")
    _sil_str   = "nan" if np.isnan(best_score) else f"{best_score:.3f}"
    param_text = (
        f"xdim={XDIM}  ydim={YDIM}  rlen={RLEN}  nc={N_CLUSTERS}"
        f"  sil={_sil_str}  trials={N_TRIALS}"
    )
    ax.text(0.01, 0.01, param_text, transform=ax.transAxes, fontsize=7,
            verticalalignment="bottom", color="gray", family="monospace")
    fig.tight_layout()

    tag     = f"_opt_xdim{XDIM}_ydim{YDIM}_rlen{RLEN}_nc{N_CLUSTERS}"
    out_png = os.path.join(out_dir, f"ddPCR_clusters{tag}.png")
    fig.savefig(out_png, dpi=150, facecolor="white", bbox_inches="tight")
    print(f"\nSaved scatter plot \u2192 {out_png}")

    # ── 8. Save counts CSV ─────────────────────────────────────────────────────
    out_csv = os.path.join(out_dir, f"ddPCR_cluster_counts{tag}.csv")
    counts.to_csv(out_csv)
    print(f"Saved cluster counts  → {out_csv}")
    print(counts.to_string())

    # ── 9. Append to meta-dataset (for supervised predictor training) ──────────
    def extract_features(X: np.ndarray) -> dict:
        """Compute summary statistics from raw amplitude data for meta-learning."""
        feat: dict = {}
        feat["n_droplets"] = len(X)
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
            feat[f"{ch}_bimodality"] = (feat[f"{ch}_skew"] ** 2 + 1) / (feat[f"{ch}_kurt"] + 3 + 1e-9)
            log_vals = np.log1p(np.clip(vals, 0, None))
            feat[f"{ch}_log_std"]  = float(np.std(log_vals))
            feat[f"{ch}_log_skew"] = float(skew(log_vals))
            mu, sigma = feat[f"{ch}_mean"], feat[f"{ch}_std"]
            feat[f"{ch}_outlier_frac"] = float(np.mean((vals < mu - 3 * sigma) | (vals > mu + 3 * sigma)))
        feat["ch1_ch2_corr"] = float(np.corrcoef(X[:, 0], X[:, 1])[0, 1])
        return feat

    features = extract_features(data)
    features["dataset"]          = os.path.splitext(os.path.basename(csv_file))[0]
    features["best_n_clusters"]  = N_CLUSTERS
    features["best_xdim"]        = float("nan") if is_one_cluster else XDIM
    features["best_ydim"]        = float("nan") if is_one_cluster else YDIM
    features["best_rlen"]        = float("nan") if is_one_cluster else RLEN
    features["best_silhouette"]  = float("nan") if is_one_cluster else round(best_score, 6)

    if skip_meta:
        print("\nMeta-dataset entry skipped (user requested 'skip').")
    else:
        # ── Manual label inputs ────────────────────────────────────────────────
        def _ask_bool(prompt: str) -> bool:
            while True:
                ans = input(f"{prompt} [y/n]: ").strip().lower()
                if ans in ("y", "yes", "true", "1"):
                    return True
                if ans in ("n", "no", "false", "0"):
                    return False
                print("  Please enter y or n.")

        print("\nManual labels for this well:")
        features["clear_clustering"]            = _ask_bool("  Clear clustering?")
        features["mut_found"]                   = _ask_bool("  Mut found?")
        meta_row = pd.DataFrame([features])

        def _append_to_meta(fname: str) -> None:
            path = os.path.join(META_DIR, fname)
            if os.path.exists(path):
                existing = pd.read_csv(path)
                existing = existing[existing["dataset"] != features["dataset"]]
                df = pd.concat([existing, meta_row], ignore_index=True)
            else:
                df = meta_row
            df.to_csv(path, index=False)
            print(f"\nAppended features + best params → {path}  ({len(df)} total rows)")

        _append_to_meta("meta_dataset.csv")
        print("Run  train_param_predictor.py  once you have collected enough datasets.")

    # ── 9. Optuna visualisation (optional, requires plotly) ────────────────────
    try:
        import plotly  # noqa: F401
        fig_imp = optuna.visualization.plot_param_importances(study)
        out_imp = os.path.join(out_dir, "optuna_param_importances.html")
        fig_imp.write_html(out_imp)
        print(f"\nSaved param importance chart → {out_imp}")

        fig_his = optuna.visualization.plot_optimization_history(study)
        out_his = os.path.join(out_dir, "optuna_optimization_history.html")
        fig_his.write_html(out_his)
        print(f"Saved optimisation history  → {out_his}")
    except ImportError:
        print("\n(Install plotly for interactive Optuna visualisations: pip install plotly)")


# ── Main: iterate over all CSV files in CSV_FOLDER ────────────────────────────
csv_files = sorted(glob.glob(os.path.join(CSV_FOLDER, "*.csv")))
if not csv_files:
    raise FileNotFoundError(f"No CSV files found in: {CSV_FOLDER}")

print(f"Found {len(csv_files)} CSV file(s) in '{os.path.basename(os.path.normpath(CSV_FOLDER))}'")

for _i, _csv_file in enumerate(csv_files, 1):
    print(f"\n{'=' * 70}")
    print(f"[{_i}/{len(csv_files)}]  {os.path.basename(_csv_file)}")
    print(f"{'=' * 70}")
    process_csv(_csv_file)

print("\nAll files processed.")

