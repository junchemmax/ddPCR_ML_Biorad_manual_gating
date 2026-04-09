"""
FlowSOM clustering of all ddPCR amplitude CSV files.

Outputs:
  ddPCR_clusters.png        – Ch1 vs Ch2 scatter coloured by FlowSOM metacluster
  ddPCR_cluster_counts.csv  – per-well droplet counts per metacluster
"""
from __future__ import annotations

import glob
import os

import warnings
import anndata as ad
warnings.filterwarnings("ignore", category=FutureWarning, module="flowsom")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import flowsom_rs
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import AgglomerativeClustering

# ── 1. Load all wells ──────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "ddPCR_data/230418_MIP122_TRT01_TA421-8")
csv_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
print(f"Found {len(csv_files)} CSV files.")

adatas = []
for f in csv_files:
    parts = os.path.basename(f).split("_")
    well = parts[-2]          # e.g. "A01"

    df = pd.read_csv(f, skiprows=3)
    df = df[["Ch1Amplitude", "Ch2Amplitude"]].dropna()

    adata = ad.AnnData(df.values.astype(np.float32))
    adata.var_names = ["Ch1Amplitude", "Ch2Amplitude"]
    adata.obs_names = [f"{well}_{i}" for i in range(len(df))]
    adata.obs["well"] = well
    adatas.append(adata)

combined = ad.concat(adatas, merge="same")
print(f"Combined: {combined.n_obs:,} droplets across {len(csv_files)} wells.")

# ── 2. FlowSOM (whole plate) ───────────────────────────────────────────────────
N_CLUSTERS = 4
XDIM       = 10   # default 10. test 8, 10, 12, 14, 16
YDIM       = 10   # default 10. test 8, 10, 12, 14, 16
RLEN       = 10   # default 10. test 5, 10, 25, 50, 100
SEED       = 42

n_nodes    = XDIM * YDIM
data       = combined.X.astype(np.float64)

# SOM grid setup (same format as FlowSOM_Python)
grid       = [(x, y) for x in range(XDIM) for y in range(YDIM)]
nhbrdist   = squareform(pdist(grid, metric="chebyshev"))
radii      = (np.quantile(nhbrdist, 0.67), 0.0)
codes_init = data[np.random.default_rng(SEED).choice(len(data), n_nodes, replace=False)]

# Train SOM with Rust backend (batch-learning, deterministic)
codes, bmu_idx, _ = flowsom_rs.train_batch_som(
    data, codes_init, nhbrdist, radii, rlen=RLEN
)

# Metaclustering: hierarchical clustering of SOM nodes
node_meta = AgglomerativeClustering(n_clusters=N_CLUSTERS, linkage="average").fit_predict(codes)

combined.obs["cluster"]     = bmu_idx + 1
combined.obs["metacluster"] = node_meta[bmu_idx] + 1

print("\nDroplets per metacluster (all wells):")
print(combined.obs["metacluster"].value_counts().sort_index())

# ── 3. Scatter plot ────────────────────────────────────────────────────────────
max_pts = 200_000
rng = np.random.default_rng(0)
idx_plot = rng.choice(combined.n_obs, min(max_pts, combined.n_obs), replace=False)
idx_plot.sort()

X       = combined.X[idx_plot]
mc_plot = combined.obs["metacluster"].values[idx_plot]

fig, ax = plt.subplots(figsize=(6, 5))
for mc in sorted(np.unique(mc_plot)):
    mask = mc_plot == mc
    ax.scatter(X[mask, 1], X[mask, 0], s=0.5, alpha=0.3, label=f"Cluster {mc}", rasterized=True)
ax.set_xlabel("Ch2Amplitude")
ax.set_ylabel("Ch1Amplitude")
ax.legend(markerscale=8, title="Metacluster")
ax.set_title(f"FlowSOM – {len(csv_files)} wells, {combined.n_obs:,} droplets")
param_text = f"xdim={XDIM}  ydim={YDIM}  rlen={RLEN}"
ax.text(0.01, 0.01, param_text, transform=ax.transAxes, fontsize=7,
        verticalalignment="bottom", color="gray", family="monospace")
plt.tight_layout()
out_png = os.path.join(os.path.dirname(__file__), "ddPCR_clusters" + f"_xdim{XDIM}_ydim{YDIM}_rlen{RLEN}.png")
plt.savefig(out_png, dpi=150)
print(f"\nSaved scatter plot → {out_png}")

# ── 4. Per-well counts table ───────────────────────────────────────────────────
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

out_csv = os.path.join(os.path.dirname(__file__), "ddPCR_cluster_counts" + f"_xdim{XDIM}_ydim{YDIM}_rlen{RLEN}.csv")
counts.to_csv(out_csv)
print(f"Saved per-well counts  → {out_csv}")
print("\nPer-well summary (first 10 wells):")
print(counts.head(10).to_string())
