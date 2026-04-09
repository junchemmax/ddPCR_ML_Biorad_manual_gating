"""
Predict best FlowSOM parameters (XDIM, YDIM, RLEN) for a new ddPCR dataset
using the supervised model trained by train_param_predictor.py.

Usage:
    python predict_params.py  <path/to/data_folder>

The data folder should contain per-well *_Amplitude.csv files (same format as
the training data).  The script prints the predicted parameters and optionally
launches optimize_flowsom_params.py with those values as warm-start hints.
"""
from __future__ import annotations

import glob
import os
import sys
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import joblib
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

HERE      = os.path.dirname(os.path.abspath(__file__))
MODEL_PKL = os.path.join(HERE, "param_predictor.pkl")


# ── Feature extraction (must match optimize_flowsom_params.py) ─────────────────
def extract_features(data: np.ndarray, n_wells: int) -> dict:
    feat: dict = {}
    feat["n_droplets"] = len(data)
    feat["n_wells"]    = n_wells
    for i, ch in enumerate(["ch1", "ch2"]):
        vals = data[:, i]
        feat[f"{ch}_mean"]   = float(np.mean(vals))
        feat[f"{ch}_std"]    = float(np.std(vals))
        feat[f"{ch}_p5"]     = float(np.percentile(vals, 5))
        feat[f"{ch}_p25"]    = float(np.percentile(vals, 25))
        feat[f"{ch}_median"] = float(np.median(vals))
        feat[f"{ch}_p75"]    = float(np.percentile(vals, 75))
        feat[f"{ch}_p95"]    = float(np.percentile(vals, 95))
        feat[f"{ch}_iqr"]    = feat[f"{ch}_p75"] - feat[f"{ch}_p25"]
        feat[f"{ch}_range"]  = float(vals.max() - vals.min())
        feat[f"{ch}_cv"]     = feat[f"{ch}_std"] / (feat[f"{ch}_mean"] + 1e-9)
        feat[f"{ch}_skew"]   = float(skew(vals))
        feat[f"{ch}_kurt"]   = float(kurtosis(vals))
    feat["ch1_ch2_corr"] = float(np.corrcoef(data[:, 0], data[:, 1])[0, 1])
    return feat


# ── Load data ──────────────────────────────────────────────────────────────────
if len(sys.argv) < 2:
    print("Usage: python predict_params.py <path/to/data_folder>")
    sys.exit(1)

data_dir  = sys.argv[1]
csv_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
if not csv_files:
    print(f"No CSV files found in: {data_dir}")
    sys.exit(1)

print(f"Loading {len(csv_files)} CSV files from: {data_dir}")
frames = []
for f in csv_files:
    df = pd.read_csv(f, skiprows=3)[["Ch1Amplitude", "Ch2Amplitude"]].dropna()
    frames.append(df.values.astype(np.float64))
data = np.concatenate(frames, axis=0)
print(f"Total droplets: {len(data):,}")

# ── Load model ─────────────────────────────────────────────────────────────────
if not os.path.exists(MODEL_PKL):
    print(
        f"\nModel not found: {MODEL_PKL}\n"
        "Train it first:\n"
        "  1. Run optimize_flowsom_params.py on several datasets to build meta_dataset.csv\n"
        "  2. Run train_param_predictor.py\n"
    )
    sys.exit(1)

bundle       = joblib.load(MODEL_PKL)
pipeline     = bundle["pipeline"]
feature_cols = bundle["feature_cols"]

# ── Predict ────────────────────────────────────────────────────────────────────
features = extract_features(data, n_wells=len(csv_files))
X_new    = np.array([[features[c] for c in feature_cols]])

pred = pipeline.predict(X_new)[0]

# Round to nearest even integer for XDIM / YDIM; nearest int for RLEN
def round_even(v: float) -> int:
    i = int(round(v))
    return i if i % 2 == 0 else i + 1

xdim_pred = round_even(pred[0])
ydim_pred = round_even(pred[1])
rlen_pred = max(5, int(round(pred[2])))

print("\n── Predicted FlowSOM parameters ──────────────────")
print(f"  XDIM = {xdim_pred}")
print(f"  YDIM = {ydim_pred}")
print(f"  RLEN = {rlen_pred}")
print("──────────────────────────────────────────────────")
print("\nCopy these values into run_ddpcr_flowsom_flowsom_rs.py, or")
print("use them as starting point for optimize_flowsom_params.py.")

# ── Raw confidence hint ────────────────────────────────────────────────────────
meta_csv = os.path.join(HERE, "meta_dataset.csv")
if os.path.exists(meta_csv):
    n_train = len(pd.read_csv(meta_csv))
    if n_train < 10:
        print(
            f"\nNote: model was trained on only {n_train} dataset(s). "
            "Predictions are rough estimates — verify with Optuna optimisation."
        )
    else:
        print(f"\n(Model trained on {n_train} datasets.)")
