"""
Train a supervised model to predict best FlowSOM parameters (XDIM, YDIM, RLEN)
from ddPCR data statistics collected in meta_dataset.csv.

Each row in meta_dataset.csv corresponds to one plate / dataset that was
previously optimised with optimize_flowsom_params.py.

Outputs (saved next to this script):
  param_predictor.pkl          – trained model (RandomForest, multi-output)
  param_predictor_report.txt   – cross-validation scores and feature importances

Install once:
    pip install scikit-learn joblib
"""
from __future__ import annotations

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_validate, LeaveOneOut
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ── Config ─────────────────────────────────────────────────────────────────────
HERE          = os.path.dirname(__file__)
META_CSV      = os.path.join(HERE, "meta_dataset.csv")
MODEL_OUT     = os.path.join(HERE, "param_predictor.pkl")
REPORT_OUT    = os.path.join(HERE, "param_predictor_report.txt")
MIN_SAMPLES   = 5   # warn if fewer rows (model will be unreliable)

TARGET_COLS   = ["best_n_clusters", "best_xdim", "best_ydim", "best_rlen"]
DROP_COLS     = ["dataset", "best_silhouette", "clear_clustering", "mut_found"] + TARGET_COLS

# ── Load meta-dataset ──────────────────────────────────────────────────────────
if not os.path.exists(META_CSV):
    raise FileNotFoundError(
        f"{META_CSV} not found.\n"
        "Run optimize_flowsom_params.py on several datasets first."
    )

df = pd.read_csv(META_CSV)
print(f"Loaded meta_dataset.csv: {len(df)} rows, {df.shape[1]} columns.")
if len(df) < MIN_SAMPLES:
    print(
        f"\nWARNING: only {len(df)} sample(s). Model may overfit badly.\n"
        f"Collect at least {MIN_SAMPLES} datasets before trusting predictions."
    )

feature_cols = [c for c in df.columns if c not in DROP_COLS]
X = df[feature_cols].values.astype(np.float64)
y = df[TARGET_COLS].values.astype(np.float64)

print(f"Features ({len(feature_cols)}): {feature_cols}")
print(f"Targets : {TARGET_COLS}\n")

# ── Model ─────────────────────────────────────────────────────────────────────
#  RandomForest handles non-linearities well and gives feature importances.
#  MultiOutputRegressor wraps it so all three targets are predicted together.
base_rf = RandomForestRegressor(
    n_estimators=300,
    max_features="sqrt",
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1,
)

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model",  MultiOutputRegressor(base_rf, n_jobs=1)),
])

# ── Cross-validation ──────────────────────────────────────────────────────────
# Use LeaveOneOut when data are scarce, else 5-fold.
cv = LeaveOneOut() if len(df) <= 20 else 5

print(f"Cross-validating ({cv} folds) …")
cv_results = cross_validate(
    pipeline, X, y,
    cv=cv,
    scoring="neg_mean_absolute_error",
    return_train_score=True,
)

mean_mae = -cv_results["test_score"].mean()
std_mae  = cv_results["test_score"].std()
print(f"CV MAE (avg across all targets): {mean_mae:.3f} ± {std_mae:.3f}")
print("  (MAE is in the same units as the targets: grid steps for XDIM/YDIM, iterations for RLEN)")

# ── Train on full data ────────────────────────────────────────────────────────
pipeline.fit(X, y)

# ── Feature importances (averaged across 3 sub-estimators) ───────────────────
importances = np.mean(
    [est.feature_importances_ for est in pipeline.named_steps["model"].estimators_],
    axis=0,
)
imp_df = pd.DataFrame({"feature": feature_cols, "importance": importances})
imp_df = imp_df.sort_values("importance", ascending=False).reset_index(drop=True)
print("\nTop-10 feature importances (averaged across XDIM/YDIM/RLEN models):")
print(imp_df.head(10).to_string(index=False))

# ── Save model ────────────────────────────────────────────────────────────────
joblib.dump({"pipeline": pipeline, "feature_cols": feature_cols}, MODEL_OUT)
print(f"\nModel saved → {MODEL_OUT}")

# ── Save report ───────────────────────────────────────────────────────────────
lines = [
    "FlowSOM Param Predictor – Training Report",
    "=" * 50,
    f"Training samples : {len(df)}",
    f"Features         : {len(feature_cols)}",
    f"CV strategy      : {cv}",
    f"CV MAE (avg)     : {mean_mae:.3f} ± {std_mae:.3f}",
    "",
    "Per-target cross-validated predictions (leave-one-out):",
]

loo      = LeaveOneOut()
y_pred   = np.zeros_like(y, dtype=float)
for train_idx, test_idx in loo.split(X):
    pipeline.fit(X[train_idx], y[train_idx])
    y_pred[test_idx] = pipeline.predict(X[test_idx])

for i, tgt in enumerate(TARGET_COLS):
    mae_t = np.mean(np.abs(y[:, i] - y_pred[:, i]))
    lines.append(f"  {tgt:<12}: MAE = {mae_t:.2f}")

lines += [
    "",
    "Feature importances (averaged across targets):",
]
lines += [f"  {row.feature:<30} {row.importance:.4f}" for _, row in imp_df.iterrows()]

report_text = "\n".join(lines)
with open(REPORT_OUT, "w") as fh:
    fh.write(report_text)
print(f"Report saved  → {REPORT_OUT}")

# Re-fit on full data before saving (LOO loop above replaced the fitted model)
pipeline.fit(X, y)
joblib.dump({"pipeline": pipeline, "feature_cols": feature_cols}, MODEL_OUT)
