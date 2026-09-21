"""
Evaluate/train the cluster-count classifier on the large, ClusterData-derived
dataset (flowsom/output/full_cluster_dataset.csv, ~8.7k wells) instead of the
small manually-labeled meta_dataset.csv (222 wells).

This is a comparison/diagnostic script: it reports 5-fold CV accuracy so we can
decide whether to fold this into train_param_predictor.py_'s cluster model.

Usage:
    python flowsom/train_cluster_classifier_full.py
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

HERE     = os.path.dirname(os.path.abspath(__file__))
FULL_CSV = os.path.join(HERE, "output", "full_cluster_dataset.csv")

DROP_COLS = ["dataset", "best_n_clusters"]

df = pd.read_csv(FULL_CSV)
print(f"Loaded full_cluster_dataset.csv: {len(df)} rows, {df.shape[1]} columns.")
print("\nbest_n_clusters distribution:")
print(df["best_n_clusters"].value_counts().sort_index().to_string())

feature_cols = [c for c in df.columns if c not in DROP_COLS]
X = df[feature_cols].values.astype(np.float64)
y = df["best_n_clusters"].values.astype(int)

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", RandomForestClassifier(
        n_estimators=300,
        max_features="sqrt",
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )),
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
print(f"\nCross-validating (5-fold, n={len(df)}) …")
results = cross_validate(
    pipeline, X, y, cv=cv,
    scoring=["accuracy", "f1_macro"],
    n_jobs=1,
)
print(f"CV accuracy : {results['test_accuracy'].mean():.3f} +/- {results['test_accuracy'].std():.3f}")
print(f"CV f1_macro : {results['test_f1_macro'].mean():.3f} +/- {results['test_f1_macro'].std():.3f}")
