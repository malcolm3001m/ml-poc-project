"""Retrain all models on v2 features — locks in the winner of experiment.py.

Trains 4 models on the v2 feature set (34 features):
  - decision_tree   (kept for narrative: baseline floor)
  - random_forest   (kept: ensemble)
  - xgboost         (kept: gradient boosting champion of v1)
  - catboost        (NEW: experiment winner — native categorical handling)

Saves all four to models/*.joblib, then writes results/model_metrics.csv.
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from data_v2 import CATEGORICAL_COLS, load_dataset_split  # noqa: E402
from model_io import CatBoostLogWrapper  # noqa: E402

MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)


def _log_target(reg):
    return TransformedTargetRegressor(
        regressor=reg, func=np.log1p, inverse_func=np.expm1, check_inverse=False,
    )


def _metrics(y_true, y_pred):
    pred = np.maximum(np.asarray(y_pred, dtype=float), 0)
    return {
        "mae_eur": float(mean_absolute_error(y_true, pred)),
        "rmse_eur": float(np.sqrt(mean_squared_error(y_true, pred))),
        "r2": float(r2_score(y_true, pred)),
    }


def main() -> None:
    print("Loading v2 dataset...")
    X_train, X_test, y_train, y_test = load_dataset_split()
    print(f"  train: {X_train.shape}  test: {X_test.shape}")

    cat_idx = [list(X_train.columns).index(c) for c in CATEGORICAL_COLS]
    print(f"  categorical column indices: {cat_idx}")

    rows = []

    # Decision Tree
    print("\n[1/4] Decision Tree...")
    dt = _log_target(DecisionTreeRegressor(max_depth=10, min_samples_leaf=20, random_state=42))
    dt.fit(X_train, y_train)
    pred = dt.predict(X_test)
    rows.append({"model_key": "decision_tree", "model_name": "Decision Tree",
                 "model_path": str(MODELS_DIR / "decision_tree.joblib"),
                 **_metrics(y_test, pred)})
    joblib.dump(dt, MODELS_DIR / "decision_tree.joblib")
    print(f"  R² = {rows[-1]['r2']:.3f}  MAE = €{rows[-1]['mae_eur']/1e6:.2f}M")

    # Random Forest
    print("\n[2/4] Random Forest...")
    rf = _log_target(RandomForestRegressor(
        n_estimators=140, max_depth=14, min_samples_leaf=3,
        random_state=42, n_jobs=-1,
    ))
    rf.fit(X_train, y_train)
    pred = rf.predict(X_test)
    rows.append({"model_key": "random_forest", "model_name": "Random Forest",
                 "model_path": str(MODELS_DIR / "random_forest.joblib"),
                 **_metrics(y_test, pred)})
    joblib.dump(rf, MODELS_DIR / "random_forest.joblib")
    print(f"  R² = {rows[-1]['r2']:.3f}  MAE = €{rows[-1]['mae_eur']/1e6:.2f}M")

    # XGBoost
    print("\n[3/4] XGBoost...")
    xgb = _log_target(XGBRegressor(
        objective="reg:squarederror", n_estimators=350, learning_rate=0.05,
        max_depth=4, subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
        random_state=42, n_jobs=-1, tree_method="hist",
    ))
    xgb.fit(X_train, y_train)
    pred = xgb.predict(X_test)
    rows.append({"model_key": "xgboost_regressor", "model_name": "XGBoost Regressor",
                 "model_path": str(MODELS_DIR / "xgboost_regressor.joblib"),
                 **_metrics(y_test, pred)})
    joblib.dump(xgb, MODELS_DIR / "xgboost_regressor.joblib")
    print(f"  R² = {rows[-1]['r2']:.3f}  MAE = €{rows[-1]['mae_eur']/1e6:.2f}M")

    # CatBoost — winner
    print("\n[4/4] CatBoost (winner)...")
    cb = CatBoostLogWrapper(cat_idx)
    cb.fit(X_train, y_train)
    pred = cb.predict(X_test)
    rows.append({"model_key": "catboost_regressor", "model_name": "CatBoost Regressor",
                 "model_path": str(MODELS_DIR / "catboost_regressor.joblib"),
                 **_metrics(y_test, pred)})
    joblib.dump(cb, MODELS_DIR / "catboost_regressor.joblib")
    print(f"  R² = {rows[-1]['r2']:.3f}  MAE = €{rows[-1]['mae_eur']/1e6:.2f}M  ← winner")

    df = pd.DataFrame(rows).sort_values("r2", ascending=False)
    df.to_csv(RESULTS_DIR / "model_metrics.csv", index=False)

    print("\n=== FINAL TEST-SET METRICS ===")
    print(df[["model_name", "mae_eur", "rmse_eur", "r2"]].to_string(index=False))
    print(f"\n📊 Written: results/model_metrics.csv")


if __name__ == "__main__":
    main()
