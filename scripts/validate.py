"""Diagnostic validation — overfit, CV, baselines, leakage ablation, per-position errors.

Writes:
    results/validation_metrics.csv  — one row per (model_or_baseline) with all diagnostics
    results/ablation.csv             — XGBoost with vs without the suspected leak feature
    results/per_position_mae.csv     — MAE broken down by player position
    results/residuals.csv            — (actual, predicted, position, name) on the test set
                                       for the best model — powers the app's residual plot
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data import (  # noqa: E402
    CATEGORICAL_COLS,
    NUMERIC_COLS,
    TARGET,
    _load_appearances,
    _load_players,
    load_dataset_split,
)
from model_io import CatBoostLogWrapper  # noqa: E402

RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)


# ---------- Helpers ----------

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


def _build_models(cat_indices=None):
    """Same hyperparameters as scripts/retrain_v2.py — keeps validate.py honest."""
    return {
        "decision_tree": _log_target(
            DecisionTreeRegressor(max_depth=10, min_samples_leaf=20, random_state=42)
        ),
        "random_forest": _log_target(
            RandomForestRegressor(
                n_estimators=140, max_depth=14, min_samples_leaf=3,
                random_state=42, n_jobs=-1,
            )
        ),
        "xgboost_regressor": _log_target(
            XGBRegressor(
                objective="reg:squarederror", n_estimators=350, learning_rate=0.05,
                max_depth=4, subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
                random_state=42, n_jobs=-1, tree_method="hist",
            )
        ),
        "catboost_regressor": CatBoostLogWrapper(cat_indices=cat_indices),
    }


def _cat_indices(columns):
    return [list(columns).index(c) for c in CATEGORICAL_COLS]


# ---------- Main routines ----------

def overfit_and_test(X_train, X_test, y_train, y_test) -> list[dict]:
    """Train each model, score on train + test, return overfit gap."""
    rows = []
    for key, model in _build_models(_cat_indices(X_train.columns)).items():
        model.fit(X_train, y_train)
        train_m = _metrics(y_train, model.predict(X_train))
        test_m = _metrics(y_test, model.predict(X_test))
        rows.append({
            "model": key,
            "train_r2": train_m["r2"],
            "test_r2": test_m["r2"],
            "overfit_gap": train_m["r2"] - test_m["r2"],
            "train_mae_eur": train_m["mae_eur"],
            "test_mae_eur": test_m["mae_eur"],
        })
    return rows


def cross_validate(X, y) -> list[dict]:
    """5-fold CV R² for each model — measures stability."""
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    rows = []
    # CatBoost can't be parallelised across CV folds with n_jobs=-1 (model state
    # gets shared incorrectly). Use n_jobs=1 for CatBoost, parallel for the rest.
    for key, model in _build_models(_cat_indices(X.columns)).items():
        nj = 1 if key == "catboost_regressor" else -1
        scores = cross_val_score(model, X, y, cv=cv, scoring="r2", n_jobs=nj)
        rows.append({
            "model": key,
            "cv_r2_mean": float(scores.mean()),
            "cv_r2_std": float(scores.std()),
            "cv_r2_min": float(scores.min()),
            "cv_r2_max": float(scores.max()),
        })
    return rows


def baselines(X_train, X_test, y_train, y_test) -> list[dict]:
    """Predict-median + linear regression — establishes the floor."""
    rows = []

    median_model = DummyRegressor(strategy="median")
    median_model.fit(X_train, y_train)
    rows.append({
        "baseline": "predict_median",
        **_metrics(y_test, median_model.predict(X_test)),
    })

    linreg = _log_target(LinearRegression())
    linreg.fit(X_train, y_train)
    rows.append({
        "baseline": "linear_regression_log_target",
        **_metrics(y_test, linreg.predict(X_test)),
    })

    return rows


def leakage_ablation(X_train, X_test, y_train, y_test) -> list[dict]:
    """Re-train XGBoost without the suspected leak feature — report delta."""
    leak_col = "current_club_domestic_competition_id"

    full = _build_models()["xgboost_regressor"]
    full.fit(X_train, y_train)
    full_m = _metrics(y_test, full.predict(X_test))

    X_train_ab = X_train.drop(columns=[leak_col])
    X_test_ab = X_test.drop(columns=[leak_col])
    ablated = _build_models()["xgboost_regressor"]
    ablated.fit(X_train_ab, y_train)
    ablated_m = _metrics(y_test, ablated.predict(X_test_ab))

    return [
        {"variant": "xgboost_full", **full_m},
        {"variant": "xgboost_without_current_club_competition", **ablated_m},
        {
            "variant": "delta",
            "mae_eur": ablated_m["mae_eur"] - full_m["mae_eur"],
            "rmse_eur": ablated_m["rmse_eur"] - full_m["rmse_eur"],
            "r2": ablated_m["r2"] - full_m["r2"],
        },
    ]


def per_position_with_model(model, X_test, y_test) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute per-position MAE + residuals using an already-fitted model."""
    raw = _load_players().merge(_load_appearances(), on="player_id", how="left")
    raw = raw.dropna(subset=[TARGET])
    positions = raw.loc[X_test.index, "position"].fillna("Unknown")
    names = raw.loc[X_test.index, "name"].fillna("Unknown")

    preds = np.maximum(model.predict(X_test), 0)
    df = pd.DataFrame({
        "name": names.values,
        "position": positions.values,
        "actual_eur": y_test.values,
        "predicted_eur": preds,
    })
    df["abs_error_eur"] = (df["predicted_eur"] - df["actual_eur"]).abs()

    per_pos = (
        df.groupby("position")
        .agg(
            n=("actual_eur", "size"),
            median_actual_eur=("actual_eur", "median"),
            mae_eur=("abs_error_eur", "mean"),
        )
        .reset_index()
        .sort_values("mae_eur")
    )

    return per_pos, df


# ---------- Entry point ----------

def main() -> None:
    print("Loading dataset...")
    X_train, X_test, y_train, y_test = load_dataset_split()
    X_all = pd.concat([X_train, X_test])
    y_all = pd.concat([y_train, y_test])
    print(f"  train={len(X_train):,}  test={len(X_test):,}  features={X_train.shape[1]}")

    print("\n[1/5] Overfit check (train R² vs test R²)...")
    overfit_rows = overfit_and_test(X_train, X_test, y_train, y_test)
    for r in overfit_rows:
        print(f"  {r['model']:>20s}  train={r['train_r2']:.3f}  test={r['test_r2']:.3f}  gap={r['overfit_gap']:+.3f}")

    print("\n[2/5] 5-fold cross-validation...")
    cv_rows = cross_validate(X_all, y_all)
    for r in cv_rows:
        print(f"  {r['model']:>20s}  R² {r['cv_r2_mean']:.3f} ± {r['cv_r2_std']:.3f}  "
              f"[min={r['cv_r2_min']:.3f}, max={r['cv_r2_max']:.3f}]")

    print("\n[3/5] Baselines (median + linear regression)...")
    base_rows = baselines(X_train, X_test, y_train, y_test)
    for r in base_rows:
        print(f"  {r['baseline']:>30s}  R²={r['r2']:.3f}  MAE=€{r['mae_eur']/1e6:.2f}M")

    print("\n[4/5] Leakage ablation — XGBoost ± current_club_domestic_competition_id...")
    ab_rows = leakage_ablation(X_train, X_test, y_train, y_test)
    for r in ab_rows:
        print(f"  {r['variant']:>50s}  R²={r['r2']:+.3f}  MAE=€{r['mae_eur']/1e6:+.2f}M")

    print("\n[5/5] Per-position MAE (using fitted XGBoost)...")
    final_xgb = _build_models()["xgboost_regressor"]
    final_xgb.fit(X_train, y_train)
    per_pos, residuals = per_position_with_model(final_xgb, X_test, y_test)
    print(per_pos.to_string(index=False))

    # ---- Write outputs ----
    flat = []
    for r in overfit_rows:
        flat.append({"kind": "overfit", "name": r["model"], **{k: v for k, v in r.items() if k != "model"}})
    for r in cv_rows:
        flat.append({"kind": "cv", "name": r["model"], **{k: v for k, v in r.items() if k != "model"}})
    for r in base_rows:
        flat.append({"kind": "baseline", "name": r["baseline"], **{k: v for k, v in r.items() if k != "baseline"}})

    pd.DataFrame(flat).to_csv(RESULTS / "validation_metrics.csv", index=False)
    pd.DataFrame(ab_rows).to_csv(RESULTS / "ablation.csv", index=False)
    per_pos.to_csv(RESULTS / "per_position_mae.csv", index=False)
    residuals.to_csv(RESULTS / "residuals.csv", index=False)

    print("\n✅ Written:")
    print("  results/validation_metrics.csv")
    print("  results/ablation.csv")
    print("  results/per_position_mae.csv")
    print("  results/residuals.csv")


if __name__ == "__main__":
    main()
