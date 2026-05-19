"""Systematic comparison of model variants — the deep dive.

Trains 7 variants and scores each via a metric suite that captures what
users actually care about, not just R².

Variants:
  1. v1-XGB         baseline (current production)
  2. v2-XGB         v1 model on v2 features (does feature engineering help?)
  3. v2-LightGBM    LGBM on v2 features (different tree algo)
  4. v2-CatBoost    CatBoost with NATIVE categorical handling (no label encoding)
  5. v2-XGB-Calib   v2-XGB + isotonic post-hoc calibration (anti regression-to-mean)
  6. v2-XGB-Tweedie Tweedie objective (designed for skewed positive targets)
  7. v2-Stack       Average of (v2-XGB-Calib + v2-CatBoost) — ensemble

Metrics per variant:
  r2_eur           R² in raw € (the current headline metric)
  r2_log           R² in log space (closer to what % accuracy feels like)
  mae_eur          MAE in raw €
  mdape            Median Absolute Percentage Error (the user-felt metric)
  spearman         Rank correlation
  calib_ratio      max(pred) / max(actual) — anti regression-to-mean
  range_ratio      (p95 pred - p5 pred) / (p95 actual - p5 actual)
  mdape_micro      MdAPE on <€200k bucket
  mdape_low        MdAPE on €200k–1M bucket
  mdape_mid        MdAPE on €1–5M bucket
  mdape_high       MdAPE on €5–20M bucket
  mdape_elite      MdAPE on >€20M bucket
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.compose import TransformedTargetRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_absolute_error, r2_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)


# ---------- Metrics ----------

BUCKETS = [
    (0, 200_000, "micro"),
    (200_000, 1_000_000, "low"),
    (1_000_000, 5_000_000, "mid"),
    (5_000_000, 20_000_000, "high"),
    (20_000_000, 1e10, "elite"),
]


def evaluate(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.maximum(np.asarray(y_pred, dtype=float), 0.0)
    safe_true = np.maximum(y_true, 1)

    out = {
        "model": name,
        "r2_eur": float(r2_score(y_true, y_pred)),
        "r2_log": float(r2_score(np.log1p(y_true), np.log1p(y_pred))),
        "mae_eur": float(mean_absolute_error(y_true, y_pred)),
        "mdape": float(np.median(np.abs(y_pred - y_true) / safe_true * 100)),
        "spearman": float(spearmanr(y_true, y_pred).statistic),
        "calib_ratio": float(np.max(y_pred) / np.max(y_true)),
        "range_ratio": float(
            (np.percentile(y_pred, 95) - np.percentile(y_pred, 5))
            / (np.percentile(y_true, 95) - np.percentile(y_true, 5))
        ),
    }
    for lo, hi, label in BUCKETS:
        mask = (y_true >= lo) & (y_true < hi)
        if mask.sum() > 5:
            mdape = np.median(np.abs(y_pred[mask] - y_true[mask]) / safe_true[mask] * 100)
            out[f"mdape_{label}"] = float(mdape)
            out[f"n_{label}"] = int(mask.sum())
        else:
            out[f"mdape_{label}"] = np.nan
            out[f"n_{label}"] = int(mask.sum())
    return out


def fmt_results(rows: list[dict]) -> str:
    df = pd.DataFrame(rows)
    keep_cols = ["model", "r2_eur", "r2_log", "mae_eur", "mdape", "spearman",
                 "calib_ratio", "range_ratio",
                 "mdape_micro", "mdape_low", "mdape_mid", "mdape_high", "mdape_elite"]
    df = df[keep_cols]
    lines = []
    lines.append(f"{'model':<22s} {'r2€':>6s} {'r2log':>6s} {'MAE€':>8s} {'MdAPE':>6s} {'rho':>6s} {'cal':>5s} {'rng':>5s} {'<200k':>6s} {'<1M':>6s} {'<5M':>6s} {'<20M':>6s} {'>20M':>6s}")
    lines.append("-" * 130)
    for _, r in df.iterrows():
        lines.append(
            f"{r['model']:<22s} "
            f"{r['r2_eur']:>6.3f} {r['r2_log']:>6.3f} "
            f"{r['mae_eur']/1e6:>7.2f}M {r['mdape']:>5.0f}% "
            f"{r['spearman']:>6.3f} {r['calib_ratio']:>5.2f} {r['range_ratio']:>5.2f} "
            f"{r['mdape_micro']:>5.0f}% {r['mdape_low']:>5.0f}% "
            f"{r['mdape_mid']:>5.0f}% {r['mdape_high']:>5.0f}% {r['mdape_elite']:>5.0f}%"
        )
    return "\n".join(lines)


# ---------- Model builders ----------

def _log_target(reg):
    return TransformedTargetRegressor(
        regressor=reg, func=np.log1p, inverse_func=np.expm1, check_inverse=False,
    )


def build_xgb_baseline():
    from xgboost import XGBRegressor
    return _log_target(XGBRegressor(
        objective="reg:squarederror", n_estimators=350, learning_rate=0.05,
        max_depth=4, subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
        random_state=42, n_jobs=-1, tree_method="hist",
    ))


def build_lgbm():
    import lightgbm as lgb
    return _log_target(lgb.LGBMRegressor(
        n_estimators=500, learning_rate=0.04, max_depth=6, num_leaves=48,
        subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
        random_state=42, n_jobs=-1, verbose=-1,
    ))


class CatBoostLogWrapper:
    """Manual log1p wrapper — sklearn's TransformedTargetRegressor can't clone CatBoost."""
    def __init__(self, cat_indices):
        from catboost import CatBoostRegressor
        self.model = CatBoostRegressor(
            iterations=500, learning_rate=0.05, depth=6, l2_leaf_reg=3,
            cat_features=cat_indices, random_state=42, verbose=False,
        )
    def fit(self, X, y):
        self.model.fit(X, np.log1p(y))
        return self
    def predict(self, X):
        return np.maximum(np.expm1(self.model.predict(X)), 0)


def build_catboost(cat_indices):
    return CatBoostLogWrapper(cat_indices)


def build_xgb_tweedie():
    from xgboost import XGBRegressor
    # Tweedie loss handles skewed positive targets natively — no log transform
    return XGBRegressor(
        objective="reg:tweedie", tweedie_variance_power=1.5,
        n_estimators=400, learning_rate=0.05, max_depth=5, subsample=0.9,
        colsample_bytree=0.9, reg_lambda=1.0, random_state=42,
        n_jobs=-1, tree_method="hist",
    )


# ---------- Post-hoc calibration (anti regression-to-mean) ----------

def isotonic_calibrate(y_train_true, y_train_pred, y_test_pred):
    """Fit isotonic regression on (predicted, actual) from train; apply to test preds.

    This learns a monotonic correction that expands the prediction range
    toward the tails — directly attacks regression-to-the-mean.
    """
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(y_train_pred, y_train_true)
    return iso.predict(y_test_pred), iso


# ---------- Run all experiments ----------

def run() -> None:
    print("\n" + "=" * 100)
    print("LOADING DATA")
    print("=" * 100)
    from data import load_dataset_split as v1_split
    from data import CATEGORICAL_COLS as v1_cat
    from data_v2 import load_dataset_split as v2_split
    from data_v2 import CATEGORICAL_COLS as v2_cat

    X1_tr, X1_te, y1_tr, y1_te = v1_split()
    X2_tr, X2_te, y2_tr, y2_te = v2_split()

    print(f"  v1: {X1_tr.shape[1]} features  ({len(X1_tr):,} train, {len(X1_te):,} test)")
    print(f"  v2: {X2_tr.shape[1]} features  ({len(X2_tr):,} train, {len(X2_te):,} test)")

    # CatBoost needs categorical column INDICES (since v2 uses label-encoded ints currently
    # we'll reconstruct from the v2 module's CATEGORICAL_COLS)
    feat_cols_v2 = list(X2_tr.columns)
    cat_idx = [feat_cols_v2.index(c) for c in v2_cat]
    print(f"  CatBoost will treat columns at indices {cat_idx} as native categoricals")

    rows = []

    # ── 1. v1-XGB baseline ─────────────────────────────────────────────────
    print("\n[1/7] v1-XGB baseline...")
    m = build_xgb_baseline()
    m.fit(X1_tr, y1_tr)
    p = m.predict(X1_te)
    rows.append(evaluate("v1-XGB (current)", y1_te, p))

    # ── 2. v2-XGB ──────────────────────────────────────────────────────────
    print("[2/7] v2-XGB (enhanced features)...")
    m_v2 = build_xgb_baseline()
    m_v2.fit(X2_tr, y2_tr)
    p_v2 = m_v2.predict(X2_te)
    p_v2_train = m_v2.predict(X2_tr)
    rows.append(evaluate("v2-XGB", y2_te, p_v2))

    # ── 3. v2-LightGBM ─────────────────────────────────────────────────────
    print("[3/7] v2-LightGBM...")
    m_lgb = build_lgbm()
    m_lgb.fit(X2_tr, y2_tr)
    rows.append(evaluate("v2-LightGBM", y2_te, m_lgb.predict(X2_te)))

    # ── 4. v2-CatBoost (native categorical handling) ──────────────────────
    print("[4/7] v2-CatBoost (native cat features)...")
    m_cb = build_catboost(cat_idx)
    m_cb.fit(X2_tr, y2_tr)
    p_cb = m_cb.predict(X2_te)
    p_cb_train = m_cb.predict(X2_tr)
    rows.append(evaluate("v2-CatBoost", y2_te, p_cb))

    # ── 5. v2-XGB + isotonic calibration ──────────────────────────────────
    print("[5/7] v2-XGB + isotonic post-hoc calibration...")
    p_v2_calib, _ = isotonic_calibrate(y2_tr.values, p_v2_train, p_v2)
    rows.append(evaluate("v2-XGB+Iso", y2_te, p_v2_calib))

    # ── 6. v2-XGB-Tweedie (no log transform) ──────────────────────────────
    print("[6/7] v2-XGB-Tweedie (no log transform)...")
    m_tw = build_xgb_tweedie()
    m_tw.fit(X2_tr, y2_tr)
    rows.append(evaluate("v2-XGB-Tweedie", y2_te, m_tw.predict(X2_te)))

    # ── 7. Stack: average of v2-XGB+Iso and v2-CatBoost (both strong) ─────
    print("[7/7] v2-Stack (XGB+Iso ⊕ CatBoost average)...")
    p_cb_calib, _ = isotonic_calibrate(y2_tr.values, p_cb_train, p_cb)
    p_stack = (p_v2_calib + p_cb_calib) / 2
    rows.append(evaluate("v2-Stack", y2_te, p_stack))

    # ── Output ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 100)
    print("RESULTS  (lower is better for MAE, MdAPE; higher better for R², spearman; closer to 1 for calib, range)")
    print("=" * 100)
    print(fmt_results(rows))
    print()

    # Write CSV + JSON
    pd.DataFrame(rows).to_csv(RESULTS / "experiment_comparison.csv", index=False)
    (RESULTS / "experiment_comparison.json").write_text(json.dumps(rows, indent=2))
    print(f"📊 Written: results/experiment_comparison.csv")
    print(f"📊 Written: results/experiment_comparison.json")

    # ── Pick winner ─────────────────────────────────────────────────────────
    # Composite score: weight MdAPE down (the metric users feel), reward elite-bucket performance
    df = pd.DataFrame(rows)
    # Normalize: lower mdape better, higher r2_log better, calib closer to 1 better
    df["score"] = (
        -df["mdape"] / 100              # lower median % error
        + df["r2_log"] * 50             # log-space R²
        + df["spearman"] * 30           # rank correlation
        - df["mdape_elite"].fillna(100) / 100  # elite-bucket error
        - np.abs(1 - df["calib_ratio"]) * 20  # how close max pred is to max actual
    )
    df = df.sort_values("score", ascending=False)
    print("\n=== RANKED BY COMPOSITE SCORE ===")
    print(df[["model", "mdape", "r2_log", "spearman", "calib_ratio", "mdape_elite", "score"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\n🏆 Winner: {df.iloc[0]['model']}")


if __name__ == "__main__":
    run()
