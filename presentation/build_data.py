"""Bake all the data the HTML presentation needs into a single data.js file.

Output: presentation/data.js (loaded by index.html as a regular <script> tag)

What gets baked:
- Validation numbers (CV, baselines, overfit, ablation, per-position MAE)
- Histogram bins for the target distribution
- Country-aggregated median peak value (for the choropleth)
- Position box-plot statistics
- ~60 hand-picked famous + 40 random TEST players, each with all 3 model predictions
- Top 5 countries by median peak value
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from config import MODELS, RESULTS_DIR  # noqa: E402
from data import (  # noqa: E402
    TARGET,
    _load_appearances,
    _load_players,
    load_dataset_split,
)

OUT = Path(__file__).resolve().parent / "data.js"


def _eur_short(x: float) -> str:
    if x >= 1_000_000:
        return f"€{x / 1_000_000:.2f}M"
    if x >= 1_000:
        return f"€{x / 1_000:.0f}k"
    return f"€{x:.0f}"


FAMOUS = [
    # Substring patterns — matched case-insensitive against players.name
    "Mbappé", "Mbappe", "Haaland", "Bellingham", "Vinic", "Vinicius",
    "Messi", "Ronaldo", "Neymar", "Salah", "Kane", "Müller", "Muller",
    "Modric", "De Bruyne", "Lewandowski", "Benzema", "Pedri", "Gavi",
    "Pulisic", "Reece James", "Saka", "Foden", "Yamal", "Cole Palmer",
    "Wirtz", "Musiala", "Camavinga", "Tchouameni", "Tchouaméni",
    "Casemiro", "Sergio Ramos", "Van Dijk", "Alisson", "Courtois",
    "Klose", "Iniesta", "Xavi", "Ronaldinho", "Zidane",
]


def pick_famous(raw: pd.DataFrame, X_all: pd.DataFrame, n_per_pattern: int = 1) -> list[int]:
    picks: list[int] = []
    for pattern in FAMOUS:
        mask = raw["name"].str.contains(pattern, case=False, na=False)
        candidates = raw[mask & raw.index.isin(X_all.index)]
        if not candidates.empty:
            candidate = candidates.sort_values(TARGET, ascending=False).head(n_per_pattern)
            for idx in candidate.index:
                if idx not in picks:
                    picks.append(int(idx))
    return picks


def main() -> None:
    print("Loading splits + models...")
    X_train, X_test, y_train, y_test = load_dataset_split()
    train_idx = set(X_train.index)
    X_all = pd.concat([X_train, X_test])
    y_all = pd.concat([y_train, y_test])

    print("Loading raw player metadata...")
    players = _load_players()
    appearances = _load_appearances()
    raw = players.merge(appearances, on="player_id", how="left")
    raw = raw.dropna(subset=[TARGET])
    for col in ["goals", "assists", "minutes_played", "yellow_cards", "red_cards"]:
        raw[col] = raw[col].fillna(0)

    # Most-recent valuation per player — proxy for "current market price"
    # Keyed by player_id so we can look up without breaking raw's index.
    print("Loading latest valuations (proxy for current value)...")
    val = pd.read_csv(ROOT / "data" / "player_valuations.csv", low_memory=False)
    val["date"] = pd.to_datetime(val["date"])
    latest_by_pid = (
        val.sort_values("date")
        .drop_duplicates("player_id", keep="last")
        .set_index("player_id")["market_value_in_eur"]
    )

    # Enriched appearances aggregates for highlight selection — keyed by player_id
    from data_v2 import _enriched_appearances
    enriched_by_pid = _enriched_appearances().set_index("player_id")

    # Pre-compute percentile thresholds for highlight selection
    STAT_FIELDS = {
        "goals": ("⚽ Goals", "{v:,.0f}"),
        "assists": ("🎯 Assists", "{v:,.0f}"),
        "minutes_played": ("⏱ Minutes", "{v:,.0f}"),
        "appearance_count": ("📋 Apps", "{v:,.0f}"),
        "international_caps": ("🌍 Int. caps", "{v:,.0f}"),
        "international_goals": ("🇺🇸 Int. goals", "{v:,.0f}"),
        "top_tier_minutes": ("🏆 UEFA mins", "{v:,.0f}"),
        "top_tier_apps": ("🏆 UEFA apps", "{v:,.0f}"),
        "career_span_years": ("📆 Career yrs", "{v:.0f}"),
        "best_season_goals": ("🔥 Best season goals", "{v:,.0f}"),
    }
    # Compute percentile thresholds from the enriched-stats table (population-wide)
    quantiles = {}
    for col in STAT_FIELDS:
        source = enriched_by_pid if col in enriched_by_pid.columns else raw
        if col in source.columns:
            quantiles[col] = {
                "p99": float(source[col].quantile(0.99)),
                "p95": float(source[col].quantile(0.95)),
                "p90": float(source[col].quantile(0.90)),
                "p75": float(source[col].quantile(0.75)),
            }

    # ---- Predictions for selected players ----
    print("Loading 3 trained models...")
    loaded = {key: joblib.load(spec["path"]) for key, spec in MODELS.items()}

    print("Picking famous players...")
    famous_idx = pick_famous(raw, X_all)

    print("Sampling random TEST players...")
    test_pool = list(X_test.index)
    rng = np.random.default_rng(42)
    random_test = rng.choice(test_pool, size=min(40, len(test_pool)), replace=False).tolist()

    # Also bake the top-N by current value — guarantees every well-known player
    # (Antony, Pulisic, Rashford, Sancho etc.) is searchable in the demo
    print("Baking top 300 players by current valuation...")
    top_by_value = (
        latest_by_pid.dropna()
        .sort_values(ascending=False)
        .head(500)
        .index.tolist()
    )
    pid_to_row = players.set_index("player_id")
    top_idx = []
    for pid in top_by_value:
        if pid in pid_to_row.index:
            # pid_to_row.loc[pid] could be a Series or DataFrame if duplicates
            row_idx = pid_to_row.index.get_indexer([pid])[0]
            # Get the original index position in the players DataFrame
            orig_idx_candidates = players.index[players["player_id"] == pid].tolist()
            for oi in orig_idx_candidates:
                if oi in X_all.index:
                    top_idx.append(int(oi))
                    break
        if len(top_idx) >= 300:
            break
    print(f"  matched {len(top_idx)} top-value players to the X_all index")

    # ── Compute top ROI opportunities on the FULL test set ───────────────
    # Filters: at-peak (current >= 85% of highest_recorded), age <= 28,
    # current >= €5M (real scout targets), model prediction > current.
    # Sort by absolute upside (€).
    print("Scanning full test set for top ROI opportunities...")
    cb_model = joblib.load(MODELS["catboost_regressor"]["path"])
    test_preds = np.maximum(cb_model.predict(X_test), 0.0)
    test_ids = players.loc[X_test.index, "player_id"].values
    test_ages = players.loc[X_test.index, "age"].fillna(0).astype(int).values

    opportunities = []
    for i, (pid, age, pred) in enumerate(zip(test_ids, test_ages, test_preds)):
        cur = latest_by_pid.get(int(pid))
        peak = float(y_test.iloc[i])
        if cur is None or pd.isna(cur) or cur < 5_000_000: continue
        if age < 18 or age > 28: continue
        if cur < peak * 0.85: continue
        if pred <= cur: continue
        upside_eur = float(pred) - float(cur)
        if upside_eur < 1_000_000: continue
        opportunities.append((X_test.index[i], int(pid), age, float(cur), float(pred), upside_eur))

    opportunities.sort(key=lambda x: -x[5])
    top_opps = opportunities[:10]
    print(f"  found {len(opportunities)} opportunities · keeping top {len(top_opps)}")
    opp_idx = [o[0] for o in top_opps]

    selected = list(dict.fromkeys(famous_idx + [int(i) for i in random_test] + opp_idx + top_idx))
    print(f"  picked {len(selected)} players ({len(famous_idx)} famous + {len(random_test)} random + {len(top_opps)} top ROI + {len(top_idx)} top-value)")

    X_selected = X_all.loc[selected]
    preds = {}
    for key, model in loaded.items():
        preds[key] = np.maximum(model.predict(X_selected), 0.0)

    def _build_highlights(player_id):
        """Pick the 3-5 stats where this player ranks highest, with percentile labels."""
        if player_id not in enriched_by_pid.index:
            return []
        stats_row = enriched_by_pid.loc[player_id]
        scored = []
        for col, (label, fmt) in STAT_FIELDS.items():
            if col not in stats_row.index or pd.isna(stats_row[col]):
                continue
            v = float(stats_row[col])
            if v <= 0:
                continue
            q = quantiles.get(col, {})
            if v >= q.get("p99", float("inf")):
                tier, badge = "elite", "TOP 1%"
            elif v >= q.get("p95", float("inf")):
                tier, badge = "great", "TOP 5%"
            elif v >= q.get("p90", float("inf")):
                tier, badge = "good", "TOP 10%"
            elif v >= q.get("p75", float("inf")):
                tier, badge = "decent", "TOP 25%"
            else:
                tier, badge = None, None
            # Rank by percentile achieved (use p99 first, then p95, etc.)
            rank_score = 4 if tier == "elite" else 3 if tier == "great" else 2 if tier == "good" else 1 if tier == "decent" else 0
            scored.append({
                "field": col,
                "label": label,
                "value": v,
                "value_fmt": fmt.format(v=v),
                "tier": tier,
                "badge": badge,
                "rank_score": rank_score,
            })
        # Sort by rank then by raw value within bucket
        scored.sort(key=lambda s: (s["rank_score"], s["value"]), reverse=True)
        # Take top 5; if fewer than 3 have any tier, fill with the top values regardless
        with_tier = [s for s in scored if s["tier"] is not None]
        if len(with_tier) >= 3:
            return [{k: v for k, v in s.items() if k != "rank_score"} for s in with_tier[:5]]
        # Fall back: top 3 by raw value (no badge)
        return [{k: v for k, v in s.items() if k != "rank_score"} for s in scored[:3]]

    players_out = []
    for n, idx in enumerate(selected):
        row = raw.loc[idx]
        actual = float(y_all.loc[idx])
        split = "TRAIN" if idx in train_idx else "TEST"
        pid = int(row["player_id"])
        latest = latest_by_pid.get(pid)
        players_out.append({
            "id": int(idx),
            "player_id": pid,
            "name": str(row.get("name", "Unknown")) if pd.notna(row.get("name")) else "Unknown",
            "image_url": str(row.get("image_url")) if pd.notna(row.get("image_url")) else None,
            "position": str(row.get("position", "?")) if pd.notna(row.get("position")) else "?",
            "foot": str(row.get("foot", "?")) if pd.notna(row.get("foot")) else "?",
            "country": str(row.get("country_of_citizenship", "?")) if pd.notna(row.get("country_of_citizenship")) else "?",
            "age": int(row["age"]) if pd.notna(row.get("age")) else None,
            "highest_recorded": actual,           # what the dataset calls "highest_market_value_in_eur"
            "current_value": float(latest) if pd.notna(latest) else None,
            "actual": actual,                     # kept for backward-compat with renderer
            "split": split,
            "predictions": {key: float(preds[key][n]) for key in MODELS.keys()},
            "highlights": _build_highlights(pid),
        })

    # ---- Histogram bins for target distribution ----
    print("Building histogram + box + country aggregates...")
    log_target = np.log10(np.maximum(raw[TARGET].values, 1))
    counts, edges = np.histogram(log_target, bins=60)
    hist = [
        {"bin_lo": float(10 ** edges[i]), "bin_hi": float(10 ** edges[i + 1]),
         "count": int(counts[i])}
        for i in range(len(counts))
    ]

    # ---- Country aggregates ----
    home_nations = {"England": "United Kingdom", "Scotland": "United Kingdom",
                    "Wales": "United Kingdom", "Northern Ireland": "United Kingdom"}
    df_c = raw.dropna(subset=["country_of_citizenship"]).copy()
    df_c["country"] = df_c["country_of_citizenship"].replace(home_nations)
    agg = (
        df_c.groupby("country")
        .agg(n=("player_id", "count"), median_peak=(TARGET, "median"))
        .reset_index()
    )
    countries = agg[agg["n"] >= 50].sort_values("median_peak", ascending=False)
    countries_out = [
        {"country": r["country"], "n": int(r["n"]), "median_peak": float(r["median_peak"])}
        for _, r in countries.iterrows()
    ]
    top5 = countries_out[:5]

    # ---- Per-position summary for the box chart ----
    pos_summary = []
    for pos, grp in raw.dropna(subset=["position"]).groupby("position"):
        vals = grp[TARGET].values
        pos_summary.append({
            "position": pos,
            "n": int(len(vals)),
            "median": float(np.median(vals)),
            "q1": float(np.percentile(vals, 25)),
            "q3": float(np.percentile(vals, 75)),
            "min": float(np.percentile(vals, 5)),
            "max": float(np.percentile(vals, 95)),
            "mean": float(np.mean(vals)),
        })
    pos_summary.sort(key=lambda x: x["median"], reverse=True)

    # ---- Read validation CSVs ----
    print("Reading validation CSVs...")
    def _read(name):
        p = RESULTS_DIR / name
        return pd.read_csv(p) if p.exists() else pd.DataFrame()

    metrics = _read("model_metrics.csv")
    validation = _read("validation_metrics.csv")
    ablation = _read("ablation.csv")
    per_position = _read("per_position_mae.csv")

    metrics_out = []
    if not metrics.empty:
        for _, r in metrics.iterrows():
            metrics_out.append({
                "key": r["model_key"], "name": r["model_name"],
                "mae": float(r["mae_eur"]), "rmse": float(r["rmse_eur"]),
                "r2": float(r["r2"]),
            })

    overfit_out, cv_out, baselines_out = [], [], []
    if not validation.empty:
        for _, r in validation[validation["kind"] == "overfit"].iterrows():
            overfit_out.append({
                "model": r["name"],
                "train_r2": float(r["train_r2"]), "test_r2": float(r["test_r2"]),
                "gap": float(r["overfit_gap"]),
            })
        for _, r in validation[validation["kind"] == "cv"].iterrows():
            cv_out.append({
                "model": r["name"],
                "mean": float(r["cv_r2_mean"]), "std": float(r["cv_r2_std"]),
                "min": float(r["cv_r2_min"]), "max": float(r["cv_r2_max"]),
            })
        for _, r in validation[validation["kind"] == "baseline"].iterrows():
            baselines_out.append({
                "name": r["name"], "r2": float(r["r2"]), "mae": float(r["mae_eur"]),
            })

    ablation_out = []
    if not ablation.empty:
        for _, r in ablation.iterrows():
            ablation_out.append({
                "variant": r["variant"],
                "r2": float(r["r2"]), "mae": float(r["mae_eur"]), "rmse": float(r["rmse_eur"]),
            })

    per_pos_out = []
    if not per_position.empty:
        for _, r in per_position.iterrows():
            per_pos_out.append({
                "position": r["position"],
                "n": int(r["n"]),
                "median_actual": float(r["median_actual_eur"]),
                "mae": float(r["mae_eur"]),
                "mae_pct": float(r["mae_eur"]) / float(r["median_actual_eur"]) * 100,
            })

    # ---- Top features by XGBoost importance ----
    print("Extracting feature importances...")
    xgb = joblib.load(MODELS["xgboost_regressor"]["path"])
    inner = getattr(xgb, "regressor_", xgb)
    feats = []
    if hasattr(inner, "feature_importances_"):
        for name, importance in zip(X_train.columns, inner.feature_importances_):
            feats.append({"feature": name, "importance": float(importance)})
        feats.sort(key=lambda x: x["importance"], reverse=True)

    # ---- Residuals (sample for the scatter) ----
    print("Sampling residuals...")
    residuals_path = RESULTS_DIR / "residuals.csv"
    residuals_out = []
    if residuals_path.exists():
        res = pd.read_csv(residuals_path).sample(min(2000, 7000), random_state=42)
        for _, r in res.iterrows():
            residuals_out.append({
                "name": str(r["name"]) if pd.notna(r["name"]) else "?",
                "position": str(r["position"]) if pd.notna(r["position"]) else "?",
                "actual": float(r["actual_eur"]),
                "predicted": float(r["predicted_eur"]),
            })

    # ---- Bucket diagnostic — % error per price tier, the user-felt metric ----
    print("Computing prediction bucket diagnostic on full test set...")
    cb_model = joblib.load(MODELS["catboost_regressor"]["path"])
    test_preds = np.maximum(cb_model.predict(X_test), 0)
    y_arr = y_test.values
    safe = np.maximum(y_arr, 1)
    pct_err = np.abs(test_preds - y_arr) / safe * 100
    bucket_diagnostic = []
    for lo, hi, label in [
        (0, 200_000, "Micro (<€200k)"),
        (200_000, 1_000_000, "Low (€200k–1M)"),
        (1_000_000, 5_000_000, "Mid (€1–5M)"),
        (5_000_000, 20_000_000, "High (€5–20M)"),
        (20_000_000, 1e10, "Elite (>€20M)"),
    ]:
        mask = (y_arr >= lo) & (y_arr < hi)
        if mask.sum() > 5:
            bucket_diagnostic.append({
                "bucket": label,
                "n": int(mask.sum()),
                "median_pct_error": float(np.median(pct_err[mask])),
                "over_predicted_rate": float((test_preds[mask] > y_arr[mask]).mean() * 100),
                "median_predicted": float(np.median(test_preds[mask])),
                "median_actual": float(np.median(y_arr[mask])),
            })

    # ---- Stats ----
    summary = {
        "n_players": int(len(raw)),
        "n_features": 14,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_countries": int(raw["country_of_citizenship"].nunique()),
        "median_value": float(raw[TARGET].median()),
        "mean_value": float(raw[TARGET].mean()),
        "max_value": float(raw[TARGET].max()),
    }

    # Top opportunities list — references player ids from players_out
    # (each opportunity is a row from the earlier scan, in upside-€ order)
    top_opportunities = [
        {"id": int(o[0]), "upside_eur": float(o[5]),
         "upside_pct": float((o[4] - o[3]) / o[3] * 100)}
        for o in top_opps
    ]

    data = {
        "summary": summary,
        "metrics": metrics_out,
        "validation": {
            "overfit": overfit_out,
            "cv": cv_out,
            "baselines": baselines_out,
        },
        "ablation": ablation_out,
        "per_position": per_pos_out,
        "histogram": hist,
        "countries": countries_out,
        "top5_countries": top5,
        "position_box": pos_summary,
        "feature_importance": feats,
        "residuals_sample": residuals_out,
        "bucket_diagnostic": bucket_diagnostic,
        "top_opportunities": top_opportunities,
        "players": players_out,
    }

    print(f"Writing {OUT} ...")
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    OUT.write_text(f"window.POC_DATA = {payload};\n", encoding="utf-8")
    print(f"✅ {OUT.name} — {OUT.stat().st_size // 1024} KB")
    print(f"  players: {len(players_out)}")
    print(f"  countries: {len(countries_out)}")
    print(f"  residual samples: {len(residuals_out)}")


if __name__ == "__main__":
    main()
