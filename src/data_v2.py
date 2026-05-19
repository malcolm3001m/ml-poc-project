"""Enhanced feature pipeline — v2.

Goals:
- Keep the v1 contract: returns (X_train, X_test, y_train, y_test)
- Add high-signal features the v1 model is missing
- Avoid target leakage (don't touch player_valuations — that table IS the target)

New features vs v1 (14 → 31):

Career / activity shape:
- appearance_count: number of matches (different from minutes — distinguishes
  starter vs benchwarmer at same minute total)
- seasons_played: career length in seasons
- career_span_years: years between first and last appearance
- goals_per_90, assists_per_90, cards_per_90: standard football per-90 stats
- card_rate, goal_assist_ratio
- best_season_goals: peak single-season goal output (rarity signal for top talent)
- best_season_assists
- comp_diversity: number of unique competitions played in

Elite-tier signals:
- top_tier_minutes: minutes in UEFA international_cup competitions
- top_tier_apps: matches in UEFA-tier
- top_tier_share: fraction of career minutes in UEFA-tier (the key elite-player signal)

Club-quality proxies (joined from clubs.csv):
- club_squad_size, club_avg_age, club_foreigners_pct, club_n_internationals
  (clubs.total_market_value is all-NaN in our data; skipped)

Position interaction:
- intl_caps_per_age: international caps relative to age (top talents play for country young)
- has_agent: known agent flag

Engineered transforms:
- age_squared: non-linear age effect

We KEEP the v1 features so v2 is a strict superset.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

DATA_DIR = Path(__file__).parent.parent / "data"

# UEFA-tier (international_cup) — derived from competitions.csv inspection
TOP_TIER_COMPS = {"CL", "CLQ", "USC", "EL", "ELQ", "ECL", "ECLQ"}  # CL, Europa, Conference, qualifiers

CATEGORICAL_COLS = [
    # v1 features (kept)
    "position",
    "sub_position",
    "foot",
    "country_of_citizenship",
    "current_club_domestic_competition_id",
    # v2 additions
    "has_agent",
]

NUMERIC_COLS = [
    # v1 features (kept)
    "age",
    "height_in_cm",
    "international_caps",
    "international_goals",
    "goals",
    "assists",
    "minutes_played",
    "yellow_cards",
    "red_cards",
    # v2: activity shape
    "appearance_count",
    "seasons_played",
    "career_span_years",
    "goals_per_90",
    "assists_per_90",
    "cards_per_90",
    "goal_assist_ratio",
    "best_season_goals",
    "best_season_assists",
    "comp_diversity",
    # v2: elite-tier
    "top_tier_minutes",
    "top_tier_apps",
    "top_tier_share",
    # v2: club quality
    "club_squad_size",
    "club_avg_age",
    "club_foreigners_pct",
    "club_n_internationals",
    # v2: derived
    "intl_caps_per_age",
    "age_squared",
]

TARGET = "highest_market_value_in_eur"


def _load_players() -> pd.DataFrame:
    p = pd.read_csv(DATA_DIR / "players.csv", low_memory=False)
    p["date_of_birth"] = pd.to_datetime(p["date_of_birth"], errors="coerce")
    p["age"] = (pd.Timestamp.today() - p["date_of_birth"]).dt.days / 365.25
    p["age_squared"] = p["age"] ** 2
    p["has_agent"] = p["agent_name"].notna().astype(int)
    return p


def _load_appearances() -> pd.DataFrame:
    """Simple per-player aggregation — kept for the Streamlit/HTML raw joins.
    For model features see _enriched_appearances()."""
    appearances = pd.read_csv(DATA_DIR / "appearances.csv", low_memory=False)
    return appearances.groupby("player_id").agg(
        goals=("goals", "sum"),
        assists=("assists", "sum"),
        minutes_played=("minutes_played", "sum"),
        yellow_cards=("yellow_cards", "sum"),
        red_cards=("red_cards", "sum"),
    ).reset_index()


def _enriched_appearances() -> pd.DataFrame:
    """Aggregate appearances into per-player features including elite-tier signals."""
    apps = pd.read_csv(DATA_DIR / "appearances.csv", low_memory=False)
    apps["date"] = pd.to_datetime(apps["date"], errors="coerce")
    apps["season"] = apps["date"].dt.year  # rough — football seasons span 2 years, but year is OK as a count

    # Mark elite-tier rows
    apps["is_top_tier"] = apps["competition_id"].isin(TOP_TIER_COMPS).astype(int)

    # Career totals (mirrors v1 but adds appearance count)
    base = apps.groupby("player_id").agg(
        goals=("goals", "sum"),
        assists=("assists", "sum"),
        minutes_played=("minutes_played", "sum"),
        yellow_cards=("yellow_cards", "sum"),
        red_cards=("red_cards", "sum"),
        appearance_count=("appearance_id", "count"),
        comp_diversity=("competition_id", "nunique"),
        first_year=("season", "min"),
        last_year=("season", "max"),
        seasons_played=("season", "nunique"),
        top_tier_minutes=("minutes_played", lambda s: apps.loc[s.index].query("is_top_tier == 1")["minutes_played"].sum()),
        top_tier_apps=("is_top_tier", "sum"),
    ).reset_index()

    base["career_span_years"] = (base["last_year"] - base["first_year"]).clip(lower=0)
    base["top_tier_share"] = (base["top_tier_minutes"] / base["minutes_played"].replace(0, np.nan)).fillna(0)

    # Per-90 stats — the football standard
    nineties = base["minutes_played"] / 90.0
    base["goals_per_90"] = (base["goals"] / nineties.replace(0, np.nan)).fillna(0).clip(0, 5)
    base["assists_per_90"] = (base["assists"] / nineties.replace(0, np.nan)).fillna(0).clip(0, 5)
    base["cards_per_90"] = ((base["yellow_cards"] + base["red_cards"]) / nineties.replace(0, np.nan)).fillna(0).clip(0, 3)
    base["goal_assist_ratio"] = (base["goals"] / (base["goals"] + base["assists"] + 1)).fillna(0)

    # Best-season output (rarity signal)
    season_goals = apps.groupby(["player_id", "season"])["goals"].sum().reset_index()
    best_g = season_goals.groupby("player_id")["goals"].max().rename("best_season_goals")
    season_assists = apps.groupby(["player_id", "season"])["assists"].sum().reset_index()
    best_a = season_assists.groupby("player_id")["assists"].max().rename("best_season_assists")
    base = base.merge(best_g, on="player_id", how="left").merge(best_a, on="player_id", how="left")
    base["best_season_goals"] = base["best_season_goals"].fillna(0)
    base["best_season_assists"] = base["best_season_assists"].fillna(0)

    base = base.drop(columns=["first_year", "last_year"])
    return base


def _club_features() -> pd.DataFrame:
    """Club-level proxies for player quality — joined via current_club_id."""
    c = pd.read_csv(DATA_DIR / "clubs.csv", low_memory=False)
    out = c[[
        "club_id", "squad_size", "average_age",
        "foreigners_percentage", "national_team_players",
    ]].rename(columns={
        "club_id": "current_club_id",
        "squad_size": "club_squad_size",
        "average_age": "club_avg_age",
        "foreigners_percentage": "club_foreigners_pct",
        "national_team_players": "club_n_internationals",
    })
    return out


def _build_features(players: pd.DataFrame, apps_agg: pd.DataFrame, clubs: pd.DataFrame) -> pd.DataFrame:
    df = players.merge(apps_agg, on="player_id", how="left")
    df = df.merge(clubs, on="current_club_id", how="left")

    # Derived feature — compute before column filter
    df["intl_caps_per_age"] = (
        df["international_caps"].fillna(0) / df["age"].replace(0, np.nan)
    ).fillna(0)

    keep = CATEGORICAL_COLS + NUMERIC_COLS + [TARGET]
    df = df[keep].copy()
    df = df.dropna(subset=[TARGET])

    # Fill missing numerics
    int_zero = ["goals", "assists", "minutes_played", "yellow_cards", "red_cards",
                "appearance_count", "seasons_played", "career_span_years",
                "comp_diversity", "top_tier_minutes", "top_tier_apps",
                "international_caps", "international_goals",
                "best_season_goals", "best_season_assists"]
    df[int_zero] = df[int_zero].fillna(0)

    median_fill = ["height_in_cm", "age", "age_squared", "club_squad_size",
                   "club_avg_age", "club_foreigners_pct", "club_n_internationals",
                   "goals_per_90", "assists_per_90", "cards_per_90",
                   "goal_assist_ratio", "top_tier_share", "intl_caps_per_age"]
    for col in median_fill:
        df[col] = df[col].fillna(df[col].median())

    # Encode categoricals
    for col in CATEGORICAL_COLS:
        df[col] = df[col].fillna("Unknown")
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    return df


def load_dataset_split() -> tuple[Any, Any, Any, Any]:
    """v2 — enhanced features, same contract."""
    players = _load_players()
    apps_agg = _enriched_appearances()
    clubs = _club_features()
    df = _build_features(players, apps_agg, clubs)

    feature_cols = CATEGORICAL_COLS + NUMERIC_COLS
    X = df[feature_cols]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_dataset_split()
    print(f"v2 features: {X_train.shape[1]} (v1 was 14)")
    print(f"Train: {len(X_train):,}  Test: {len(X_test):,}")
    print(f"Categoricals ({len(CATEGORICAL_COLS)}):", CATEGORICAL_COLS)
    print(f"Numerics ({len(NUMERIC_COLS)}):", NUMERIC_COLS)
