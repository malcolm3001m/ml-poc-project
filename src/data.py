from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

DATA_DIR = Path(__file__).parent.parent / "data"

CATEGORICAL_COLS = [
    "position",
    "sub_position",
    "foot",
    "country_of_citizenship",
    "current_club_domestic_competition_id",
]

NUMERIC_COLS = [
    "age",
    "height_in_cm",
    "international_caps",
    "international_goals",
    "highest_market_value_in_eur",
    "goals",
    "assists",
    "minutes_played",
    "yellow_cards",
    "red_cards",
]

TARGET = "market_value_in_eur"


def _load_players() -> pd.DataFrame:
    players = pd.read_csv(DATA_DIR / "players.csv", low_memory=False)
    players["date_of_birth"] = pd.to_datetime(players["date_of_birth"], errors="coerce")
    players["age"] = (pd.Timestamp.today() - players["date_of_birth"]).dt.days / 365.25
    return players


def _load_appearances() -> pd.DataFrame:
    appearances = pd.read_csv(DATA_DIR / "appearances.csv", low_memory=False)
    agg = appearances.groupby("player_id").agg(
        goals=("goals", "sum"),
        assists=("assists", "sum"),
        minutes_played=("minutes_played", "sum"),
        yellow_cards=("yellow_cards", "sum"),
        red_cards=("red_cards", "sum"),
    ).reset_index()
    return agg


def _build_features(players: pd.DataFrame, appearances: pd.DataFrame) -> pd.DataFrame:
    df = players.merge(appearances, on="player_id", how="left")

    keep = CATEGORICAL_COLS + NUMERIC_COLS + [TARGET]
    df = df[keep].copy()

    df[["goals", "assists", "minutes_played", "yellow_cards", "red_cards"]] = (
        df[["goals", "assists", "minutes_played", "yellow_cards", "red_cards"]].fillna(0)
    )
    df[["international_caps", "international_goals"]] = (
        df[["international_caps", "international_goals"]].fillna(0)
    )
    df["highest_market_value_in_eur"] = df["highest_market_value_in_eur"].fillna(
        df["highest_market_value_in_eur"].median()
    )
    df["height_in_cm"] = df["height_in_cm"].fillna(df["height_in_cm"].median())
    df["age"] = df["age"].fillna(df["age"].median())

    for col in CATEGORICAL_COLS:
        df[col] = df[col].fillna("Unknown")
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    df = df.dropna(subset=[TARGET])
    return df


def load_dataset_split() -> tuple[Any, Any, Any, Any]:
    players = _load_players()
    appearances = _load_appearances()
    df = _build_features(players, appearances)

    feature_cols = CATEGORICAL_COLS + NUMERIC_COLS
    X = df[feature_cols]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test
