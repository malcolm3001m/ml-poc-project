"""Streamlit entry point — Transfermarkt peak market value PoC."""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

from config import MODEL_METRICS_FILE, MODELS
from data import (
    CATEGORICAL_COLS,
    NUMERIC_COLS,
    TARGET,
    _load_appearances,
    _load_players,
    load_dataset_split,
)


# ---------- Cached loaders ----------

@st.cache_data(show_spinner=False)
def _raw_players_and_appearances() -> pd.DataFrame:
    players = _load_players()
    appearances = _load_appearances()
    df = players.merge(appearances, on="player_id", how="left")
    df = df.dropna(subset=[TARGET])
    for col in ["goals", "assists", "minutes_played", "yellow_cards", "red_cards"]:
        df[col] = df[col].fillna(0)
    return df


@st.cache_data(show_spinner=False)
def _splits():
    X_train, X_test, y_train, y_test = load_dataset_split()
    return X_train, X_test, y_train, y_test


@st.cache_data(show_spinner=False)
def _metrics() -> pd.DataFrame:
    if not MODEL_METRICS_FILE.exists():
        return pd.DataFrame()
    return pd.read_csv(MODEL_METRICS_FILE)


@st.cache_resource(show_spinner=False)
def _load_model(path: Path):
    return joblib.load(path)


def _eur(x: float) -> str:
    if x >= 1_000_000:
        return f"€{x / 1_000_000:.2f}M"
    if x >= 1_000:
        return f"€{x / 1_000:.0f}k"
    return f"€{x:.0f}"


# ---------- Sections ----------

def section_overview() -> None:
    st.title("⚽ Predicting Football Players' Peak Market Value")
    st.caption("DAT0424 — ML Proof of Concept — Malcolm Morgan")

    st.markdown(
        """
        ### The business question
        **Can we predict a football player's career-peak market value from biographical
        and cumulative performance data?**

        Why anyone cares:

        - **Transfer scouting** — flag undervalued players before the market reprices them.
        - **Contract valuation** — anchor wage negotiations to a defensible ceiling.
        - **Club asset accounting** — value a squad as a portfolio of human capital.
        - **Fantasy / betting markets** — input feature for downstream models.

        ### The approach
        Three tree-based regressors trained on **Transfermarkt** data (12 CSVs covering
        players, appearances, clubs, games, transfers). Target: `highest_market_value_in_eur`.
        Best model: **XGBoost — R² = 0.73, MAE ≈ €1.79M.**

        ### What this app shows
        1. Dataset overview & EDA
        2. Model comparison
        3. Feature importance
        4. **Live prediction demo** — pick any player, see what the model predicts vs. their real peak
        5. Honest limitations
        """
    )


def section_dataset() -> None:
    st.header("📊 Dataset & exploration")
    raw = _raw_players_and_appearances()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Players", f"{len(raw):,}")
    c2.metric("Median peak value", _eur(raw[TARGET].median()))
    c3.metric("Mean peak value", _eur(raw[TARGET].mean()))
    c4.metric("Max peak value", _eur(raw[TARGET].max()))

    st.subheader("Peak market value — distribution")
    st.caption("Long right tail. The bulk of players sit under €5M; a small elite reach >€100M.")
    fig = px.histogram(
        raw, x=TARGET, nbins=80, log_y=True,
        labels={TARGET: "Peak market value (€)"},
    )
    st.plotly_chart(fig, width='stretch')

    st.subheader("Position vs. peak value")
    fig = px.box(
        raw.dropna(subset=["position"]),
        x="position", y=TARGET, log_y=True,
        labels={TARGET: "Peak market value (€)"},
    )
    st.plotly_chart(fig, width='stretch')

    st.subheader("Age vs. peak value")
    sample = raw.dropna(subset=["age", TARGET]).sample(min(5000, len(raw)), random_state=42)
    fig = px.scatter(
        sample, x="age", y=TARGET, opacity=0.35, log_y=True,
        labels={"age": "Age (years)", TARGET: "Peak market value (€)"},
    )
    st.plotly_chart(fig, width='stretch')

    with st.expander("Features used by the model"):
        st.markdown(
            f"**Categorical ({len(CATEGORICAL_COLS)}):** "
            + ", ".join(f"`{c}`" for c in CATEGORICAL_COLS)
        )
        st.markdown(
            f"**Numeric ({len(NUMERIC_COLS)}):** "
            + ", ".join(f"`{c}`" for c in NUMERIC_COLS)
        )


def section_models() -> None:
    st.header("🤖 Model comparison")
    metrics = _metrics()
    if metrics.empty:
        st.warning("Run `python scripts/main.py` to generate `results/model_metrics.csv`.")
        return

    display = metrics[["model_name", "mae_eur", "rmse_eur", "r2"]].copy()
    display["mae_eur"] = display["mae_eur"].apply(_eur)
    display["rmse_eur"] = display["rmse_eur"].apply(_eur)
    display["r2"] = display["r2"].apply(lambda v: f"{v:.3f}")
    display.columns = ["Model", "MAE", "RMSE", "R²"]
    st.dataframe(display, width='stretch', hide_index=True)

    st.markdown(
        """
        **Read:** XGBoost wins on every metric. Random Forest is close. Decision Tree
        underfits the long-tail variance.

        The gap between models is smaller than the gap between any model and "perfect."
        R² ≈ 0.73 means we explain ~73% of the variance in peak value — the rest is signal
        we don't have access to (deals, agents, hype, injuries, league context, time).
        """
    )

    fig = px.bar(
        metrics, x="model_name", y="r2", text="r2",
        labels={"model_name": "Model", "r2": "R²"},
        range_y=[0, 1],
    )
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    st.plotly_chart(fig, width='stretch')


def section_feature_importance() -> None:
    st.header("🔍 What drives the prediction?")
    X_train, _, _, _ = _splits()

    model_key = st.selectbox(
        "Model",
        list(MODELS.keys()),
        index=list(MODELS.keys()).index("xgboost_regressor"),
        format_func=lambda k: MODELS[k]["name"],
    )
    model = _load_model(MODELS[model_key]["path"])

    if not hasattr(model, "feature_importances_"):
        st.info("Selected model exposes no feature importances.")
        return

    importances = pd.DataFrame(
        {"feature": X_train.columns, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=True)

    fig = px.bar(
        importances, x="importance", y="feature", orientation="h",
        labels={"importance": "Importance", "feature": "Feature"},
    )
    st.plotly_chart(fig, width='stretch')

    top = importances.tail(3)["feature"].tolist()[::-1]
    st.markdown(
        f"**Top 3 drivers:** `{top[0]}`, `{top[1]}`, `{top[2]}`. "
        "Cumulative minutes + position together account for most of the signal — "
        "playing a lot at a valuable position is the strongest predictor of a high peak."
    )


def section_predict() -> None:
    st.header("🎯 Live prediction demo")
    st.markdown(
        "Pick any player from the dataset. The app pulls their feature row, "
        "asks each trained model for a prediction, and shows it next to the real peak value."
    )

    raw = _raw_players_and_appearances()
    X_train, X_test, y_train, y_test = _splits()
    X_all = pd.concat([X_train, X_test])
    y_all = pd.concat([y_train, y_test])

    # Map encoded rows back to player names via the raw dataframe index
    raw_aligned = raw.loc[X_all.index].copy()
    raw_aligned["__display"] = (
        raw_aligned["name"].fillna("Unknown")
        + " (" + raw_aligned["position"].fillna("?") + ", "
        + raw_aligned[TARGET].apply(_eur) + ")"
    )

    default_idx = 0
    famous = raw_aligned[raw_aligned["name"].str.contains("Mbapp|Haaland|Messi|Ronaldo|Vinic|Bellingham", case=False, na=False)]
    if not famous.empty:
        default_idx = list(raw_aligned.index).index(famous.index[0])

    choice = st.selectbox(
        "Player",
        options=raw_aligned.index,
        format_func=lambda i: raw_aligned.loc[i, "__display"],
        index=default_idx,
    )

    row = X_all.loc[[choice]]
    actual = float(y_all.loc[choice])

    st.subheader(raw_aligned.loc[choice, "name"])
    meta = raw_aligned.loc[choice]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Position", str(meta.get("position", "?")))
    c2.metric("Foot", str(meta.get("foot", "?")))
    c3.metric("Age", f"{meta.get('age', 0):.0f}" if pd.notna(meta.get("age")) else "?")
    c4.metric("Real peak", _eur(actual))

    preds = []
    for key, spec in MODELS.items():
        model = _load_model(spec["path"])
        pred = float(model.predict(row)[0])
        pred = max(pred, 0.0)
        preds.append({"Model": spec["name"], "Predicted": pred, "Error": pred - actual})

    pred_df = pd.DataFrame(preds)
    show = pred_df.copy()
    show["Predicted"] = show["Predicted"].apply(_eur)
    show["Error"] = show["Error"].apply(lambda v: ("+" if v >= 0 else "") + _eur(v))
    st.dataframe(show, width='stretch', hide_index=True)

    fig = px.bar(
        pred_df, x="Model", y="Predicted", text="Predicted",
        labels={"Predicted": "Predicted peak value (€)"},
    )
    fig.update_traces(texttemplate="€%{text:,.0f}", textposition="outside")
    fig.add_hline(
        y=actual, line_dash="dash", line_color="red",
        annotation_text=f"Real peak: {_eur(actual)}", annotation_position="top right",
    )
    st.plotly_chart(fig, width='stretch')


def section_limitations() -> None:
    st.header("⚠️ Limitations & honest caveats")
    st.markdown(
        """
        ### 1. MAE looks great in absolute terms — but the target is skewed.
        - €1.79M MAE next to Mbappé (€180M) is rounding error.
        - €1.79M MAE next to the **median** player (~€500k) is **3.5× the actual value**.
        - The model is most useful at the top of the distribution, where most of the
          financial action is anyway — but don't claim it predicts a youth-academy
          player's peak value.

        ### 2. Possible target leakage.
        - `current_club_domestic_competition_id` correlates strongly with value (a player
          at PSG is worth more *because* they're at PSG). Career-peak value may have
          *caused* the club, not the other way around. A cleaner setup would freeze
          features at a fixed age (e.g. 21) and predict forward.

        ### 3. Train/test split is random, not temporal.
        - Real deployment would train on players who peaked before year Y and predict
          for players still active in year Y. A random split lets the model peek at
          the era it's predicting.

        ### 4. We aggregate appearances over an entire career.
        - For active players, "cumulative goals" is a moving target. The features for a
          21-year-old in 2026 are not the features they'll have in 2030.

        ### 5. Missing context the model can't see.
        - Agent fees, marketability, social-media reach, injury history, geopolitical
          factors (Saudi/MLS spending booms), club-specific premiums.

        ### What we'd do next
        - Re-frame as "predict peak value at age 25 given features up to age 21."
        - Add time-series features (form, trajectory) instead of career aggregates.
        - Try a quantile regression to give a price *range* instead of a point estimate.
        - SHAP analysis to make predictions explainable per-player.
        """
    )


# ---------- Entry point ----------

SECTIONS = {
    "1. Overview": section_overview,
    "2. Dataset & EDA": section_dataset,
    "3. Model comparison": section_models,
    "4. Feature importance": section_feature_importance,
    "5. Live prediction": section_predict,
    "6. Limitations": section_limitations,
}


def build_app() -> None:
    st.set_page_config(
        page_title="Player Peak Value — ML PoC",
        page_icon="⚽",
        layout="wide",
    )

    with st.sidebar:
        st.markdown("### Navigation")
        choice = st.radio("Section", list(SECTIONS.keys()), label_visibility="collapsed")
        st.markdown("---")
        st.caption("DAT0424 — Malcolm Morgan")
        st.caption("Data: Transfermarkt")
        st.caption("Best model: XGBoost (R² = 0.73)")

    SECTIONS[choice]()


if __name__ == "__main__":
    build_app()
