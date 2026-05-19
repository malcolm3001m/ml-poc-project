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

from config import MODEL_METRICS_FILE, MODELS, RESULTS_DIR
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
    return load_dataset_split()


@st.cache_data(show_spinner=False)
def _metrics() -> pd.DataFrame:
    if not MODEL_METRICS_FILE.exists():
        return pd.DataFrame()
    return pd.read_csv(MODEL_METRICS_FILE)


@st.cache_data(show_spinner=False)
def _validation_table() -> pd.DataFrame:
    p = RESULTS_DIR / "validation_metrics.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


@st.cache_data(show_spinner=False)
def _ablation_table() -> pd.DataFrame:
    p = RESULTS_DIR / "ablation.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


@st.cache_data(show_spinner=False)
def _per_position_table() -> pd.DataFrame:
    p = RESULTS_DIR / "per_position_mae.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


@st.cache_data(show_spinner=False)
def _residuals_table() -> pd.DataFrame:
    p = RESULTS_DIR / "residuals.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


@st.cache_resource(show_spinner=False)
def _load_model(path: Path):
    return joblib.load(path)


def _underlying_regressor(model):
    """Unwrap TransformedTargetRegressor to expose feature_importances_, etc."""
    return getattr(model, "regressor_", model)


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
        players, appearances, clubs, games, transfers). Target: `highest_market_value_in_eur`,
        log-transformed during training, inverse-transformed for prediction. Best model:
        **XGBoost — test R² = 0.726, MAE €1.79M, 5-fold CV R² = 0.707 ± 0.013.**

        ### What this app shows
        1. Dataset overview & EDA
        2. Model comparison (vs. baselines, with CV stability)
        3. Feature importance + leakage ablation
        4. Predicted vs. actual residuals + per-position errors
        5. **Live prediction demo** — pick any player, see what the model predicts vs. their real peak
        6. Limitations + defence FAQ
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
    st.caption("Long right tail. The bulk of players sit under €5M; a small elite reach >€100M. Log scale on y.")
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
        labels={"age": "Age (years, today)", TARGET: "Peak market value (€)"},
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
        st.caption(
            "⚠️ `age` is computed as *today − date of birth*, not age at peak. "
            "This means for retired players the feature is post-peak — addressed in Limitations."
        )


def section_models() -> None:
    st.header("🤖 Model comparison")

    metrics = _metrics()
    validation = _validation_table()

    if metrics.empty:
        st.warning("Run `python scripts/main.py` to generate `results/model_metrics.csv`.")
        return

    st.subheader("Test-set performance")
    display = metrics[["model_name", "mae_eur", "rmse_eur", "r2"]].copy()
    display["mae_eur"] = display["mae_eur"].apply(_eur)
    display["rmse_eur"] = display["rmse_eur"].apply(_eur)
    display["r2"] = display["r2"].apply(lambda v: f"{v:.3f}")
    display.columns = ["Model", "MAE", "RMSE", "R²"]
    st.dataframe(display, width='stretch', hide_index=True)

    fig = px.bar(
        metrics, x="model_name", y="r2", text="r2",
        labels={"model_name": "Model", "r2": "R²"},
        range_y=[-0.2, 1],
    )
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    st.plotly_chart(fig, width='stretch')

    if not validation.empty:
        st.subheader("Overfit check — train R² vs test R²")
        overfit = validation[validation["kind"] == "overfit"].copy()
        if not overfit.empty:
            view = overfit[["name", "train_r2", "test_r2", "overfit_gap"]].copy()
            view["train_r2"] = view["train_r2"].apply(lambda v: f"{v:.3f}")
            view["test_r2"] = view["test_r2"].apply(lambda v: f"{v:.3f}")
            view["overfit_gap"] = view["overfit_gap"].apply(lambda v: f"{v:+.3f}")
            view.columns = ["Model", "Train R²", "Test R²", "Gap"]
            st.dataframe(view, width='stretch', hide_index=True)
            st.caption(
                "XGBoost shows almost no train/test gap (+0.017) — strong generalisation. "
                "Random Forest overfits modestly (+0.168) but still beats Decision Tree on test."
            )

        st.subheader("Stability — 5-fold cross-validation")
        cv = validation[validation["kind"] == "cv"].copy()
        if not cv.empty:
            fig = px.bar(
                cv, x="name", y="cv_r2_mean",
                error_y="cv_r2_std",
                labels={"name": "Model", "cv_r2_mean": "CV R² (mean ± std)"},
                range_y=[0, 1],
            )
            st.plotly_chart(fig, width='stretch')
            best = cv.loc[cv["cv_r2_mean"].idxmax()]
            st.caption(
                f"Best CV: **{best['name']}** at R² {best['cv_r2_mean']:.3f} ± {best['cv_r2_std']:.3f} "
                f"(min {best['cv_r2_min']:.3f}, max {best['cv_r2_max']:.3f}). The test-set 0.726 "
                "sits inside this band → not a lucky split."
            )

        st.subheader("Baselines — does the model actually beat dumb predictors?")
        bl = validation[validation["kind"] == "baseline"].copy()
        if not bl.empty:
            view = bl[["name", "r2", "mae_eur"]].copy()
            view["r2"] = view["r2"].apply(lambda v: f"{v:.3f}")
            view["mae_eur"] = view["mae_eur"].apply(_eur)
            view.columns = ["Baseline", "R²", "MAE"]
            st.dataframe(view, width='stretch', hide_index=True)
            st.markdown(
                """
                **Read:** predicting the global median gives R² ≈ −0.08 (worse than the mean — the
                median is biased low because of the long tail). Linear regression on label-encoded
                categoricals collapses catastrophically — proving that **non-linear models are
                required**, not optional, for this data. XGBoost's 0.726 is real signal.
                """
            )


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
    inner = _underlying_regressor(model)

    if not hasattr(inner, "feature_importances_"):
        st.info("Selected model exposes no feature importances.")
        return

    importances = pd.DataFrame(
        {"feature": X_train.columns, "importance": inner.feature_importances_}
    ).sort_values("importance", ascending=True)

    fig = px.bar(
        importances, x="importance", y="feature", orientation="h",
        labels={"importance": "Importance", "feature": "Feature"},
    )
    st.plotly_chart(fig, width='stretch')

    top = importances.tail(3)["feature"].tolist()[::-1]
    st.markdown(
        f"**Top 3 drivers:** `{top[0]}`, `{top[1]}`, `{top[2]}`."
    )

    st.divider()
    st.subheader("Leakage ablation — what if we drop `current_club_domestic_competition_id`?")
    ab = _ablation_table()
    if ab.empty:
        st.info("Run `python scripts/validate.py` to generate `results/ablation.csv`.")
    else:
        view = ab.copy()
        view["mae_eur"] = view["mae_eur"].apply(lambda v: ("+" if v >= 0 else "") + _eur(v) if abs(v) < 1e8 else _eur(v))
        view["rmse_eur"] = view["rmse_eur"].apply(lambda v: ("+" if v >= 0 else "") + _eur(v) if abs(v) < 1e8 else _eur(v))
        view["r2"] = view["r2"].apply(lambda v: f"{v:+.3f}")
        view.columns = ["Variant", "MAE", "RMSE", "R²"]
        st.dataframe(view, width='stretch', hide_index=True)
        st.markdown(
            """
            **Read:** the suspected leak feature contributes only **0.02 R²** and **€0.15M MAE**.
            Material but not load-bearing — the model would still hit R² ≈ 0.71 without it.
            This is the right answer to "isn't this just leakage?" — no, we tested it.
            """
        )


def section_diagnostics() -> None:
    st.header("🔬 Predicted vs. actual + per-position errors")
    residuals = _residuals_table()
    per_pos = _per_position_table()

    if residuals.empty:
        st.warning("Run `python scripts/validate.py` to generate `results/residuals.csv`.")
        return

    st.subheader("Predicted vs. actual peak value — XGBoost on test set")
    st.caption("Closer to the diagonal = better. Log scale to handle the long tail.")
    fig = px.scatter(
        residuals, x="actual_eur", y="predicted_eur", color="position",
        hover_data=["name"], opacity=0.5, log_x=True, log_y=True,
        labels={"actual_eur": "Actual peak value (€)", "predicted_eur": "Predicted (€)"},
    )
    lo = max(residuals["actual_eur"].min(), 1)
    hi = residuals["actual_eur"].max()
    fig.add_shape(type="line", x0=lo, y0=lo, x1=hi, y1=hi, line=dict(dash="dash", color="red"))
    st.plotly_chart(fig, width='stretch')

    if not per_pos.empty:
        st.subheader("MAE by position")
        view = per_pos.copy()
        view["median_actual_eur"] = view["median_actual_eur"].apply(_eur)
        view["mae_eur"] = view["mae_eur"].apply(_eur)
        view.columns = ["Position", "N (test)", "Median actual", "MAE"]
        st.dataframe(view, width='stretch', hide_index=True)
        st.caption(
            "Relative error (MAE / median actual) is fairly constant across positions — "
            "the model isn't biased against goalkeepers or forwards in proportional terms. "
            "Absolute MAE scales with position value."
        )


def section_predict() -> None:
    st.header("🎯 Live prediction demo")
    st.markdown(
        "Pick any player. The app pulls their feature row, asks each model to predict, "
        "and compares to the **real** peak. Each player is tagged TRAIN or TEST — predictions on "
        "TRAIN players are *not* generalisation; pick a TEST player for an honest read."
    )

    raw = _raw_players_and_appearances()
    X_train, X_test, y_train, y_test = _splits()
    train_idx = set(X_train.index)

    X_all = pd.concat([X_train, X_test])
    y_all = pd.concat([y_train, y_test])

    raw_aligned = raw.loc[X_all.index].copy()
    raw_aligned["split"] = raw_aligned.index.map(lambda i: "TRAIN" if i in train_idx else "TEST")
    raw_aligned["__display"] = (
        raw_aligned["split"] + " · "
        + raw_aligned["name"].fillna("Unknown")
        + " (" + raw_aligned["position"].fillna("?") + ", "
        + raw_aligned[TARGET].apply(_eur) + ")"
    )

    c1, c2 = st.columns([1, 1])
    show_only_test = c1.toggle("Show only TEST players (honest generalisation)", value=True)
    random_pick = c2.button("🎲 Random test player")

    pool = raw_aligned[raw_aligned["split"] == "TEST"] if show_only_test else raw_aligned

    if random_pick:
        choice = int(pool.sample(1, random_state=None).index[0])
        st.session_state["picked_player"] = choice

    default_idx = 0
    if "picked_player" in st.session_state and st.session_state["picked_player"] in pool.index:
        default_idx = list(pool.index).index(st.session_state["picked_player"])
    else:
        famous = pool[pool["name"].str.contains("Mbapp|Haaland|Bellingham|Vinic", case=False, na=False)]
        if not famous.empty:
            default_idx = list(pool.index).index(famous.index[0])

    choice = st.selectbox(
        "Player",
        options=pool.index,
        format_func=lambda i: pool.loc[i, "__display"],
        index=default_idx,
    )

    row = X_all.loc[[choice]]
    actual = float(y_all.loc[choice])
    split_label = raw_aligned.loc[choice, "split"]

    st.subheader(raw_aligned.loc[choice, "name"])
    if split_label == "TRAIN":
        st.warning(
            "⚠️ This player was in the **training set** — the model has seen them before. "
            "Errors will look artificially low. Toggle 'Show only TEST players' for honest predictions."
        )
    else:
        st.success("✅ TEST-set player — model has never seen this row. Predictions are honest.")

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
        ### 1. Post-peak feature leak (the conceptual issue)
        Features are **career-cumulative**: total goals, total minutes, total assists. The
        target is **highest market value reached**. For a player who peaked at 22 and is
        now 30, we're using 8 years of post-peak data to "predict" their peak. Strictly
        speaking this is a **retrospective valuation model**, not a forward-looking one.
        A cleaner version would freeze features at age 21 and predict peak at age 25+.

        ### 2. `age` = today − date of birth
        Not age at peak. For retired players the feature is whatever their current age is,
        not the age they hit their peak. This adds noise but doesn't bias the result
        systematically. Same fix as (1): use age-at-peak instead.

        ### 3. Random vs. temporal split
        Train/test split is random (seed 42), not chronological. A deployed model would
        train on players who peaked before year Y and predict for players still active.
        We didn't do that — the random split lets the model see contemporaries of its
        targets.

        ### 4. Suspected leakage tested — small effect
        We suspected `current_club_domestic_competition_id` of leaking value (a player at
        PSG is valuable *because* they're at PSG). We ran the ablation: dropping the
        feature costs **−0.020 R²** and **+€0.15M MAE**. Material but not load-bearing.
        The model's signal is real, not just league-prestige memorisation.

        ### 5. MAE is misleading in absolute terms
        €1.79M MAE next to Mbappé (~€180M peak) is rounding error. €1.79M next to the
        median player (~€500k) is **3.5× their value**. The model is most useful at the
        top of the distribution — exactly where the money is, but don't claim it predicts
        an academy player's peak.

        ### 6. Missing context
        Agent fees, marketability, social-media reach, injury history, club transfer
        policy, Saudi/MLS spending booms. Things the model can't see explain the 27%
        unexplained variance.

        ### What we'd build next
        - Re-frame as "predict peak at age 25 given features up to age 21."
        - Add time-series features (form, trajectory) instead of career aggregates.
        - Try quantile regression for prediction intervals instead of point estimates.
        - SHAP for per-player explanations.
        - Temporal split (train on pre-2018, test on 2018+).
        """
    )

    st.divider()
    st.subheader("🎤 Defence FAQ — likely teacher questions, pre-canned answers")
    st.markdown(
        """
        **Q1: "Isn't this just leakage? You're using career stats to predict a career peak."**
        Yes — it's a retrospective valuation, not a prediction in the strict sense. We
        disclose this up-front. The honest fix is to freeze features at a fixed age and
        predict forward; we ran out of time for that re-frame. The ablation on the most
        obviously-leaky feature (`current_club_competition_id`) shows the model survives
        without it (R² 0.706 vs 0.726), so it's not pure memorisation.

        **Q2: "How do you know R² 0.73 is good?"**
        Baselines. Predict-median gives R² −0.08. Linear regression on these features
        collapses to R² −755 (catastrophic on the long tail). Tree models cut MAE from
        €3.33M (median predictor) to €1.79M (XGBoost). 5-fold CV puts XGB at
        **0.707 ± 0.013** — the 0.726 we report sits inside that band, not above it.

        **Q3: "Is the model overfit?"**
        XGBoost train R² 0.743, test R² 0.726 — gap of **0.017**. Effectively no overfit.
        Random Forest overfits more (+0.168) but still beats Decision Tree on test.
        Decision Tree under-fits (train 0.648, test 0.547). XGBoost was the right choice.

        **Q4: "Where does the model fail?"**
        Per-position MAE: Goalkeepers €0.94M, Defenders €1.54M, Midfielders €1.95M,
        Attackers €2.25M. Absolute MAE scales with position value, but **relative** error
        is roughly flat — model isn't biased against a position class. It fails worst on
        the long right tail (€100M+ players) where individual deals dominate.

        **Q5: "Could you have used a deep learning model?"**
        Probably not usefully. 39k rows is small for neural nets; the features are
        tabular and label-encoded. XGBoost is the right tool for this size and data shape.
        A small MLP might marginally beat XGB but would need much more regularisation
        and lose interpretability. Trade-off not worth it at PoC stage.
        """
    )


# ---------- Entry point ----------

SECTIONS = {
    "1. Overview": section_overview,
    "2. Dataset & EDA": section_dataset,
    "3. Model comparison + baselines": section_models,
    "4. Feature importance + ablation": section_feature_importance,
    "5. Diagnostics (residuals + per-position)": section_diagnostics,
    "6. Live prediction": section_predict,
    "7. Limitations + Defence FAQ": section_limitations,
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
        st.caption("Best model: XGBoost · test R² 0.726 · CV 0.707 ± 0.013")

    SECTIONS[choice]()


if __name__ == "__main__":
    build_app()
