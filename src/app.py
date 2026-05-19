"""Streamlit presentation — Transfermarkt peak market value PoC.

Structured as a non-technical pitch: I had X problem → did Y → got Z solution.
"""

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


# ---------- Brand / CSS ----------

BRAND_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;700&family=Space+Grotesk:wght@500;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, sans-serif;
}

h1, h2, h3 {
    font-family: 'Space Grotesk', sans-serif !important;
    letter-spacing: -0.02em;
}

h1 { font-weight: 700 !important; }

.pitch-hero {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 3.6rem;
    line-height: 1.05;
    font-weight: 700;
    letter-spacing: -0.03em;
    margin: 0.5rem 0 1.5rem 0;
}
.pitch-hero .accent { color: #0a7c4a; }
.pitch-hero .alt    { color: #c1272d; }

.pitch-kicker {
    font-size: 0.9rem;
    font-weight: 500;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #6b7280;
    margin-bottom: 0.25rem;
}

.pitch-lede {
    font-size: 1.25rem;
    line-height: 1.55;
    color: #374151;
    max-width: 46rem;
    margin: 0 0 2rem 0;
}

.section-question {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.7rem;
    font-weight: 700;
    line-height: 1.25;
    margin: 0 0 0.25rem 0;
}

.section-answer {
    color: #6b7280;
    font-size: 1rem;
    margin: 0 0 1.25rem 0;
}

.stat-strip {
    display: flex;
    gap: 3rem;
    margin: 2rem 0;
    padding: 1.25rem 0;
    border-top: 1px solid #e5e7eb;
    border-bottom: 1px solid #e5e7eb;
}
.stat-strip .stat .num {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.1rem;
    font-weight: 700;
    color: #0a7c4a;
    line-height: 1;
}
.stat-strip .stat .lbl {
    font-size: 0.78rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #6b7280;
    margin-top: 0.4rem;
}

.callout {
    background: #f8faf9;
    border-left: 3px solid #0a7c4a;
    padding: 1rem 1.25rem;
    margin: 1.25rem 0;
    font-size: 0.97rem;
    line-height: 1.55;
}
.callout.warn { border-left-color: #c1272d; background: #fdf6f6; }

[data-testid="stSidebar"] {
    background: #f9fafb;
}
[data-testid="stSidebar"] .stRadio label {
    font-size: 0.95rem;
}

footer { visibility: hidden; }
</style>
"""


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
    return pd.read_csv(MODEL_METRICS_FILE) if MODEL_METRICS_FILE.exists() else pd.DataFrame()


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
    return getattr(model, "regressor_", model)


def _eur(x: float) -> str:
    if x >= 1_000_000:
        return f"€{x / 1_000_000:.2f}M"
    if x >= 1_000:
        return f"€{x / 1_000:.0f}k"
    return f"€{x:.0f}"


def _question_header(question: str, answer: str | None = None) -> None:
    st.markdown(f'<div class="section-question">{question}</div>', unsafe_allow_html=True)
    if answer:
        st.markdown(f'<div class="section-answer">{answer}</div>', unsafe_allow_html=True)


def _kicker(text: str) -> None:
    st.markdown(f'<div class="pitch-kicker">{text}</div>', unsafe_allow_html=True)


# ---------- Sections ----------

def section_problem() -> None:
    _kicker("DAT0424 · ML Proof of Concept · Malcolm Morgan")
    st.markdown(
        '<div class="pitch-hero">'
        'What is a football player <span class="accent">actually worth</span> '
        'at their <span class="alt">peak</span>?'
        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="pitch-lede">'
        'Transfer market values on Transfermarkt are crowd-sourced and lag reality. '
        "Clubs, agents, and betting markets all need a defensible number for what a "
        "player will be worth at their career peak. This proof of concept asks: can "
        "we predict that number from biographical and performance data alone?"
        '</div>',
        unsafe_allow_html=True,
    )

    raw = _raw_players_and_appearances()
    st.markdown(
        f'''
        <div class="stat-strip">
            <div class="stat"><div class="num">{len(raw):,}</div><div class="lbl">Players</div></div>
            <div class="stat"><div class="num">189</div><div class="lbl">Nationalities</div></div>
            <div class="stat"><div class="num">14</div><div class="lbl">Features</div></div>
            <div class="stat"><div class="num">€1.79M</div><div class="lbl">Avg. prediction error</div></div>
            <div class="stat"><div class="num">0.726</div><div class="lbl">Test R²</div></div>
        </div>
        ''',
        unsafe_allow_html=True,
    )

    _question_header(
        "Why does this matter?",
        "Three concrete use-cases where a defensible peak-value estimate is worth real money.",
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**🔎 Transfer scouting**")
        st.caption(
            "Flag undervalued players before the market reprices them. A €5M signing "
            "that peaks at €40M is a fund-defining return."
        )
    with c2:
        st.markdown("**📝 Contract negotiation**")
        st.caption(
            "Anchor wage offers to a defensible ceiling. Agents and clubs both need a "
            "neutral third-party number to argue around."
        )
    with c3:
        st.markdown("**📊 Squad valuation**")
        st.caption(
            "Value a squad as a portfolio of human capital — required by accountants, "
            "useful for investors, betting markets, and fantasy products."
        )

    st.markdown(
        '<div class="callout">'
        "<b>The story of this app, in three lines:</b><br>"
        "I had a <b>problem</b> — peak market value is unpredictable and crowd-sourced.<br>"
        "I did <b>Y</b> — trained three tree-based models on 39k players' biographies.<br>"
        "I got <b>Z</b> — XGBoost predicts peak value with R² 0.726, validated against baselines, ablations and cross-validation."
        '</div>',
        unsafe_allow_html=True,
    )


# ---- Section 2: The Data ----

@st.cache_data(show_spinner=False)
def _country_peak_value() -> pd.DataFrame:
    raw = _raw_players_and_appearances()
    df = raw.dropna(subset=["country_of_citizenship"]).copy()
    # Plotly's "country names" mode wants UK, not England/Scotland/Wales
    home_nations = {"England": "United Kingdom", "Scotland": "United Kingdom",
                    "Wales": "United Kingdom", "Northern Ireland": "United Kingdom"}
    df["country"] = df["country_of_citizenship"].replace(home_nations)
    agg = (
        df.groupby("country")
        .agg(n=("player_id", "count"), median_peak=(TARGET, "median"),
             mean_peak=(TARGET, "mean"))
        .reset_index()
    )
    # Filter long tail: only countries with at least 50 players → median is robust
    agg = agg[agg["n"] >= 50].sort_values("median_peak", ascending=False)
    return agg


def section_data() -> None:
    _kicker("Step 1 of 3 · The data")
    _question_header(
        "Do we even have enough signal to predict this?",
        "39,226 players with non-null peak-value labels. 14 features. Let's look.",
    )

    raw = _raw_players_and_appearances()

    # The "hook" chart: target distribution showing why this is hard
    st.markdown("##### How is value distributed across players?")
    st.caption(
        "**Problem:** the target has a brutal long right tail. Median player is worth ~€500k; "
        "Mbappé/Haaland-tier is €150M+. Any model has to handle a 300× value spread."
    )
    fig = px.histogram(
        raw, x=TARGET, nbins=80, log_y=True,
        labels={TARGET: "Peak market value (€)"},
        color_discrete_sequence=["#0a7c4a"],
    )
    fig.update_layout(margin=dict(t=10, b=10), height=320, plot_bgcolor="white")
    st.plotly_chart(fig, width='stretch')

    # Geographic story
    st.markdown("##### Where do the most valuable players come from?")
    st.caption(
        "**Following the data-viz principle that** *ponderated data is better*: instead of "
        "showing player counts (which would just map population), we show **median peak value** "
        "per country among countries with at least 50 players. This isolates *talent density* "
        "from *population size*."
    )
    countries = _country_peak_value()
    fig = px.choropleth(
        countries,
        locations="country", locationmode="country names",
        color="median_peak",
        color_continuous_scale=["#e8f5ee", "#a5d4b8", "#3da66c", "#0a7c4a"],
        labels={"median_peak": "Median peak (€)"},
        hover_data={"n": True, "median_peak": ":,.0f"},
    )
    fig.update_layout(margin=dict(t=0, b=0, l=0, r=0), height=420,
                      geo=dict(showframe=False, showcoastlines=True, projection_type="natural earth"))
    st.plotly_chart(fig, width='stretch')

    top5 = countries.head(5)[["country", "n", "median_peak"]].copy()
    top5["median_peak"] = top5["median_peak"].apply(_eur)
    top5.columns = ["Country", "Players (n)", "Median peak value"]
    st.markdown("**Top 5 countries by median peak value:**")
    st.dataframe(top5, width='stretch', hide_index=True)

    # Position story
    st.markdown("##### Does position predict peak value?")
    st.caption(
        "**Yes — but not as much as you'd think.** Attackers and midfielders skew higher, "
        "goalkeepers lower, but every position has its outlier €100M+ player."
    )
    fig = px.box(
        raw.dropna(subset=["position"]), x="position", y=TARGET, log_y=True,
        labels={TARGET: "Peak market value (€)"},
        color="position",
        color_discrete_sequence=["#0a7c4a", "#1f7a4d", "#3da66c", "#6cbf8e"],
    )
    fig.update_layout(margin=dict(t=10, b=10), height=320, showlegend=False, plot_bgcolor="white")
    st.plotly_chart(fig, width='stretch')

    # Age story
    st.markdown("##### Does age tell us anything?")
    st.caption(
        "Not on its own. **Caveat we own up to:** `age` here is *current age*, not *age at peak* — "
        "for retired players it's whatever their age is today. This is a known limitation we "
        "address in the Roadmap."
    )
    sample = raw.dropna(subset=["age", TARGET]).sample(min(5000, len(raw)), random_state=42)
    fig = px.scatter(
        sample, x="age", y=TARGET, opacity=0.35, log_y=True,
        labels={"age": "Age (today)", TARGET: "Peak market value (€)"},
        color_discrete_sequence=["#c1272d"],
    )
    fig.update_layout(margin=dict(t=10, b=10), height=320, plot_bgcolor="white")
    st.plotly_chart(fig, width='stretch')

    with st.expander("📋 Full feature list"):
        st.markdown(
            f"**Categorical ({len(CATEGORICAL_COLS)}):** "
            + ", ".join(f"`{c}`" for c in CATEGORICAL_COLS)
        )
        st.markdown(
            f"**Numeric ({len(NUMERIC_COLS)}):** "
            + ", ".join(f"`{c}`" for c in NUMERIC_COLS)
        )


# ---- Section 3: The Approach ----

def section_approach() -> None:
    _kicker("Step 2 of 3 · The approach")
    _question_header(
        "Why these three models, and why log-transform?",
        "Plain-English methodology — no jargon required.",
    )

    st.markdown(
        """
        **The target is brutal.** A model that loses €0.5M on Mbappé's €180M peak is fine. The
        same €0.5M loss on a €500k median player is a 100% error. Squared loss in € would just
        make the model obsess over the top 1%.

        **The fix:** train every model on a **log-transformed target** (`log(1+value)`) and
        invert back to € at prediction time. Suddenly the loss function cares equally about
        getting a €1M player right and a €100M player right.
        """
    )

    st.markdown("##### Three models, three philosophies")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**🌳 Decision Tree**")
        st.caption(
            "*The simplest possible answer.* One tree, split features greedily, max depth 10. "
            "We expect this to be our baseline floor — interpretable, but unsophisticated."
        )
    with c2:
        st.markdown("**🌲 Random Forest**")
        st.caption(
            "*Ask 140 trees and vote.* Each tree sees a random subset of data + features; "
            "averaging cancels individual errors. Robust, but tends to overfit on deep trees."
        )
    with c3:
        st.markdown("**🚀 XGBoost**")
        st.caption(
            "*Learn from your mistakes, iteratively.* 350 small trees, each correcting the "
            "previous one's residuals. Heavily regularised (max_depth=4, subsample=0.9, L2)."
        )

    st.markdown(
        '<div class="callout">'
        "<b>Why trees, not deep learning or linear models?</b> 39k rows is small for neural nets. "
        "Tabular features with no spatial/sequential structure. Categoricals encoded as integers — "
        "linear models would treat that as ordinal (meaningless). Trees split on values directly, "
        "no scaling needed. XGBoost is the right tool for this size and data shape."
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown("##### What the model sees, in plain terms")
    st.caption(
        "We give each model **5 categorical signals** (position, foot, nationality, league) "
        "and **9 numeric signals** (age, height, international caps, career goals/assists/minutes, "
        "cards). The model has to figure out — purely from these 14 numbers — how high this "
        "player's market value will go."
    )


# ---- Section 4: The Discovery ----

def section_discovery() -> None:
    _kicker("Step 3 of 3 · The discovery")
    _question_header(
        "Did it work?",
        "Spoiler: yes. Here's the evidence in three forms.",
    )

    metrics = _metrics()
    validation = _validation_table()

    if metrics.empty:
        st.warning("Run `python scripts/main.py` to generate `results/model_metrics.csv`.")
        return

    # The headline
    st.markdown("##### Headline: XGBoost wins, by a clean margin")
    display = metrics[["model_name", "mae_eur", "rmse_eur", "r2"]].copy()
    display["mae_eur"] = display["mae_eur"].apply(_eur)
    display["rmse_eur"] = display["rmse_eur"].apply(_eur)
    display["r2"] = display["r2"].apply(lambda v: f"{v:.3f}")
    display.columns = ["Model", "MAE (€)", "RMSE (€)", "R²"]
    st.dataframe(display, width='stretch', hide_index=True)
    st.caption(
        "Read: XGBoost cuts the average prediction error to **€1.79M** and explains "
        "**72.6%** of the variance in peak market value. Random Forest is close behind; "
        "Decision Tree underfits."
    )

    # Vs baselines — the "does this beat trivial?" answer
    st.markdown("##### But how do we know that 0.726 is actually *good*?")
    st.caption(
        "**Compared to what?** Two trivial baselines establish the floor."
    )
    bl = validation[validation["kind"] == "baseline"].copy() if not validation.empty else pd.DataFrame()
    if not bl.empty:
        view = bl[["name", "r2", "mae_eur"]].copy()
        view["r2"] = view["r2"].apply(lambda v: f"{v:.3f}")
        view["mae_eur"] = view["mae_eur"].apply(_eur)
        view.columns = ["Baseline", "R²", "MAE"]
        st.dataframe(view, width='stretch', hide_index=True)
        st.markdown(
            """
            **Read:** predicting the median for every player gives R² **−0.08** and MAE **€3.33M**.
            Linear regression on encoded categoricals **collapses catastrophically** (R² −755) —
            useful evidence that **non-linear models are required, not just preferred** for this
            kind of data. XGBoost cuts the trivial baseline's MAE by **46%**.
            """
        )

    # Stability — the "is this a lucky split?" answer
    st.markdown("##### Is the 0.726 a lucky split, or stable?")
    cv = validation[validation["kind"] == "cv"].copy() if not validation.empty else pd.DataFrame()
    if not cv.empty:
        fig = px.bar(
            cv, x="name", y="cv_r2_mean", error_y="cv_r2_std",
            labels={"name": "Model", "cv_r2_mean": "CV R² (mean ± std)"},
            range_y=[0, 1],
            color_discrete_sequence=["#0a7c4a"],
        )
        fig.update_layout(margin=dict(t=10, b=10), height=300, plot_bgcolor="white")
        st.plotly_chart(fig, width='stretch')
        best = cv.loc[cv["cv_r2_mean"].idxmax()]
        st.caption(
            f"5-fold cross-validation puts XGBoost at **R² {best['cv_r2_mean']:.3f} ± "
            f"{best['cv_r2_std']:.3f}** (range {best['cv_r2_min']:.3f}–{best['cv_r2_max']:.3f}). "
            "Our test-set 0.726 sits inside this band — not a lucky split."
        )

    # Overfit — the "is this memorisation?" answer
    overfit = validation[validation["kind"] == "overfit"].copy() if not validation.empty else pd.DataFrame()
    if not overfit.empty:
        with st.expander("📐 Technical: train R² vs. test R² (overfit check)"):
            view = overfit[["name", "train_r2", "test_r2", "overfit_gap"]].copy()
            view["train_r2"] = view["train_r2"].apply(lambda v: f"{v:.3f}")
            view["test_r2"] = view["test_r2"].apply(lambda v: f"{v:.3f}")
            view["overfit_gap"] = view["overfit_gap"].apply(lambda v: f"{v:+.3f}")
            view.columns = ["Model", "Train R²", "Test R²", "Gap"]
            st.dataframe(view, width='stretch', hide_index=True)
            st.caption(
                "XGBoost's train/test gap is only **+0.017** — effectively no overfit, "
                "thanks to depth=4 + subsampling + L2 regularisation. RF overfits more (+0.168)."
            )

    # What did the model learn — feature importance
    st.markdown("##### What did the model actually learn?")
    X_train, _, _, _ = _splits()
    xgb = _load_model(MODELS["xgboost_regressor"]["path"])
    inner = _underlying_regressor(xgb)
    importances = pd.DataFrame(
        {"feature": X_train.columns, "importance": inner.feature_importances_}
    ).sort_values("importance", ascending=True)
    top3 = importances.tail(3)["feature"].tolist()[::-1]

    fig = px.bar(
        importances, x="importance", y="feature", orientation="h",
        labels={"importance": "Importance", "feature": "Feature"},
        color_discrete_sequence=["#0a7c4a"],
    )
    fig.update_layout(margin=dict(t=10, b=10), height=380, plot_bgcolor="white")
    st.plotly_chart(fig, width='stretch')
    st.caption(
        f"**Top 3 drivers:** `{top3[0]}`, `{top3[1]}`, `{top3[2]}`. Roughly: how much "
        "you've played + at what level + in what position."
    )


# ---- Section 5: The Validation ----

def section_validation() -> None:
    _kicker("Pressure-testing the result")
    _question_header(
        "How could this be wrong, and did we test for it?",
        "Three checks: residuals, per-position errors, and a leakage ablation.",
    )

    residuals = _residuals_table()
    per_pos = _per_position_table()
    ablation = _ablation_table()

    st.markdown("##### Predicted vs. actual — how calibrated is the model?")
    st.caption(
        "Each dot is one player in the test set. A perfect model puts every dot on the diagonal. "
        "Log scale on both axes because of the long tail."
    )
    if not residuals.empty:
        fig = px.scatter(
            residuals, x="actual_eur", y="predicted_eur", color="position",
            hover_data=["name"], opacity=0.5, log_x=True, log_y=True,
            labels={"actual_eur": "Actual peak value (€)", "predicted_eur": "Predicted (€)"},
            color_discrete_sequence=["#0a7c4a", "#3da66c", "#c1272d", "#6b7280", "#9ca3af"],
        )
        lo = max(residuals["actual_eur"].min(), 1)
        hi = residuals["actual_eur"].max()
        fig.add_shape(type="line", x0=lo, y0=lo, x1=hi, y1=hi, line=dict(dash="dash", color="#111"))
        fig.update_layout(margin=dict(t=10, b=10), height=460, plot_bgcolor="white")
        st.plotly_chart(fig, width='stretch')
        st.caption(
            "**Read:** the model is well-calibrated up to ~€20M peak value. Above that, "
            "the long-tail superstars are systematically *under*-predicted — the model is "
            "conservative when it should extrapolate."
        )

    st.markdown("##### Where does the model fail?")
    if not per_pos.empty:
        view = per_pos.copy()
        view["mae_pct_of_median"] = (view["mae_eur"] / view["median_actual_eur"] * 100).round(0)
        view["median_actual_eur"] = view["median_actual_eur"].apply(_eur)
        view["mae_eur"] = view["mae_eur"].apply(_eur)
        view = view[["position", "n", "median_actual_eur", "mae_eur", "mae_pct_of_median"]]
        view.columns = ["Position", "N (test)", "Median actual", "MAE", "MAE as % of median"]
        st.dataframe(view, width='stretch', hide_index=True)
        st.caption(
            "**Read:** absolute MAE scales with position value (forwards cost more, so absolute "
            "errors are bigger). But the **proportional** error — MAE as % of median actual — is "
            "roughly flat across positions. **No positional bias.**"
        )

    st.markdown("##### What about that suspicious feature?")
    st.caption(
        "We flagged `current_club_domestic_competition_id` as a potential leak — a player at "
        "PSG is valuable *because* they're at PSG. Did we test it?"
    )
    if not ablation.empty:
        view = ablation.copy()
        view["mae_eur"] = view["mae_eur"].apply(lambda v: ("+" if v >= 0 else "") + _eur(v) if abs(v) < 1e8 else _eur(v))
        view["rmse_eur"] = view["rmse_eur"].apply(lambda v: ("+" if v >= 0 else "") + _eur(v) if abs(v) < 1e8 else _eur(v))
        view["r2"] = view["r2"].apply(lambda v: f"{v:+.3f}")
        view.columns = ["Variant", "MAE", "RMSE", "R²"]
        st.dataframe(view, width='stretch', hide_index=True)
        st.markdown(
            '<div class="callout">'
            "<b>Yes, and the result defends us:</b> dropping the suspected leak feature costs "
            "only <b>−0.020 R²</b> and <b>+€0.15M MAE</b>. The model still hits R² ≈ 0.71 "
            "without it. The signal is real — not just league-prestige memorisation."
            '</div>',
            unsafe_allow_html=True,
        )


# ---- Section 6: Try it yourself ----

def section_predict() -> None:
    _kicker("Try it yourself")
    _question_header(
        "Pick a player. What does the model think they're worth?",
        "Tagged TRAIN or TEST — pick TEST players for an honest read.",
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
    show_only_test = c1.toggle("Show only TEST players (honest predictions)", value=True)
    random_pick = c2.button("🎲 Random TEST player")

    pool = raw_aligned[raw_aligned["split"] == "TEST"] if show_only_test else raw_aligned

    if random_pick:
        choice = int(pool.sample(1).index[0])
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
        st.markdown(
            '<div class="callout warn">⚠️ This player was in the <b>training set</b> — '
            "the model has seen them before. Errors will look artificially low. Toggle "
            "<b>'Show only TEST players'</b> for honest predictions.</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="callout">✅ TEST-set player — model has never seen this row. '
            "Predictions are honest generalisation.</div>",
            unsafe_allow_html=True,
        )

    meta = raw_aligned.loc[choice]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Position", str(meta.get("position", "?")))
    c2.metric("Foot", str(meta.get("foot", "?")))
    c3.metric("Age", f"{meta.get('age', 0):.0f}" if pd.notna(meta.get("age")) else "?")
    c4.metric("Real peak value", _eur(actual))

    preds = []
    for key, spec in MODELS.items():
        model = _load_model(spec["path"])
        pred = float(model.predict(row)[0])
        pred = max(pred, 0.0)
        err = pred - actual
        pct = (abs(err) / actual * 100) if actual > 0 else 0
        preds.append({"Model": spec["name"], "Predicted": pred, "Error": err, "Error %": pct})

    pred_df = pd.DataFrame(preds)
    show = pred_df.copy()
    show["Predicted"] = show["Predicted"].apply(_eur)
    show["Error"] = show["Error"].apply(lambda v: ("+" if v >= 0 else "") + _eur(v))
    show["Error %"] = show["Error %"].apply(lambda v: f"{v:.0f}%")
    st.dataframe(show, width='stretch', hide_index=True)

    fig = px.bar(
        pred_df, x="Model", y="Predicted", text="Predicted",
        labels={"Predicted": "Predicted peak value (€)"},
        color_discrete_sequence=["#0a7c4a"],
    )
    fig.update_traces(texttemplate="€%{text:,.0f}", textposition="outside")
    fig.add_hline(
        y=actual, line_dash="dash", line_color="#c1272d",
        annotation_text=f"Real peak: {_eur(actual)}", annotation_position="top right",
    )
    fig.update_layout(margin=dict(t=20, b=10), height=360, plot_bgcolor="white")
    st.plotly_chart(fig, width='stretch')


# ---- Section 7: Honest limits ----

def section_limits() -> None:
    _kicker("Honest limits")
    _question_header(
        "Where does this PoC fall short?",
        "Three real limitations we own up to. Volunteering these is more credible than hiding them.",
    )

    st.markdown(
        """
        ##### 1. Retrospective valuation, not forward prediction
        Features are **career-cumulative** (total goals, total minutes); target is the **career peak**.
        For a retired player, those features include years of post-peak play. This is a
        *retrospective* model, not a forward-looking one. The honest reframe is *"predict peak
        at age 25 given features up to age 21"* — that's the Roadmap.

        ##### 2. `age` is current age, not age-at-peak
        `today − date_of_birth`. For retired players that's whatever their age is today, not
        the age they peaked at. Adds noise; doesn't systematically bias the result.

        ##### 3. Random train/test split, not temporal
        A deployed model would train on players who peaked before year Y and predict for
        players active in year Y. Our split is random (seed 42) — the model can see
        contemporaries of its targets. The CV result protects against overfit but doesn't
        cure this conceptual gap.
        """
    )

    with st.expander("🎤 Defence FAQ — likely questions, pre-canned answers"):
        st.markdown(
            """
            **Q: Isn't this just leakage? You're using career stats to predict a career peak.**
            Yes — retrospective valuation, not forward prediction. We disclose it. We tested
            the most-suspect feature (`current_club_competition_id`) by ablation — dropping
            it costs only −0.020 R². The model isn't pure memorisation.

            **Q: How do you know R² 0.73 is good?**
            Baselines. Predict-median: R² −0.08. Linear regression: R² −755. XGBoost cuts
            MAE from €3.33M to €1.79M — a 46% reduction over the trivial baseline. CV
            confirms the 0.726 is real, not luck (CV 0.707 ± 0.013).

            **Q: Is the model overfit?**
            XGBoost train R² 0.743 / test R² 0.726 — gap of +0.017. Effectively no overfit.
            RF overfits more (+0.168) but still beats DT on test.

            **Q: Where does the model fail?**
            Long right tail. Players above €50M are systematically under-predicted —
            individual transfer deals dominate up there. Per-position MAE scales with
            value but stays roughly flat as % of median (no positional bias).

            **Q: Why XGBoost and not deep learning?**
            39k rows is small for neural nets. Features are tabular. Trees handle
            label-encoded categoricals natively; linear models (and most NNs without
            embedding layers) treat them as ordinal — meaningless. XGBoost is right-sized.
            """
        )


# ---- Section 8: The roadmap ----

def section_roadmap() -> None:
    _kicker("What we'd build next")
    _question_header(
        "If this PoC became a v1, what changes?",
        "Five upgrades — each defendable today, each testable. This is the proof, "
        "looking forward.",
    )

    st.markdown(
        """
        ##### 1. Age-at-peak refraining (kills Limitation #1 and #2 in one move)
        Freeze every player's features at age 21 — career stats up to 21, no current-club, no
        age-leak — and predict their eventual peak. This converts the model from retrospective
        to genuinely forward-looking. Requires per-player snapshot data (we have it; just need
        to slice appearances by date).

        ##### 2. Temporal train/test split (kills Limitation #3)
        Train on players who peaked before 2018; test on players who peaked 2018+. Tests
        whether the model generalises across eras (market inflation, league economics shift).
        Expected outcome: lower R² but a more defensible deployment story.

        ##### 3. Quantile regression → confidence intervals, not point estimates
        Instead of saying "€42M", say "**€32–€58M, 80% confidence**". Gradient-boosted
        quantile regression (already supported in XGBoost via custom objective). Way more
        useful to a scout or analyst than a point number.

        ##### 4. SHAP for per-player explanations
        For each prediction, surface *which features drove this specific number up or down*.
        E.g. "Bellingham gets +€40M from being a 21-year-old midfielder at a top-5 league
        club, −€8M from his international caps being lower than peers." Turns the model
        from black box into negotiation tool.

        ##### 5. Temporal performance features
        Replace career-cumulative stats with **trajectory** features: goals-per-90 trend,
        minutes growth rate, age-vs-output curve fit. Captures *trajectory*, not just totals.
        Strongest expected lift.
        """
    )

    st.markdown(
        '<div class="callout">'
        "<b>The teacher said theoretically provable, not-yet-achieved work counts as part "
        "of the PoC.</b> All five items above are testable — none requires new data, all "
        "extend the existing pipeline. The roadmap is the proof that this isn't just a "
        "one-shot model, it's the foundation of a real product."
        '</div>',
        unsafe_allow_html=True,
    )


# ---------- Entry point ----------

SECTIONS = {
    "1. The problem": section_problem,
    "2. The data": section_data,
    "3. The approach": section_approach,
    "4. The discovery": section_discovery,
    "5. Pressure-testing": section_validation,
    "6. Try it yourself": section_predict,
    "7. Honest limits": section_limits,
    "8. The roadmap": section_roadmap,
}


def build_app() -> None:
    st.set_page_config(
        page_title="Player Peak Value — ML PoC",
        page_icon="⚽",
        layout="wide",
    )
    st.markdown(BRAND_CSS, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("### Sections")
        choice = st.radio("Section", list(SECTIONS.keys()), label_visibility="collapsed")
        st.markdown("---")
        st.caption("DAT0424 · Malcolm Morgan")
        st.caption("Data · Transfermarkt")
        st.caption("Model · XGBoost")
        st.caption("Test R² · 0.726")
        st.caption("CV R² · 0.707 ± 0.013")

    SECTIONS[choice]()


if __name__ == "__main__":
    build_app()
