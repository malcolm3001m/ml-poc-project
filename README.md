# Predicting football players' peak market value

**DAT0424 — ML Proof of Concept — Malcolm Morgan**

The question this project asks: can we predict the **highest market value a player
will hit during their career** from biographical and performance data alone?

That number matters because transfer market values on Transfermarkt are crowd-sourced
and lag reality. Clubs, agents, and betting markets all need a defensible third-party
estimate. If the model says a 19-year-old's predicted peak is €40M and his current
price is €5M, that's an arbitrage signal worth investigating.

## Quick start

```bash
# Install dependencies (Python 3.11+)
pip install -r requirements.txt

# Run the orchestrator — evaluates all 4 models then launches Streamlit
python scripts/main.py
# Open http://localhost:8501
```

To re-run the full diagnostic suite (overfit, CV, baselines, ablation, per-position):

```bash
python scripts/validate.py
```

This writes four CSVs to `results/`:

- `validation_metrics.csv` — train/test gap, 5-fold CV, baselines
- `ablation.csv` — XGBoost ± the suspected leak feature
- `per_position_mae.csv` — error broken down by position
- `residuals.csv` — actual vs. predicted on the test set

There's also a standalone HTML version of the presentation at `presentation/` —
open `index.html` (or run `python -m http.server` inside it) for the French
slide deck used during the oral defence.

## What's in here

```
├── data/                       Transfermarkt CSVs (gitignored — 12 files)
├── models/                     4 trained .joblib models
│   ├── decision_tree.joblib
│   ├── random_forest.joblib
│   ├── xgboost_regressor.joblib
│   └── catboost_regressor.joblib   ← winner
├── notebooks/
│   └── 02_model_training.ipynb  Original training notebook (v1, 3 models)
├── presentation/                Standalone HTML version of the slide deck
├── results/                     All metric outputs (model + validation CSVs)
├── scripts/
│   ├── main.py                  Template orchestrator (evaluate → launch app)
│   ├── validate.py              Diagnostic suite
│   ├── experiment.py            Compares 7 model variants
│   └── retrain_v2.py            Retrains the 4 models on the v2 feature set
├── src/
│   ├── app.py                   8-section Streamlit presentation
│   ├── config.py                Paths + MODELS registry
│   ├── data.py                  Re-exports the active v2 feature pipeline
│   ├── data_v1.py               Original 14-feature pipeline (kept for reference)
│   ├── data_v2.py               Enhanced 34-feature pipeline (active)
│   ├── metrics.py               MAE / RMSE / R²
│   ├── model_io.py              Joblib loader + CatBoost wrapper class
│   └── results.py               CSV writer for metrics
├── PRESENTATION_BRIEF.md        Defence narrative + Q&A bank
└── README.md                    You are here
```

## How the project evolved

I started with three tree-based regressors (Decision Tree, Random Forest, XGBoost)
on a 14-feature set built from `players.csv` + a basic appearances aggregate. That
v1 hit R² 0.726 on the test set — fine, but I noticed two problems when I dug
into the predictions:

1. The model regressed heavily to the mean — cheap players over-predicted, top-tier
   players under-predicted. The R² 0.726 in raw € was hiding this.
2. XGBoost was forcing the 6 categorical features into ordinal integers via label
   encoding, which doesn't make sense (France=12, Italy=15 is meaningless).

So I built a v2 pipeline with **34 features** (per-90 stats, top-tier UEFA minutes,
career length, club quality proxies, age-squared, agent flag) and tested seven
model variants in `scripts/experiment.py`:

1. v1-XGBoost baseline
2. v2-XGBoost
3. v2-LightGBM
4. v2-CatBoost
5. v2-XGBoost + isotonic post-hoc calibration
6. v2-XGBoost-Tweedie
7. v2-Stack (XGB+Iso ⊕ CatBoost average)

**CatBoost won** on a composite score that weighed median % error, R² in log space,
spearman rank correlation, calibration ratio, and elite-tier accuracy. The main
reason: CatBoost handles categoricals natively (ordered target statistics +
built-in leakage protection) without forcing them into integers.

## Final numbers

| Metric | Decision Tree | Random Forest | XGBoost | **CatBoost** |
|---|---:|---:|---:|---:|
| Test R² | 0.643 | 0.716 | 0.759 | **0.798** |
| Test MAE | €1.99M | €1.73M | €1.63M | **€1.53M** |
| 5-fold CV R² | 0.624 ± 0.021 | 0.708 ± 0.013 | 0.754 ± 0.012 | **0.777 ± 0.005** |
| Train/test gap | +0.069 | +0.161 | +0.028 | **+0.007** |

vs. baselines:

- Predict-median: R² −0.08, MAE €3.33M
- Linear regression (log target): R² 0.12, MAE €2.46M
- **CatBoost cuts the trivial-baseline MAE by 54%**

The leakage ablation on `current_club_domestic_competition_id` (the suspected leak
feature) costs only **−0.002 R²** and +€0.05M MAE — the new v2 features absorbed
its signal almost entirely.

## What the model is honest about

The model is **retrospective**: features are career-cumulative, target is career
peak. For a retired player, the features include years of post-peak play. The
predicted peak for a player who has already declined is describing their
**historical** peak, not a future one.

The presentation app handles this honestly: if `current_value < highest_recorded × 0.85`
(declined 15%+ from peak), the demo shows *"Pic historique reconnu — déjà atteint,
pas un potentiel à capturer"* rather than a fake upside signal.

The Roadmap section of the presentation (Section 8) lists five testable v2 upgrades
that would convert this from a retrospective to a forward-looking model:

1. Reformulation à l'âge-au-pic (freeze features at age 21, predict peak)
2. Temporal train/test split (train pre-2018, test post-2018)
3. Quantile regression (prediction intervals instead of point estimates)
4. SHAP for per-player explanations
5. Trajectory features (rate-of-change instead of cumulative totals)

## Stack

Python 3.11+, pandas, scikit-learn, XGBoost, LightGBM, CatBoost, joblib,
Streamlit, plotly, streamlit-folium. The standalone HTML page uses vanilla
HTML/CSS/JS with D3 for the world choropleth and topojson-client for the
country geometry — no build step.

## Template contract (preserved from the project skeleton)

- `src/data.py` exposes `load_dataset_split() -> (X_train, X_test, y_train, y_test)`
- `src/metrics.py` exposes `compute_metrics(y_true, y_pred) -> dict[str, float]`
- `src/app.py` exposes `build_app() -> None` as the Streamlit entry point
- `src/config.py` defines the `MODELS` registry consumed by `scripts/main.py`
- `scripts/main.py` runs evaluation + launches Streamlit at `localhost:8501`

These contracts are unchanged from the upstream `basile-desjuzeur/ml-project-poc-template`.
