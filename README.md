# ⚽ Predicting Football Players' Peak Market Value

**DAT0424 — ML Proof of Concept — Malcolm Morgan**

> **The story.** Transfer market values on Transfermarkt are crowd-sourced and lag reality.
> Can we predict a player's career-peak market value from biographical + cumulative
> performance data alone?
>
> **Approach.** Three tree-based regressors (Decision Tree, Random Forest, XGBoost) trained
> on 39,226 players with 14 features. Log-transformed target to handle the long right tail.
>
> **Result.** XGBoost — **test R² 0.726, 5-fold CV R² 0.707 ± 0.013, MAE €1.79M.** Validated
> against baselines, ablations, and cross-validation. See [PRESENTATION_BRIEF.md](PRESENTATION_BRIEF.md)
> for the defended narrative.

---

## Quick start (for live verification)

```bash
# 1. Activate the virtual environment
source venv/bin/activate

# 2. Run the orchestrator — evaluates all 3 models then launches Streamlit
python scripts/main.py

# 3. Open http://localhost:8501 — 8-section narrative pitch
```

To re-run the full diagnostic suite (overfit check, CV, baselines, ablation, per-position):

```bash
python scripts/validate.py
```

This writes 4 CSVs to `results/`:

- `validation_metrics.csv` — overfit, CV, baseline diagnostics
- `ablation.csv` — XGBoost ± suspected leak feature
- `per_position_mae.csv` — error breakdown by position
- `residuals.csv` — (actual, predicted, position, name) for the calibration plot

---

## Repo layout

```
├── data/                       Transfermarkt CSVs (12 files, gitignored)
├── models/                     Trained .joblib models (3 files)
│   ├── decision_tree.joblib
│   ├── random_forest.joblib
│   └── xgboost_regressor.joblib
├── notebooks/
│   └── 02_model_training.ipynb Original training notebook
├── results/                    Evaluation outputs
│   ├── model_metrics.csv       Test-set metrics per model
│   ├── validation_metrics.csv  Overfit + CV + baselines
│   ├── ablation.csv            Leakage ablation
│   ├── per_position_mae.csv    Per-position breakdown
│   └── residuals.csv           Calibration data
├── scripts/
│   ├── main.py                 Orchestrator: evaluate → launch Streamlit
│   └── validate.py             Diagnostic suite
├── src/
│   ├── app.py                  8-section Streamlit presentation
│   ├── config.py               Paths + MODELS registry
│   ├── data.py                 load_dataset_split() + feature engineering
│   ├── metrics.py              compute_metrics(): MAE, RMSE, R²
│   ├── model_io.py             Model loader (joblib/pickle)
│   └── results.py              CSV writer for metrics
├── PRESENTATION_BRIEF.md       Defended narrative + Q&A + numbers
└── README.md                   You are here
```

---

## The presentation — 8 sections

The Streamlit app at `localhost:8501` is structured as a non-technical pitch following
the **"I had X problem → did Y → got Z solution"** template:

1. **The problem** — hero hook, three use cases, the spine
2. **The data** — distribution, choropleth map (peak value by country), position + age
3. **The approach** — three models in plain English, why trees not deep learning
4. **The discovery** — XGB wins; proved 3 ways (baselines, CV, feature importance)
5. **Pressure-testing** — residuals, per-position errors, leakage ablation
6. **Try it yourself** — interactive prediction demo, TRAIN/TEST tagged
7. **Honest limits** — 3 real limitations + Defence FAQ in expander
8. **The roadmap** — theoretically-provable v2 items (age-at-peak reframe, temporal split,
   quantile regression, SHAP, trajectory features)

---

## Validation summary

| Metric | XGBoost | Random Forest | Decision Tree |
|---|---:|---:|---:|
| Test R² | **0.726** | 0.671 | 0.547 |
| Test MAE | **€1.79M** | €1.89M | €2.21M |
| CV R² (5-fold) | 0.707 ± 0.013 | 0.657 ± 0.010 | 0.570 ± 0.019 |
| Train/test gap | **+0.017** | +0.168 | +0.101 |

**Baselines:**
- Predict-median: R² −0.08, MAE €3.33M
- Linear regression: R² −755 (catastrophic — demonstrates non-linear required)
- XGBoost cuts trivial-baseline MAE by **46%**

**Leakage ablation:** dropping `current_club_domestic_competition_id` costs only **−0.020 R²**
and **+€0.15M MAE**. The signal is real, not league-prestige memorisation.

---

## Stack

- Python 3.11+
- pandas, numpy, scikit-learn
- xgboost
- streamlit + plotly
- joblib for model persistence
- (No GPU required; full training + validation runs in ~3 min on a MacBook M4.)

---

## Template contract (preserved from project skeleton)

The project follows the DAT0424 template structure:

- `src/data.py` exposes `load_dataset_split() -> (X_train, X_test, y_train, y_test)`
- `src/metrics.py` exposes `compute_metrics(y_true, y_pred) -> dict[str, float]`
- `src/app.py` exposes `build_app() -> None` as the Streamlit entry point
- `src/config.py` defines the `MODELS` registry consumed by `scripts/main.py`
- `scripts/main.py` runs evaluation + launches Streamlit at `localhost:8501`

These contracts are unchanged from the template — verifiable by diffing against the
upstream `basile-desjuzeur/ml-project-poc-template` README.
