# DAT0424 ML PoC — Presentation Brief

**Project:** Predicting football players' peak market value from Transfermarkt data.
**Best model:** XGBoost — test R² 0.726, 5-fold CV R² 0.707 ± 0.013, MAE €1.79M.

---

## The 3-minute narrative

1. **Problem.** Clubs, agents, and betting markets all need a number for "what's this player worth at their peak?" Transfer market values exist on Transfermarkt, but they're crowd-sourced and lagging. Can we predict the career peak from biographical + cumulative performance data?

2. **Data.** Transfermarkt: 39,226 players with non-null peak-value labels, 12 CSVs (players, appearances, clubs, games, transfers...). 14 features after engineering: 5 categorical (position, sub-position, foot, nationality, current club league) + 9 numeric (age, height, caps, international goals, career goals/assists/minutes, cards). Target: `highest_market_value_in_eur`, log1p-transformed during training.

3. **Models.** Three tree-based regressors with identical preprocessing — Decision Tree, Random Forest, XGBoost — chosen because (a) tabular data with mixed types, (b) long right tail in the target, (c) tree splits handle label-encoded categoricals natively. All three wrap a log-transform of the target so the loss is in a balanced space.

4. **Results.** XGBoost wins: **test R² 0.726, MAE €1.79M**. Cross-validated R² is **0.707 ± 0.013** — tight, not a lucky split. Trained vs test R² gap is only **+0.017**, so no meaningful overfit.

5. **vs. baselines.** Predict-median gives R² −0.08 (worse than mean — long tail). Linear regression collapses to R² −755 — proving that non-linear models are required, not just preferred. XGBoost cuts MAE from €3.33M (median predictor) to €1.79M.

6. **Where it fails.** Per-position MAE scales with position value (GK €0.94M → ATT €2.25M) but **relative** error is roughly constant — no positional bias. The model is most useful at the top of the distribution; for sub-€1M players it can be wrong by 3-4×.

7. **Limitations we own.** Three real ones: (a) features are career-cumulative but target is career-peak — *retrospective valuation*, not forward prediction; (b) `age` is current age, not age-at-peak; (c) split is random, not temporal. We tested the most obvious leakage suspect (`current_club_competition_id`) — dropping it costs only 0.02 R², so the model isn't pure league-prestige memorisation.

8. **What we'd build next.** Re-frame as *"predict peak at age 25 given features up to age 21"* — eliminates leak (a) and (b) in one move. Temporal split for (c). Quantile regression for prediction intervals. SHAP for per-player explanations.

**Closing line:** "It's an honest proof of concept: XGBoost handles this data well, the validation supports the headline number, and the limitations we have are about reframing the problem — not about whether the modelling worked."

---

## The Defence FAQ — 5 questions the teacher will probably ask

### Q1. "Isn't this just leakage? Career stats predict a career peak — same career."

**Answer:** Yes, it's a retrospective valuation model, not a forward predictor. We disclose this up-front in Limitations. We tested the most-suspect feature (`current_club_domestic_competition_id`) by re-training without it: R² drops from 0.726 to 0.706 and MAE rises by €0.15M. Material but not load-bearing. The honest re-frame is "predict peak at 25 from features at 21" — that's the next iteration.

### Q2. "How do you know R² 0.73 is good?"

**Answer:** Two baselines.
- **Predict-median:** R² = −0.08, MAE €3.33M. The trivial floor.
- **Linear regression** (with log-transformed target): R² collapses to −755 because label-encoded categoricals are treated as ordinal numbers — meaningless. This proves non-linear models are necessary, not optional.

XGBoost cuts MAE from €3.33M to €1.79M — that's a 46% reduction over the trivial baseline. And **5-fold CV** gives 0.707 ± 0.013, so the test 0.726 isn't a lucky split.

### Q3. "Is the model overfit?"

**Answer:**
- **Decision Tree:** train R² 0.648 / test 0.547 / gap +0.101 — under-fits and slightly over-fits.
- **Random Forest:** train 0.839 / test 0.671 / gap +0.168 — moderate overfit, deep trees memorising.
- **XGBoost:** train 0.743 / test 0.726 / gap **+0.017** — effectively no overfit. The regularisation (max_depth=4, subsample=0.9, reg_lambda=1.0) is doing its job.

### Q4. "Where does the model fail? Show me."

**Answer:** Two views.

**Per-position MAE on the test set:**

| Position | N | Median actual | MAE |
|---|---:|---:|---:|
| Goalkeeper | 842 | €400k | €0.94M |
| Defender | 2,536 | €800k | €1.54M |
| Midfield | 2,257 | €850k | €1.95M |
| Attack | 2,188 | €900k | €2.25M |

Absolute MAE scales with position value. Relative error (MAE / median) is ~2-2.5× across all positions, so no positional bias.

**Residual scatter** (in the app's Diagnostics section): model is well-calibrated up to ~€20M; above that the long-tail superstars are systematically under-predicted (the model is conservative when it should extrapolate).

### Q5. "Why XGBoost and not [deep learning / linear model / SVM]?"

**Answer:**
- **Linear / logistic models:** demonstrated above — R² −755. Long tail + label-encoded categoricals breaks them.
- **Deep learning:** 39k rows is small for neural nets. The features are tabular with no obvious sequential or spatial structure. An MLP would need heavy regularisation and lose the interpretability that XGBoost offers (feature importance, SHAP). Trade-off not worth it at PoC stage.
- **SVM:** doesn't scale to 39k rows efficiently in non-linear kernels; tree models do.
- **Why XGBoost specifically over Random Forest?** XGB has 2.5 points of test R² over RF (0.726 vs 0.671) at the same scale, plus less overfit (+0.017 vs +0.168 gap). Gradient boosting iteratively corrects residuals — better suited to the long-tail target than RF's averaging.

---

## Things to NOT do during the defence

- ❌ Don't open with "R² is 0.73" without context. Open with the **problem**.
- ❌ Don't claim the model predicts an academy player's value — be explicit about where it fails (sub-€1M, super-rich tail).
- ❌ Don't read the slides. The Streamlit app *is* the slides.
- ❌ Don't hide the post-peak leak. **Volunteer it** — it shows you understood your own work.
- ❌ Don't say "we ran out of time" for the temporal-split version. Say *"the next iteration is..."* — sounds like a roadmap, not an excuse.

---

## The numbers you must remember cold

| Thing | Number |
|---|---|
| Players | 39,226 |
| Features | 14 (5 cat + 9 num) |
| Train / test | 31,380 / 7,846 (80/20, seed 42) |
| XGB test R² | **0.726** |
| XGB CV R² | **0.707 ± 0.013** |
| XGB train R² | 0.743 (gap +0.017) |
| XGB MAE | **€1.79M** |
| Median-predictor R² | −0.08 |
| Median-predictor MAE | €3.33M |
| Leakage ablation Δ | −0.020 R², +€0.15M MAE |

---

*Rehearse twice cold. The first run will be too long; cut to 4 minutes. The second run will be too rushed; recover the "honest about limitations" tone. After two runs you're ready.*
