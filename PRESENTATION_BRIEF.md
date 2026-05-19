# DAT0424 ML PoC — Presentation Brief

> **The spine.** I had a *problem* — peak market value is unpredictable and crowd-sourced.
> I did a *Y* — engineered 34 features and tested 7 model variants on 39,226 players.
> I got a *Z* — CatBoost predicts peak value with R² 0.798, validated against baselines,
> ablations, and cross-validation. CV R² 0.777 ± 0.005 — extremely tight.

This is a non-technical pitch. The audience does not need to hear "CatBoostRegressor with
ordered target statistics." The audience does need to hear *what the problem is, what we
tried, what we learned, and what we'd build next.* Lead with story; drop into technical
detail only when defending.

---

## Section-by-section talking points

### 1. The problem (≈ 45 sec)
Open with the hero line: *"What is a football player actually worth at their peak?"*

Frame: *"Transfermarkt values are crowd-sourced — they lag reality and are sensitive to
hype. Clubs, agents, and betting markets all need a defensible third-party number. This
proof of concept asks: can we predict that number from biography alone?"*

Land the three concrete use-cases (scouting, contract negotiation, squad valuation).
Close on the spine callout: **I had X → did Y → got Z.**

> 🎤 *Don't mention the model name yet. Sell the problem first.*

---

### 2. The data (≈ 60 sec)
*"39,226 players, 34 features, peak market value as the target. The first thing to notice
is how brutal the distribution is."*

Walk the histogram. Explain the 300× spread between median and top-tier — this is the
technical problem the model has to solve.

Move to the **choropleth map** — visually engaging for non-technical audience.

*"Following the data-viz principle that ponderated data is better, we don't show player
counts — that would just map population. We show median peak value per country with at
least 50 players. This isolates talent density from population size. Brazil, France,
Spain, Argentina lead — defensible."*

Position box plot → *"attackers skew higher, every position has a €100M outlier."*

Age scatter → flag the limitation honestly: *"current age, not age-at-peak — a known
issue we'll address in the Roadmap."*

> 🎤 *Volunteer the age-feature caveat here. It earns trust.*

---

### 3. The approach (≈ 60 sec)
This is the antidote to "magic from nowhere."

*"The target is brutal. A €0.5M error on Mbappé's €180M peak is fine; the same €0.5M on
a €500k median player is a 100% error. We solved this by training every model on the
log-transformed target — the loss function now cares equally about getting a €1M player
right and a €100M player right."*

Then the four-model story (highlight the 3D carousel):
- **Decision Tree** (R² 0.643) — simplest possible answer, baseline floor.
- **Random Forest** (R² 0.716) — 140 trees voting, ensemble averaging.
- **XGBoost** (R² 0.759) — gradient boosting, but categoricals are label-encoded ints.
- **CatBoost** (R² **0.798**) — *native* categorical handling, no label-encoding hack.
  Ordered boosting prevents target leakage.

*"Same data, same features. CatBoost wins because it handles the 6 categorical features
properly — XGBoost treats them as ordinal integers, which is meaningless when the
encoding is arbitrary."*

> 🎤 *This is the technical insight. Land it slowly.*

---

### 4. The discovery (≈ 75 sec)
Headline: *"CatBoost wins — test R² 0.798, MAE €1.53M."*

Then **prove it three ways:**

1. **Vs. baselines.** *"How do we know 0.80 is good? Predict-median: R² −0.08, MAE
   €3.33M. Linear regression: R² 0.12. CatBoost cuts trivial-baseline MAE by **54%**."*
2. **Stability.** *"Could 0.798 be a lucky split? 5-fold CV puts CatBoost at
   **0.777 ± 0.005** — incredibly tight. Test 0.798 sits inside that band."*
3. **What did the model learn?** Feature importance chart. *"Top drivers: minutes played,
   top-tier-share, height, foreigners pct of club. How much you've played, at what tier,
   at what club."*

> 🎤 *If asked about overfit: train R² 0.805, test 0.798, gap **+0.007**. Inside the expander.*

---

### 5. Pressure-testing (≈ 75 sec)
"How could this be wrong, and did we test for it?"

1. **Residual scatter:** *"Each dot is a player. Perfect model puts dots on the diagonal.
   Well-calibrated up to €20M; above that, long-tail superstars are systematically
   under-predicted. The model is conservative when it should extrapolate."*

2. **Per-position MAE:** *"Absolute MAE scales with position value — forwards cost more.
   But proportional error stays roughly flat across positions. No positional bias."*

3. **Leakage ablation:** *"We flagged `current_club_competition_id` as a possible leak.
   We tested it: dropping the feature costs only **−0.002 R²** and +€0.05M MAE — the new
   features absorbed its signal. The model is real, not league-prestige memorisation."*

4. **NEW: Prediction error by price bucket.** *"R² in € is dominated by getting the
   order-of-magnitude right. The user-felt metric — median % error per bucket — tells a
   different story: the model is best in the €1–20M band where it has most training data.
   It's biased toward the mean: cheap players over-predicted, elite players under-predicted.
   This is why a quantile regression v2 is the right next step."*

> 🎤 *This is your strongest defence against "isn't this just leakage?" — volunteer
> the ablation result before they ask.*

---

### 6. Try it yourself (≈ 60 sec, interactive)
Open the demo. *"Pick a TEST-set player — one the model has never seen. The model gives
its prediction, we compare to the real peak."*

Use the random TEST player button. Show a famous player + a long-tail outlier.

*"TRAIN/TEST tag matters. Predictions on TRAIN players are not generalisation."*

> 🎤 *If Basile asks for live code: `src/app.py` Section 6 — `model = joblib.load(...)`
> then `model.predict(X_test_row)`. The CatBoostLogWrapper handles log-inverse automatically.*

---

### 7. Honest limits (≈ 45 sec)
*"Three real limitations we own up to."*

1. Retrospective valuation, not forward prediction (feature/target temporal overlap).
2. `age` is current age, not age-at-peak.
3. Random train/test split, not temporal.

> 🎤 *Land each in one sentence. Defence FAQ has the depth if asked.*

---

### 8. The roadmap (≈ 60 sec)
*"If this PoC became a v1, here's what changes. Each one is testable today, none needs
new data, all extend the existing pipeline."*

1. **Age-at-peak reframing** — kills limitations 1 and 2.
2. **Temporal train/test split** — kills limitation 3.
3. **Quantile regression** — confidence intervals, not point estimates. **This is the
   direct fix for the regression-to-the-mean bias we showed in Section 5.**
4. **SHAP** — per-player explanations.
5. **Trajectory features** — replace cumulative totals with rate-of-change.

> 🎤 *Closing line: "An honest proof of concept. The headline metric holds up under
> baselines, cross-validation, ablation. Every limitation has a testable next step."*

---

## Likely questions, with prepared answers

### Q1. *"Isn't this just leakage?"*
Yes, it's a retrospective valuation, not forward prediction. We disclose this in Section 7.
The ablation on the most-suspect feature (`current_club_competition_id`) shows the model
loses **only 0.002 R²** without it — feature absorbed by v2 enhancements (top-tier minutes,
club quality). Not pure memorisation.

### Q2. *"How do you know R² 0.80 is good?"*
- **Predict-median:** R² −0.08, MAE €3.33M
- **Linear regression:** R² 0.12, MAE €2.46M
- **CatBoost:** R² 0.798, MAE €1.53M — **54% MAE reduction over trivial baseline.**
- **CV R² 0.777 ± 0.005** — extremely tight.

### Q3. *"Is the model overfit?"*
CatBoost: train 0.805 / test 0.798 / gap **+0.007**. Essentially zero (ordered boosting).
RF overfits more (+0.161). XGB sits in between (+0.028).

### Q4. *"Where does the model fail?"*
Long right tail. Players above €50M are systematically under-predicted because the model
is mean-biased — a known issue we surface in Section 5's bucket diagnostic. Quantile
regression (Roadmap #3) is the fix.

### Q5. *"Why CatBoost over XGBoost?"*
XGBoost requires label-encoding the 6 categorical features into integers. It then treats
them as ordinal numbers — which is meaningless (France=12 vs Italy=15 has no order).
CatBoost handles categoricals natively with ordered target statistics and built-in leakage
protection. Same data + features: **XGBoost 0.759 vs CatBoost 0.798.**

### Q6. *"Can you show me where the prediction logic lives in the code?"*
`src/app.py` Section "Try it yourself" — `model = joblib.load(...)` then
`model.predict(X_test_row)`. The CatBoostLogWrapper class in `src/model_io.py` handles
the log-inverse transform automatically.

### Q7. *"What would you do differently with more time?"*
The Roadmap — five testable items. **Quantile regression** is the highest-priority item;
it directly fixes the regression-to-the-mean we showed.

---

## Numbers you must remember cold

| Thing | Value |
|---|---|
| Players | **39,226** |
| Features | **34** (6 categorical + 28 numeric) |
| Train / test | 31,380 / 7,846 (80/20, seed 42) |
| CatBoost test R² | **0.798** |
| CatBoost CV R² | **0.777 ± 0.005** |
| CatBoost train R² | 0.805 (gap **+0.007**) |
| CatBoost MAE | **€1.53M** |
| Predict-median baseline | R² −0.08, MAE €3.33M |
| MAE improvement vs trivial | **54%** |
| Leakage ablation Δ | −0.002 R², +€0.05M MAE |
| v1→v2 R² improvement | 0.726 → 0.798 (+9.9%) |

---

## Things to NOT do
- ❌ Don't open with "R² is 0.80." Open with the **problem**.
- ❌ Don't say "I used CatBoostLogWrapper." Say "we trained CatBoost with log-transformed target."
- ❌ Don't apologise for limitations. **Volunteer** them.
- ❌ Don't say "we ran out of time." Say *"the next iteration is..."*
- ❌ Don't read off the screen. The Streamlit app is your support, not your script.
- ❌ Don't skip the Roadmap — theoretical-but-defensible future work counts as PoC.

---

*Rehearse cold once → recognise where the story drags. Rehearse timed once → trim.
After two runs you're ready. Total presentation length should sit around 7–9 minutes.*
