# DAT0424 ML PoC — Presentation Brief

> **The spine.** I had a *problem* — peak market value is unpredictable and crowd-sourced.
> I did a *Y* — trained three tree-based models on 39,226 players' biographies.
> I got a *Z* — XGBoost predicts peak value with R² 0.726, validated against baselines,
> ablations, and cross-validation.

This is a non-technical pitch. The audience does not need to hear "TransformedTargetRegressor."
The audience does need to hear *what the problem is, what we tried, what we learned, and what
we'd build next.* Lead with story, drop into technical detail only when defending.

---

## Section-by-section talking points

### 1. The problem (≈ 45 sec)

Open with the hero line: *"What is a football player actually worth at their peak?"*

Then the framing: *"Transfermarkt values are crowd-sourced — they lag reality and are
sensitive to hype. Clubs, agents, and betting markets all need a defensible third-party
number. This proof of concept asks: can we predict that number from biography alone?"*

Land the three concrete use-cases (scouting, contract negotiation, squad valuation).
Close on the spine callout: **I had X → did Y → got Z.**

> 🎤 *Don't mention the model name yet. Sell the problem first.*

---

### 2. The data (≈ 60 sec)

*"39,226 players, 14 features, peak market value as the target. The first thing to notice
is how brutal the distribution is."*

Walk the histogram. Explain the 300× spread between median and top-tier — this is the
technical problem the model has to solve.

Move to the **choropleth map** — the audience will engage with this visually.

*"Following the data-viz principle that ponderated data is better, we don't show player
counts — that would just map population. We show median peak value per country with at
least 50 players. This isolates talent density from population size. Brazil, France,
Spain, Argentina lead — no surprises, but now we can defend it."*

Position box plot → *"attackers skew higher, but every position has a €100M outlier."*

Age scatter → flag the limitation honestly: *"This is current age, not age-at-peak — a known
issue we'll address in the Roadmap."*

> 🎤 *Volunteer the age-feature caveat here. It earns trust.*

---

### 3. The approach (≈ 45 sec)

This is the section that turns "magic from nowhere" into a defendable methodology.

*"The target is brutal. A €0.5M error on Mbappé's €180M peak is fine; the same €0.5M error
on a €500k median player is a 100% error. We solved this by training every model on the
log-transformed target — suddenly the loss function cares equally about getting a €1M
player right and a €100M player right."*

Then the three-model story:
- **Decision Tree:** the simplest answer, our baseline floor.
- **Random Forest:** ask 140 trees and vote, averaging cancels errors.
- **XGBoost:** 350 small trees that each correct the previous one's mistakes.

*"Why trees and not deep learning? 39k rows is small for neural nets. Features are
tabular. Categoricals encoded as integers — linear models would treat that as ordinal,
which is meaningless. Trees split on values directly. XGBoost is the right tool for
this size and shape."*

> 🎤 *This section is the antidote to "magic from nowhere". Don't rush it.*

---

### 4. The discovery (≈ 75 sec)

The headline first: *"XGBoost wins — test R² 0.726, MAE €1.79M."*

Then **prove it three ways:**

1. **Vs. baselines.** *"How do we know 0.73 is good? Compared to what? Predict-median:
   R² −0.08, MAE €3.33M. Linear regression: catastrophic at R² −755 — useful evidence
   that non-linear is required. XGBoost cuts trivial-baseline MAE by 46%."*

2. **Stability.** *"Could 0.726 be a lucky split? 5-fold CV puts XGBoost at 0.707 ± 0.013.
   Our test 0.726 sits inside that band. Not luck."*

3. **What did the model learn?** Walk the feature importance chart. *"The model figured
   out that how much you've played, at what level, in what position — those are the
   strongest signals."*

> 🎤 *If asked about overfit, the train/test gap is +0.017 — no real overfit. Inside the
> expander if anyone wants it.*

---

### 5. Pressure-testing (≈ 60 sec)

This is the "how can this be wrong, and did we test for it" section.

1. **Residual scatter:** *"Each dot is a player. Perfect model puts dots on the diagonal.
   We're well-calibrated up to €20M — above that, the long-tail superstars are
   systematically under-predicted. The model is conservative when it should extrapolate."*

2. **Per-position MAE:** *"Absolute MAE scales with position value — forwards cost more,
   so we miss by more in absolute €. But proportional error stays roughly flat across
   positions. No positional bias."*

3. **Leakage ablation:** *"We flagged `current_club_competition_id` as a possible leak.
   We tested it: dropping the feature costs only −0.020 R² and +€0.15M MAE. The signal
   is real — not just league-prestige memorisation."*

> 🎤 *This is your strongest defence against "isn't this just leakage?" — volunteer
> the ablation result before they ask.*

---

### 6. Try it yourself (≈ 60 sec, interactive)

Open the demo. *"Pick a TEST-set player — one the model has never seen. The model gives
its prediction, we compare to the real peak."*

Use the random TEST player button. Show one or two interesting cases:
- A famous player where the model is close
- A long-tail superstar where the model under-predicts (defendable — we just explained why)

*"Notice the TRAIN/TEST tag. Predictions on TRAIN players are not generalisation — the
model has seen them before. Toggle off if you want to verify."*

> 🎤 *If Basile asks for live code, scroll to `src/app.py` lines 360-430 — the predict
> section. The model loads via `joblib.load`, predicts via `model.predict(X_test_row)`,
> the TransformedTargetRegressor handles the log inverse automatically.*

---

### 7. Honest limits (≈ 45 sec)

*"Three real limitations we own up to — volunteering these is more credible than hiding them."*

1. Retrospective valuation, not forward prediction (feature/target temporal overlap).
2. `age` is current age, not age-at-peak.
3. Random train/test split, not temporal.

> 🎤 *Land each in one sentence. Don't dwell. The expander has the Defence FAQ if
> anyone digs.*

---

### 8. The roadmap (≈ 60 sec)

This is where the teacher's "theoretically provable, not-yet-achieved" rule pays off.

*"If this PoC became a v1, here's what changes. Each one is testable today, none needs
new data, all extend the existing pipeline."*

1. **Age-at-peak reframing** — kills limitations 1 and 2.
2. **Temporal train/test split** — kills limitation 3.
3. **Quantile regression** — confidence intervals, not point estimates.
4. **SHAP** — per-player explanations, turns the model into a negotiation tool.
5. **Temporal performance features** — replace cumulative totals with trajectory.

*"The roadmap is the proof that this isn't a one-shot model — it's the foundation of a
real product."*

> 🎤 *Closing line: "It's an honest proof of concept. The headline metric holds up under
> baselines, cross-validation, and ablation. The limitations we have are about reframing
> the problem, not about whether the modelling worked. And every one of them has a
> testable next step."*

---

## Likely questions, with prepared answers

### Q1. *"Isn't this just leakage? Career stats predicting a career peak — same career."*
Yes, it's a retrospective valuation, not forward prediction. We disclose this in Section 7
and the Roadmap addresses it directly (item 1). The ablation on the most-suspect feature
shows the model isn't pure memorisation — it loses only 0.02 R² without it.

### Q2. *"How do you know R² 0.73 is good?"*
Three answers. (a) Predict-median: R² −0.08. (b) Linear regression: R² −755. (c) MAE drops
from €3.33M (median predictor) to €1.79M — 46% reduction. CV puts XGBoost at 0.707 ± 0.013,
so the 0.726 is real, not lucky.

### Q3. *"Is the model overfit?"*
XGBoost: train R² 0.743, test R² 0.726, gap **+0.017**. Effectively none. Random Forest
overfits more (+0.168) but still beats Decision Tree on test.

### Q4. *"Where does the model fail?"*
Long right tail. Players above €50M are under-predicted (individual deals dominate up there).
Per-position MAE scales with value but stays flat as % of median — no positional bias.

### Q5. *"Why not deep learning / a linear model?"*
39k rows is small for NNs. Features tabular with no structure. Linear regression on
label-encoded categoricals collapses (R² −755 — we showed it). Trees handle the data shape
natively. XGBoost is right-sized.

### Q6. *"Can you show me where the prediction logic lives in the code?"*
`src/app.py` Section "Try it yourself" — `model = joblib.load(...)` then
`model.predict(X_test_row)`. The `TransformedTargetRegressor` wrapping each model handles
the log-inverse transform automatically — predictions come back in raw € directly.

### Q7. *"What would you do differently with more time?"*
Roadmap (Section 8) — five items, each testable. The age-at-peak reframe is the biggest
unlock; the SHAP layer is the most user-facing.

---

## Numbers you must remember cold

| Thing | Value |
|---|---|
| Players | **39,226** |
| Features | **14** (5 categorical + 9 numeric) |
| Train / test split | 31,380 / 7,846 (80/20, seed 42) |
| XGB test R² | **0.726** |
| XGB CV R² | **0.707 ± 0.013** |
| XGB train R² | 0.743 (gap +0.017) |
| XGB MAE | **€1.79M** |
| Predict-median baseline | R² −0.08, MAE €3.33M |
| Leakage ablation Δ | −0.020 R², +€0.15M MAE |
| MAE improvement vs trivial | **46%** |

---

## Things to NOT do

- ❌ Don't open with "R² is 0.73." Open with the **problem**.
- ❌ Don't say "I used TransformedTargetRegressor." Say "we log-transformed the target."
- ❌ Don't apologise for limitations. **Volunteer** them — it's credibility.
- ❌ Don't say "we ran out of time." Say *"the next iteration is..."*
- ❌ Don't read off the screen. The Streamlit app is your support, not your script.
- ❌ Don't skip the Roadmap section — the teacher explicitly said theoretical-but-defensible
   future work counts as PoC.

---

*Rehearse cold once → recognise where the story drags. Rehearse timed once → trim. After
two runs you're ready. Total presentation length should sit around 6–8 minutes; the
Q&A is where the validation work pays off.*
