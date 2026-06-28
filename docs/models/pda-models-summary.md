# PDA Decision Models — Summary

**What this is:** a high-level summary of the three machine-learning models built on `data/pda.db` to
test whether FEMA Preliminary Damage Assessment (PDA) declaration decisions can be predicted — and, in
particular, **how much political alignment with the sitting president influences the decision, net of
disaster need and severity.** Each section explains what the model does, why it was chosen, and the
pipeline that produces it. Charts of the key performance indicators and outcomes follow inline.

Full design rationale: `docs/superpowers/specs/2026-06-28-pda-decision-prediction-models-design.md`.
Reproduce everything with `scripts/run_all_models.py`; regenerate the charts with
`scripts/plot_model_results.py`.

---

## Overview

- **Data:** the full population of 1,378 FEMA PDA reports (2007–2026). The modeling set is the
  **1,279 initial decisions — 1,177 Declared vs. 102 Denied (~8% denial rate)**; the 99 appeal-stage
  denials are excluded so a denied-then-appealed disaster isn't counted twice.
- **Target:** a binary, request-level decision — *Declared* vs. initial *Denied*. Only request-time
  information is used; every post-decision field (e.g. `disaster_number`, `denial_reason`) is excluded
  as leakage.
- **Why three models:** they triangulate the same question from different angles and at different
  levels of credibility — an interpretable **effect size** (Model 1), a flexible **predictive**
  test (Model 2), and a sharper **county-level** political test (Model 3). A finding that holds across
  all three is far more defensible than any single model's result.

### The headline finding

**A PDA denial is highly predictable — but from disaster need and severity, not from politics.**

The models predict denial well (ROC-AUC ≈ 0.92–0.93), yet adding the political features buys
essentially nothing: the predictive lift from state partisan alignment is ~0, the lift from
county-level political composition is slightly negative, and Model 1's political odds ratios are not
robust once a conservative estimator is used. The raw "politically-aligned jurisdictions are denied
less" gap largely **does not survive controls for need**.

![Predictive performance](figures/fig1_performance.png)

![Political signal adds ~no predictive lift](figures/fig2_political_lift.png)

---

## Model 1 — Hierarchical logistic regression *(the effect size)*

**What it does.** An interpretable logistic regression that estimates the odds of denial as a function
of need, request, and political features, and reports a coefficient — an **odds ratio with a
confidence interval** — for each political variable *net of* the need controls. This is the model that
produces a quotable "politics shifts the odds of denial by X" number.

**Why it was selected.** It is the most *interpretable* and *defensible* of the three. To keep the
political effect from being a regional or era artifact (party alignment is tangled up with geography
and with which administration is in office), it uses a **hierarchical** structure — random intercepts
for state and for year — so the political effects are identified from variation *within* states and
years. It is fit only on the **50 states + DC**, where partisan alignment is well-defined; sovereign
tribes and territories are excluded from this estimand and handled separately.

**Pipeline.**
1. Assemble the feature matrix; restrict to states + DC; neutralize DC's (nonexistent) gubernatorial
   alignment via an applicability flag.
2. Drop the ~84%-null IA-demographic block; add missingness indicators for moderately-null cost
   fields; standardize continuous predictors (per–standard-deviation) for numerical stability.
3. Fit a Bayesian mixed-effects logit (`BinomialBayesMixedGLM`, variational) with state + year random
   intercepts → read off odds ratios and 95% intervals.
4. **Cross-check** every political odds ratio against a conservative pooled logistic regression, since
   the variational intervals are known to be too narrow.

**Outcome.** The political odds ratios are *suggestive but not robust*. The variational-Bayes intervals
look tight, but the conservative pooled cross-check widens them so that **almost every political
effect's interval crosses OR = 1** — only `pres_margin_dispersion` is bounded away from 1 under both
estimators. Governor alignment points the intuitive way (OR ≈ 0.76 — aligned governors denied less),
and "share of affected counties the president won" points toward stronghold favoritism (OR ≈ 0.52),
but neither survives the conservative check. (Estimand: 1,163 state/DC reports, 87 denials — a small
positive class that fundamentally limits power.)

![Model 1 forest plot](figures/fig3_m1_forest.png)

---

## Model 2 — Gradient-boosted trees + ablation + SHAP *(predictability & lift)*

**What it does.** A flexible, nonlinear classifier (`HistGradientBoostingClassifier`) that answers two
questions about the *same* trained model: **(a) how well can denial be predicted at all**, and
**(b) how much does the political block contribute** — measured two complementary ways: an **ablation**
(train with vs. without the political features and compare held-out accuracy) and **SHAP** attribution
(how large is each feature's contribution).

**Why it was selected.** Where Model 1 is interpretable-but-linear, Model 2 is the *predictive* workhorse:
it natively handles missing values, captures nonlinearities and interactions a logit would miss (e.g.
"politics only matters when damage is borderline"), and its ablation gives a clean, assumption-light
answer to "does politics add predictive value?" It evaluates on **stratified cross-validation grouped
by state** (a state never spans train and test) and reports **PR-AUC** alongside ROC-AUC because the
positive class is only ~8%.

**Pipeline.**
1. Assemble; slice to Model 2's feature set — need/severity, request, election-timing, jurisdiction,
   and the **state-level** political block (the county block is Model 3's job).
2. Ordinal-encode the categorical columns; feed the gradient booster (class-weighted for the 8%
   imbalance).
3. **Ablation:** score the full model vs. a no-political-block model on out-of-fold PR-AUC, with a
   paired bootstrap confidence interval on the difference.
4. **SHAP:** rank every feature's mean contribution on the full model.

**Outcome.** Denial is **highly predictable — ROC-AUC ≈ 0.92, PR-AUC ≈ 0.73** (vs. an 8% baseline) —
but the **state partisan-alignment block adds ~zero predictive lift: ΔPR-AUC = +0.003 (95% CI
−0.021…0.030)**, an interval centered on zero. SHAP tells the same story: the top features are all
need and severity (`total_cost_estimate`, `pa_statewide_per_capita`, cost and per-capita indicators);
the first partisan feature (`state_margin`) ranks ~10th. *(Note: the ablation precisely measures the
lift from state **partisan alignment**, holding election-timing and competitiveness fixed, since those
sit in both arms.)*

![Model 2 feature importance](figures/fig4_feature_importance.png)

---

## Model 3 — County-composition block + M2→M3 ablation *(the sharpest political test)*

**What it does.** Reuses Model 2's engine but **adds a block of county-composition political features**
— aggregations over each disaster's affected counties that ask whether the president favors disasters
hitting his *strongholds*. Because the only difference from Model 2 is this block, **the M2→M3
comparison is itself the ablation** that isolates the incremental value of fine-grained county
political detail beyond state-level alignment.

**Why it was selected.** It dissociates *state* alignment from *local* alignment — a disaster can hit
deep-red counties inside a blue state — and it separates two distinct sub-hypotheses: "**one** county
heavily favored the president" (max county margin) vs. "**most** counties favored him" (share of
counties won). This is the most specific test of political favoritism in the data.

**Pipeline.**
1. Engineer per-disaster county aggregates from `report_counties` (share of counties the president won,
   max margin, damage-weighted mean margin, dispersion, etc.).
2. Assemble Model 3's feature set = Model 2's political feature set **plus** the county block.
3. Run the **same gradient-boosting engine on the same rows and folds** as Model 2 → compare M2 vs. M3
   out-of-fold PR-AUC with a paired bootstrap CI.
4. Compare the SHAP importance of "one strong county" vs. "most counties" to see which form of
   favoritism, if any, the data supports.

**Outcome.** Fine-grained county political composition **adds no predictive value** — going from M2 to
M3 *lowers* PR-AUC (0.733 → 0.674; **county lift ΔPR-AUC = −0.060, 95% CI −0.126…0.006**), the classic
signature of adding noise rather than signal. Within the (weak) county signal, "**one strongly-favoring
county**" outweighs "**most counties favored him**" by ~3× (`max_pres_margin_affected` ≈ 0.059 vs.
`share_affected_counties_pres_won` ≈ 0.018) — but against a negative overall lift, even that is faint.

![Model 3 stronghold comparison](figures/fig5_stronghold.png)

---

## Cross-model conclusion

| Model | Question | Result |
| --- | --- | --- |
| **M1** — hierarchical logit | How large is the political effect, net of need? | Suggestive but **not robust** — conservative CIs cross OR=1 for nearly all political features |
| **M2** — gradient boosting | Does state partisan alignment improve prediction? | **No** — ΔPR-AUC ≈ 0 (CI spans zero); need/severity dominate |
| **M3** — + county composition | Does local stronghold composition help? | **No** — county lift is negative; "one strong county" > "most counties" but weak |

Across three methods and two levels of analysis, the conclusion is consistent: **political alignment
has little *robust* effect on the PDA decision once disaster need and severity are accounted for.**
Decisions are well-explained by cost and per-capita impact, not by who the requesting state or county
voted for. The raw association (aligned jurisdictions denied less) is real in the unadjusted data but
largely dissolves under controls.

**Caveats (why this is a careful "little robust effect," not a hard "zero").**
- **Low power:** only ~87–102 denials, so confidence intervals are wide and a modest true effect could
  go undetected.
- **Conservative-vs-optimistic inference:** Model 1's variational intervals are anti-conservative; the
  pooled cross-check is the honest read and it is the one that erases significance.
- **Data limits:** the IA-demographic (vulnerability) block is ~84% null, so community-vulnerability
  controls are weak; effects are measured "net of *observed* need," not strict causation.
- **Scope:** sovereign tribes and territories are handled as a separate jurisdiction class (and excluded
  from Model 1's political estimand), because partisan alignment is undefined for them — tribes' notably
  higher denial rate is a *jurisdiction-status* effect, not a partisan one.
- **Reporting nuances:** SHAP importances are computed on the full corpus (N ≈ 1,279) while the
  ablations run on N ≈ 1,236 (rows lacking a state grouping key are dropped from grouped CV); the
  bootstrap resamples rows i.i.d. though CV is grouped by state.

---

*Charts generated by `scripts/plot_model_results.py` from `data/pda.db`; deterministic (`random_state=0`).*
