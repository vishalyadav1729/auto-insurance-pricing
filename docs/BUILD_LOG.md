# Build Log

A phase-by-phase record of how this project was built, including every bug found and fixed
along the way and the reasoning behind each modelling decision. `README.md` gives the
polished summary; this document is the detailed, chronological account for anyone who wants
the full story — the data-science equivalent of commit history plus decision rationale.

**Phase 1 complete:** reproducible environment, package skeleton, and import test.

**Phase 2 complete:** freMTPL2 frequency (678,013 policies) and severity (26,639 claims)
tables downloaded via `scripts/download_data.py`, profiled and reconciled in
`notebooks/01_data_understanding.ipynb`, with findings documented in
`reports/data_dictionary.md`. Six anomalies were identified (out-of-range exposure,
frequency/severity claim-count mismatches, orphaned policies, a long severity tail,
a fuel-type encoding artifact, and an implausible vehicle age) — none corrected yet;
that is Phase 3.

**Phase 3 complete:** cleaning policy decided and documented (`reports/cleaning_policy.md`,
0 policies removed), implemented in `src/auto_pricing/features.py` and
`scripts/prepare_data.py`, and explored in `notebooks/02_data_cleaning_eda.ipynb`
(exposure/claim-count distributions, frequency and severity by 9 rating factors,
large-loss concentration, and pure premium). Headline findings: portfolio frequency is
0.1006 claims/policy-year and pure premium is €167.18/policy-year (exposure-weighted;
naive per-policy averaging overstates both by 2.3-2.6x). `BonusMalus`, `DrivAge`, and
`VehAge` are strong but non-linear frequency predictors; no rating factor showed a
convincing effect on severity. Frequency × mean severity overstates true pure premium
by exactly 36.3%, a quantified consequence of `ClaimNb` counting some claims with no
matching payment (see cleaning_policy.md rule 2b) — carried into Phase 4/7 as a
documented limitation to address explicitly, not silently.

**Phase 4 complete:** policy-level 70/15/15 train/validation/test split
(`src/auto_pricing/split.py`, `scripts/split_data.py`), stratified by claim
occurrence — claim rate matches to 4 decimal places across all three splits
(5.0235%). Feature engineering (`src/auto_pricing/features.py`,
`scripts/build_model_table.py`) applies Phase 3's findings: fixed bands for
`DrivAge`/`VehAge`/`BonusMalus`, a log transform for `Density`, an ordinal
encoding for `Area`, and rare-category grouping for `Region`/`VehBrand` —
fit strictly on the training split (threshold: 2,000 training policy-years,
chosen from a real gap in the data), grouping exactly the low-credibility
segments Phase 3 flagged by name (`Region R21/R42/R43/R74/R83/R94`,
`VehBrand B14`) into `"Other"`. Verified applied identically across all three
splits. Output: `data/processed/model_table.parquet` (678,013 rows, both raw
and engineered columns — deliberately not one-hot encoded, since Phase 5's
GLMs and Phase 8's boosting models need categoricals in different shapes)
and the fitted category-grouping rule persisted to
`artifacts/preprocessors/rare_category_maps.joblib`. Full column-by-column
spec in `reports/feature_dictionary.md`.

**Phase 5 complete:** claim-frequency modelling (`src/auto_pricing/frequency.py`,
`src/auto_pricing/evaluation.py`, `notebooks/03_frequency_glm.ipynb`). Fit a baseline,
Poisson GLM, a regularization sweep, and a Negative Binomial challenger (`alpha=0.8115`,
p≈1.1×10⁻⁶⁰ — confirming material overdispersion, Pearson ratio 2.31). **Champion: Poisson
GLM (unregularized)** — ties Negative Binomial on validation deviance (0.594235 vs 0.594267;
NB's real benefit is calibrated standard errors, not different point predictions) and
comfortably beats a regularization sweep that made validation deviance monotonically *worse*
at every strength tested, despite dramatically changing specific thin-segment coefficients
(explained: blanket regularization degrades large well-supported segments to fix small ones).
Validation calibration: observed-to-expected ratio 0.9998; good calibration across all 10
risk deciles (5% to 33%). Two real bugs found and fixed while building this: statsmodels'
`.predict()` silently drops the exposure offset (confirmed across three different result
classes), and naive persistence produced 878MB model files, fixed with a 13KB lightweight
artifact format. Full relativity tables and interpretation — including a likelihood-ratio
test showing `Area`/`Density`/`Region` are jointly significant (p≈6.2×10⁻³³) despite looking
individually weak due to multicollinearity — in `reports/frequency_relativities.md`; champion
selection rationale in `reports/frequency_model_selection.md`.

**Phase 6 complete:** claim-severity modelling (`src/auto_pricing/severity.py`,
`notebooks/04_severity_glm.ipynb`). Fit a baseline, Gamma GLM, and a lognormal alternative
(Duan smearing-corrected — naive log-scale exponentiation understated true cost by 60% on
this data, a documented bias now guarded against in code). Large-loss sensitivity analysis
(deferred from Phase 3) found the full Gamma model's `DrivAgeBand` severity effect was almost
entirely an artifact of the top 1% of claims (41% of value). On validation, the full Gamma
GLM surprisingly *underperformed* the trivial baseline (overfitting traced to `Region`/
`VehBrand`, 26 of 48 parameters) while the same features *helped* the lognormal model —
explained by Gamma's raw-scale sensitivity to large claims vs. lognormal's log-scale
resistance to them. **Champion: Lognormal (smearing-corrected, full formula)** — best
validation deviance, most resilient calibration, and (checked, not assumed) the most stable
relativities under the large-loss sensitivity test. Interpretation, using heteroscedasticity-
robust standard errors (checked via Jarque-Bera/Breusch-Pagan before trusting OLS's own):
`BonusMalus` is a genuine, robust severity driver (up to 1.66× at the worst score, confirmed
stable under the sensitivity check) — but `VehAge`, `VehPower`, `VehGas`, `Area`, and
`Density`, all significant for frequency, show **no relationship to severity at all**. Full
detail in `reports/severity_model_selection.md` and `reports/severity_relativities.md`.

**Phase 7 complete:** pure premium (`src/auto_pricing/pure_premium.py`,
`notebooks/05_pure_premium_evaluation.ipynb`). Resolved the 36.3% frequency×severity
overstatement deferred since Phase 3 — not with a flat correction factor (rejected: the
"reported claims that get paid" rate varies from 0.65–1.00 by BonusMalus and 0.61–0.89 by
DrivAge), but with a second, purpose-built frequency model targeting paid claims directly,
reusing Phase 5's exact formula. Reconciliation: exact for frequency alone (ratio 1.0000),
0.9931 combined with severity. Compared against a direct Tweedie GLM (power=1.5, chosen via a
fair fixed-evaluation-power grid search): **champion is Frequency × Severity**, winning on
deviance (65.44 vs. 67.89), calibration, and risk-ranking (Gini 0.30 vs. 0.26) — Tweedie's
calibration reversed sharply between train (+23%) and validation (−32%), investigated but not
fully explained (ruled out Phase 6's Gamma-overfitting mechanism; disclosed as open rather
than forced). A real methodology bug was caught before publishing any Gini result: ranking by
raw expected loss instead of annualized rate gave a non-discriminating baseline a misleadingly
negative Gini (−0.31 vs. the correct −0.02) — fixed in `evaluation.py`, locked in with a
regression test. **First and only test-set check** of the already-chosen champion confirms
stability (O/E 0.843 → 0.850, no reversal). Honest limitation carried forward: portfolio-level
and ranking performance are good, but **segment-level calibration by Region is not** — even
the largest region (24,229 policies) shows 24% over-prediction, not explained by sample size
alone. Full detail across `reports/pure_premium_reconciliation.md`,
`reports/pure_premium_model_comparison.md`, and `reports/pure_premium_segment_stability.md`.

**Phase 8 complete:** machine-learning challengers (`src/auto_pricing/ml_challengers.py`,
`notebooks/06_ml_challengers.ipynb`). Poisson/Gamma-loss gradient boosting, properly tuned
(small deliberate grids scored on validation, never compared against the GLMs at library
defaults), for frequency, severity, and their combination. Real, checked results throughout —
frequency boosting won cleanly (−3.0% deviance) and *independently confirmed* the GLM's two
strongest findings (BonusMalus, VehAge) via permutation importance and partial dependence, a
different model family landing on the same conclusions. Severity boosting showed only a
marginal edge (−0.38%) and, more importantly, **interpretation revealed it fails to learn the
one severity signal (BonusMalus) independently validated as robust** — confirmed persistent
even after re-running the exact large-loss sensitivity check Phase 6 used for the GLM. Combined
pure premium: boosting won every aggregate metric (deviance, O/E 1.011 vs. GLM's 0.843, Gini
+8.9%) but showed *worse*, not better, segment-level calibration in the extreme case (`Other`
region: 2.54× vs. the GLM's 1.83× under-prediction) — checked directly, not assumed to improve
just because the aggregate numbers did. **Recommendation: champion-challenger, not a single
winner** — the GLM pipeline remains the operational champion (comparable accuracy, decisively
better governance and interpretability, no validated-signal loss), with the boosted pipeline
recommended for continued development (a GLM-corrected "boosting learns the residual" approach)
given its genuine calibration and ranking advantage. Full detail across
`reports/ml_frequency_challenger.md`, `reports/ml_severity_challenger.md`,
`reports/ml_pure_premium_comparison.md`, `reports/ml_interpretation.md`, and
`reports/ml_champion_challenger_recommendation.md`.

**Phase 9 complete:** interpretation, fairness, and model governance
(`reports/model_card.md`, `reports/fairness_analysis.md`, `reports/limitations.md`,
`notebooks/07_fairness_analysis.ipynb`). Done out of the usual order at explicit
request — the model card (step 3) was written first, then updated after fairness
analysis (step 1) and the consolidated limitations document (step 2) produced real
findings, rather than sequencing all three from scratch. New checks not run in any
earlier phase: `DrivAge` segment calibration (never checked before) shows O/E ranging
0.57–1.33, over-predicting ages 40-69 and under-predicting 70+, not explained by sample
size — the 60-69 band has six times the exposure of the best-calibrated 18-22 band yet
is far worse calibrated. Checked directly whether `Region`'s own known miscalibration
(0.46–1.83, Phase 7) reduces to age composition — it doesn't (regions with nearly
identical mean `DrivAge` show opposite O/E directions), leaving Region's problem
separately unexplained. A hard, disclosed limitation throughout: this dataset has no
demographic fields (race, gender, income), so no proxy-variable analysis for
`Region`/`Density`/`Area` is possible — stated plainly rather than implied as checked.
`reports/limitations.md` consolidates every limitation disclosed since Phase 2 (data,
modelling, evaluation, fairness, deployment) into one reference document; the model card
was updated twice to incorporate both follow-on steps' findings rather than left as
originally written.

**Phase 10, step 1 complete:** single-policy pure-premium calculator
(`app/app.py`, `app/pricing.py`). Loads the GLM champion pipeline (paid-frequency
Poisson GLM × lognormal severity GLM) from lightweight joblib artifacts and prices
one user-entered policy at a time. Two real gaps were found and fixed while building
this, both in code that already claimed to support it: `predict_from_artifact`
(frequency.py) and the new `predict_severity_from_artifact` (severity.py) both raised
`NameError` on a genuinely new policy with no known target value, because patsy's
formula parser requires the left-hand-side column to be *present* in the data even
though prediction only uses the right-hand side — confirmed empirically before fixing,
not assumed. Both now synthesize a dummy target column internally, with regression
tests added. A new `engineer_single_policy` function (`features.py`) reuses every
Phase 4 transform unchanged so the app can never drift from how the models were
actually trained; a new `save_severity_model`/`predict_severity_from_artifact` pair
(mirroring Phase 5's frequency artifact format) avoids repeating the 878MB-pickle
mistake for the severity model. The app itself is tested two ways: plain pytest for
`app/pricing.py`'s logic, and Streamlit's own `AppTest` harness (which actually
executes the script and can click its button) for the UI — confirmed during testing
that hitting a running server with plain HTTP only fetches its static JS shell and
never executes the Python script at all, so that approach would have missed a real
bug. The app surfaces Phase 9's fairness findings and the model-card disclaimer
directly in its UI, not just in the reports.

**Phase 10, step 2 complete:** model-exploration dashboard (`app/pages/model_exploration.py`).
Restructured the app into an explicit multipage router first: `app.py` now calls
`st.navigation` over `pages/pricing_calculator.py` (step 1's calculator, moved as-is)
and the new exploration page. This wasn't optional — confirmed directly while building
this that the older implicit `pages/`-directory auto-detection (no `st.navigation` call)
produces **zero** discovered pages on Streamlit 1.64, checked via `PagesManager` directly
rather than assumed from a still-working `curl` request. The exploration page's centerpiece
is a **live relativity explorer**: it sweeps one rating factor across its bands and computes
frequency/severity relativities from the actual deployed artifacts, not pasted report
tables. This surfaced a genuine, worth-disclosing finding, not a bug: the live *severity*
relativities reproduce `reports/severity_relativities.md` almost exactly (e.g. BonusMalus
130+ → 1.663× live vs. 1.66× published), but the live *frequency* relativities do **not**
match `reports/frequency_relativities.md` (e.g. BonusMalus 130+ → ~9.0× live vs. 6.43×
published) — because the app correctly prices using Phase 7's paid-frequency model
(`ClaimNbFromSev`), a different fitted model from Phase 5's reported-claims champion
(`ClaimNb`) that report documents, and the two diverge because payment rate itself varies
by BonusMalus band (0.65–1.00, Phase 7). The page states this explicitly rather than
letting a visitor assume a mismatch is an error. Also shown: the Region/DrivAge
calibration-gap tables (Phase 7/9) and the GLM-vs-boosted scorecard (Phase 8), both
already-computed findings presented for exploration, not recomputed. Tested with 10 total
app tests, including a regression check that the live severity relativity stays within 0.02
of the published value (drift detection, not just "doesn't crash").

**Phase 10, step 3 complete, Phase 10 done:** governance page
(`app/pages/governance.py`). Deliberately renders `reports/model_card.md`,
`reports/fairness_analysis.md`, and `reports/limitations.md` directly from disk in three
tabs, rather than a hand-written in-app summary — a summary would drift the next time any
of those reports changes (as `model_card.md` itself already has, twice, in Phase 9), while
reading the file at render time cannot drift, by construction. The missing-file fallback
path was pulled into its own plain function (`read_report_or_error_message`) specifically
so it could be unit-tested directly without needing Streamlit's test harness or deleting a
real project report to exercise it. A top-of-page warning restates the "not a real pricing
system" disclaimer. 13 app tests total (123 project-wide); a real headless server launch
confirmed the full three-page router starts cleanly with no errors.

The app now has three pages behind one router (`app/app.py` → `st.navigation`): the
pricing calculator (step 1), the model-exploration dashboard (step 2), and this governance
page (step 3) — none of the analytical findings live only in markdown anymore; the same
numbers and disclosures a reader would find in `reports/` are reachable from the running
app itself.
