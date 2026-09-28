# Phase 5 — Frequency Model Selection

Decision record for the champion frequency model, evaluated on the **validation split**
(never test — test stays untouched until the final phase-comparison). All four candidates
use the identical feature formula (`FREQUENCY_FORMULA`, `src/auto_pricing/frequency.py`) so
the comparison isolates the modelling choice, not the feature set.

## Candidates compared

| Candidate | What it is |
|---|---|
| Baseline | One portfolio-wide rate (0.1006) for every policy |
| Poisson GLM | `Var(N) = E(N)` assumed; log(Exposure) offset |
| Regularized Poisson (L2 sweep) | Same as above, with a shrinkage penalty on coefficients |
| Negative Binomial | Poisson's `alpha=0` special case relaxed; `alpha` estimated via MLE |

## Validation deviance (exposure-weighted, lower is better)

| Model | Validation deviance |
|---|---|
| Baseline | 0.625924 |
| **Poisson (unregularized)** | **0.594235** |
| Negative Binomial | 0.594267 |
| Regularized Poisson, α=0.0001 | 0.594603 |
| Regularized Poisson, α=0.001 | 0.601299 |
| Regularized Poisson, α=0.01 | 0.624713 |
| Regularized Poisson, α=0.1 | 0.650732 |
| Regularized Poisson, α=1.0 | 0.678315 |

Both Poisson and Negative Binomial clear the baseline by a wide margin — the rating factors
are genuinely informative, not noise. Between Poisson and Negative Binomial, the two are
effectively tied.

## Finding 1: Poisson and Negative Binomial predict almost identically

This is expected, not a null result. Exposure-weighted deviance measures how close a model's
**predicted mean** is to the observed rate — it says nothing about how much *uncertainty* the
model assigns to that mean. Poisson and Negative Binomial share the same log-link mean
structure and produce nearly identical fitted coefficients (Phase 5 step 3); Negative
Binomial's extra `alpha` parameter corrects the model's assumed *variance*, not its point
prediction. A tie on deviance is exactly what that distinction predicts: **Negative
Binomial's value is in honest inference (valid standard errors/p-values), not in different
point predictions** — precisely the distinction discussed before building it.

## Finding 2: regularization makes validation deviance monotonically *worse*

This contradicts what a coefficient-level check alone suggested. Step 3 found that a small L2
penalty dramatically changes the `BonusMalusBand[130+]` coefficient (1.86 → 0.16) — a segment
with only 364 training policy-years behind it. That looked like regularization correctly
"fixing" an unstable, thin-segment estimate. But the *portfolio-level* validation deviance
gets worse at every tested strength, with no exception:

**Why both findings are true at once:** a single global L2 penalty shrinks *every*
coefficient by a similar relative amount — not just the thin-segment ones. The
`BonusMalusBand[130+]` segment is a tiny fraction of total portfolio exposure, so even a
substantially "corrected" coefficient there barely moves the aggregate deviance. Meanwhile,
the same penalty also shrinks the coefficients for the *large*, well-supported segments that
didn't need any correction — and since those dominate total exposure, that degradation
outweighs the small benefit from fixing the one thin segment. A blanket penalty cannot tell
the difference between "this coefficient is uncertain because of a small sample" and "this
coefficient is precise because of a huge sample" — it treats both the same.

**This also validates a Phase 3/4 decision already made**: rather than rely on blanket
regularization to handle low-credibility segments after the fact, Phase 4 already pooled the
worst offenders (`Region`, `VehBrand`) into `"Other"` — a *targeted* fix for exactly the
categories that needed it, leaving the rest of the model untouched. `BonusMalusBand[130+]`
and `VehAgeBand[20+]` were deliberately *not* collapsed (Phase 3 judged them meaningful,
ordered, interpretable bands worth keeping distinct) — and this evaluation confirms that
decision holds up on real held-out data: leaving them unregularized is better for the model
as a whole, even though that specific coefficient is individually less trustworthy. That
imprecision is accepted as a documented limitation, not "fixed" at the cost of the rest of
the model.

## Champion: Poisson GLM (unregularized, full formula)

Simpler, faster, better-understood diagnostics (canonical link, exact training-total match),
and matches Negative Binomial's predictive accuracy exactly. Regularization is rejected — not
because it didn't do anything, but because what it does (uniform shrinkage) actively hurts
overall predictive performance here.

**Documented caveat carried forward, not discarded**: the overdispersion found in step 2
(Pearson ratio 2.31) means the Poisson model's own reported standard errors and p-values are
understated. Anyone using this model's inferential statistics (not just its point predictions)
should apply a correction — e.g. multiply standard errors by `sqrt(2.31) ≈ 1.52`, or refer to
the Negative Binomial model's standard errors directly, which already account for this.

## Champion's calibration on validation

- **Observed-to-expected ratio: 0.9998** — the portfolio total is essentially exactly right,
  not just close.
- **Calibration by predicted-risk decile**: observed and predicted rates track closely across
  every decile, from ~4.8-5.0% in the lowest-risk decile to ~32-33% in the highest — the model
  both ranks risk correctly (monotonically increasing) and doesn't systematically mis-price
  any specific risk band.

| Decile | Observed rate | Predicted rate |
|---|---|---|
| 0 (lowest risk) | 4.85% | 4.95% |
| 1 | 5.41% | 6.23% |
| 2 | 7.09% | 7.05% |
| 3 | 7.56% | 7.61% |
| 4 | 8.76% | 8.16% |
| 5 | 8.61% | 8.85% |
| 6 | 9.43% | 9.98% |
| 7 | 12.88% | 11.89% |
| 8 | 16.53% | 16.62% |
| 9 (highest risk) | 33.00% | 32.38% |
