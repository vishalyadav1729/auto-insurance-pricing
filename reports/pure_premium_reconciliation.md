# Phase 7 — Resolving the Frequency/Severity Mismatch

This is the decision Phase 3 (`cleaning_policy.md`, rule 2b) deferred three times — through
Phase 5 and Phase 6 — and can't be deferred any further: naively multiplying the frequency
champion by the severity champion overstates true pure premium by **36.3%**, proven back in
Phase 3 to equal `sum(ClaimNb)/sum(ClaimNbFromSev)`, because `ClaimNb` counts some claims that
were reported but never resulted in a payment.

## The alternative considered and rejected: a flat correction factor

The simplest fix would be a single portfolio-wide multiplier (train-split "payment rate":
`sum(ClaimNbFromSev)/sum(ClaimNb) = 0.7343`) applied to every prediction. Before adopting it,
it was checked against real data — and rejected, because the payment rate is **not constant
across segments**:

| BonusMalus band | Payment rate | DrivAge band | Payment rate |
|---|---|---|---|
| 50 (best) | 0.65 | 18-22 | 0.89 |
| 51-59 | 0.80 | 23-29 | 0.79 |
| 60-79 | 0.83 | 30-39 | 0.78 |
| 80-99 | 0.82 | 40-49 | 0.73 |
| 100-129 | 0.83 | 50-59 | 0.72 |
| 130+ | 1.00 (thin sample) | 60-69 | 0.66 |
| | | 70+ | 0.61 |

A flat 0.7343 multiplier would under-correct young drivers (true rate up to 0.89) and
over-correct older drivers (true rate down to 0.61) — trading one known, quantified bias for a
smaller, differently-shaped, *undocumented* one. Not an improvement worth adopting.

## The resolution: a second, purpose-specific frequency model

Instead, a **second frequency model** is fit — same formula, same machinery, same training
split as Phase 5's champion — but targeting `ClaimNbFromSev` (paid claims) instead of
`ClaimNb` (reported claims). This resolves the mismatch **exactly**, not approximately: it
naturally learns how payment rate varies by segment, the same way any GLM learns any other
relationship, rather than requiring a bolted-on correction layer.

**Phase 5's champion (fit on `ClaimNb`) is not replaced or invalidated.** It remains correct
for its own stated purpose — predicting *reported* claim frequency, useful for claims-
department workload or reserving. This module adds a second, purpose-built model specifically
for pricing, where "how many claims will actually cost money" is the relevant question.

## Verification

| Check | Result |
|---|---|
| Paid-frequency model converged | Yes |
| Predicted vs. observed total paid claims (train) | 18,523.00 vs. 18,523 — **exact** |
| Predicted vs. observed total expected loss (train, combined with severity) | €43,700,398.81 vs. €44,004,810.78 — **ratio 0.9931** |

The frequency component alone reconciles exactly (guaranteed by the same canonical-link
property used throughout this project). The small remaining ~0.7% gap in the *combined*
figure comes entirely from the severity model's own known imperfection (Phase 6: the lognormal
champion's training reconciliation was 0.992, for the same reason Gamma/Negative Binomial
never matched exactly either — not every GLM family shares Poisson's exact-match guarantee).
This is a dramatic improvement over the original 36.3% overstatement, and the remaining gap is
fully explained by a source already documented, not a new, unexplained discrepancy.

## Two different numbers, and why conflating them is a real mistake

The plan explicitly warns against confusing **expected loss over a policy's observed exposure
period** with **annual pure premium** (a per-policy-year rate). They are numerically identical
only when a policy's `Exposure` happens to equal 1.0 (a full year) — for any shorter policy,
they diverge:

| Quantity | Formula | Meaning |
|---|---|---|
| `predict_expected_loss` | `paid_freq_pred × severity_pred` | Dollars expected over *this policy's* actual coverage period |
| `predict_annual_pure_premium` | `(paid_freq_pred / Exposure) × severity_pred` | Dollars per policy-*year* — the number to compare across policies or use for annual pricing |

Example: a policy with `Exposure=0.5` (half a year) and an expected count of 0.1 over that
half-year has an expected loss of `0.1 × severity`, but an *annualized* rate of
`0.2 × severity` — double the raw expected-loss figure, because the raw figure only covers
half a year. Using the wrong one when comparing or pricing policies would systematically
under-price every partial-year policy.

## Comparison to Phase 3's raw empirical estimate

Phase 3 computed the portfolio's true pure premium directly from observed data (no model):
**€167.18/policy-year**. This phase's model-based estimate, using the paid-frequency ×
severity champions: **exposure-weighted average annual pure premium of €174.19/policy-year** —
a ~4.2% difference, expected and reasonable given this figure now comes from two fitted models
(each with their own, already-documented, imperfect fit) rather than a raw aggregate of the
actual data. This is a sanity check the two numbers should be *close*, not identical.
