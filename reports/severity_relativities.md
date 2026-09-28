# Phase 6 — Severity Model Interpretation

Relativity tables and plain-language interpretation for the champion severity model
(Lognormal, Duan smearing-corrected, full formula — `reports/severity_model_selection.md`).

## How to read this

Same convention as `reports/frequency_relativities.md`: a **relativity** is `exp(coefficient)`,
measuring a category against its **reference level** (1.0 means no difference). Here it means
"claims from this group cost X times as much as claims from the reference group, given a claim
happened" — not "how often" a claim happens, which is frequency's question.

**Confidence intervals use heteroscedasticity-robust (HC3) standard errors, not the raw OLS
ones.** Checked before trusting them, the same discipline Phase 5 applied to the Poisson
model's understated standard errors: a Jarque-Bera test rejected residual normality
(p ≈ 0) and a Breusch-Pagan test rejected constant variance (p ≈ 1.2×10⁻²⁶) — both
assumptions plain OLS inference relies on. Robust standard errors here are 0–16% wider than
the naive ones (a much smaller correction than Poisson's 52%, but real, and applied for the
same reason: don't trust an uncertainty estimate that hasn't been checked).

## Driver factors

### BonusMalus — a genuine severity signal, and a real discovery

| Band | Relativity | 95% CI | Significant? |
|---|---|---|---|
| 50 (best) | 1.00 (reference) | — | — |
| 51-59 | 1.06 | [1.00, 1.12] | Yes |
| 60-79 | 1.22 | [1.17, 1.28] | Yes |
| 80-99 | 1.21 | [1.15, 1.29] | Yes |
| 100-129 | 1.45 | [1.36, 1.55] | Yes |
| 130+ | 1.66 | [1.33, 2.08] | Yes |

Every band significant, monotonically increasing. Phase 3's univariate EDA found only a flat
median and a mean distorted by large claims for BonusMalus — this looked like noise, not
signal. A proper multivariate model, fit on a scale resistant to a few large claims
dominating the estimate, finds a real relationship after all: **checked directly against the
large-loss sensitivity test** (excluding the top 1% of claims barely moves these relativities
— 1.663 → 1.555 at the extreme, 1.062 → 1.050 at the smallest), confirming this is genuine,
not another large-claim artifact. Drivers with the worst score have claims that cost, on
average, 66% more than drivers with the best score — on top of BonusMalus's already-strong
frequency effect (Phase 5), the worst-scored drivers are more expensive on *both* dimensions
of pricing.

### DrivAge — a mixed, partly fragile pattern

| Band | Relativity | 95% CI | Significant? |
|---|---|---|---|
| 18-22 | 1.00 (reference) | — | — |
| 23-29 | 0.87 | [0.79, 0.96] | Yes |
| 30-39 | 0.94 | [0.85, 1.03] | No |
| 40-49 | 1.04 | [0.94, 1.15] | No |
| 50-59 | 1.08 | [0.98, 1.20] | No |
| 60-69 | 1.10 | [0.99, 1.22] | No |
| 70+ | 1.26 | [1.13, 1.41] | Yes |

Only the youngest-adult band (23-29, cheaper) and the oldest band (70+, more expensive) are
significant; the middle is flat. This is a plausible actuarial story (young-adult claims
possibly involving less expensive vehicles; elderly-driver claims possibly involving more
severe injury outcomes) but **the sensitivity check (Phase 6 step 2/3) found DrivAge's
severity relativities, even for the more robust lognormal model, do shift somewhat when the
largest claims are excluded** (23-29's relativity moved from 0.87 to 0.92) — modestly, not
the near-total reversal Gamma showed, but enough that this factor should be read with more
caution than BonusMalus's cleanly-robust result above.

## Vehicle factors — largely no severity signal

| Factor | Result |
|---|---|
| `VehAgeBand` | **No band significant.** Every relativity between 0.92 and 1.00, every CI straddling 1.0. |
| `VehPower` | Not significant (relativity 1.003 per unit). |
| `VehGasBinary` | Not significant (relativity 0.98). |
| `VehBrandGrouped` | Only `B12` significant (1.21) — the same brand that stood out as the sole significant one for frequency too. |

This is a striking contrast with Phase 5: `VehAge` was one of the *strongest* frequency
predictors (the new-vehicle spike), but shows **no relationship to severity at all**. How
often a vehicle claims and how expensive that claim is turn out to be governed by different
things — a vehicle's age predicts whether an accident happens, not how bad it is once it does.

## Geographic factors — mostly no severity signal

`AreaOrdinal` and `LogDensity` are both not significant (unlike frequency, where they were
jointly significant as a group via a likelihood-ratio test). Among 16 grouped regions, only
`R72`, `R91`, `R93`, and the grouped `"Other"` category are significant (relativities
1.11–1.15); the rest are statistically indistinguishable from the `R11` reference.

## Baseline severity

The reference-category combination (18-22 driver, brand-new vehicle, best BonusMalus, Area A,
lowest density, Diesel, lowest VehPower, Region R11, VehBrand B1) has a smearing-corrected
predicted severity of **€1,881.90**.

## Business narrative

Severity tells a genuinely different, and much sparser, story than frequency. **BonusMalus is
the one clear, robust driver of claim cost**, not just claim count — a driver's own claims
history predicts both how often they claim and how expensive those claims are, a real and
useful finding for pricing. Age plays a smaller, partly uncertain role: the youngest adult
band's claims run somewhat cheaper and the oldest band's somewhat more expensive, but this
should be treated with more caution than the BonusMalus result. Everything else that mattered
for frequency — vehicle age, vehicle power, fuel type, population density, area type — shows
**no meaningful relationship to severity** in this model. A policy's rating factors mostly
explain whether it claims, not how expensive that claim will be when it does.

## Caveat carried forward

This model only modestly beats the trivial baseline (validation deviance 1.5273 vs. 1.5690,
~2.7% better) — consistent with Phase 3's original finding, now confirmed by a proper
multivariate model rather than contradicted by one. Use this model's relativities as
directional, defensible findings where significant, not as a strong claim that severity is
well-explained by these variables overall.
