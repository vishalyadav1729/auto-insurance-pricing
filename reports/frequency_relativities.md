# Phase 5 — Frequency Model Interpretation

Full relativity tables and plain-language interpretation for the champion frequency model
(Poisson GLM, unregularized, full formula — `reports/frequency_model_selection.md`).

## How to read this

A **relativity** is `exp(coefficient)`: how many times higher (or lower) a category's expected
claim frequency is versus the **reference category** (the one band/level with no coefficient
of its own — every other level in that factor is measured against it). A relativity of 1.5
means "50% more claims than the reference group, all else held equal"; 0.7 means "30% fewer."

**Confidence intervals here use the Negative Binomial model's standard errors, not the
Poisson model's own.** This is a direct, deliberate application of the overdispersion finding
from Phase 5 step 2: the Poisson model's own reported uncertainty is known to be too narrow
(Pearson dispersion ratio 2.31), so treating its raw confidence intervals at face value would
overstate how precisely each relativity is known. Since Poisson and Negative Binomial fit
almost identical point estimates (step 4), the point estimates below are the champion's; the
uncertainty is the more honest, NB-derived version.

## Driver factors

### BonusMalus — the strongest, cleanest signal in the model

| Band | Relativity | 95% CI |
|---|---|---|
| 50 (best) | 1.00 (reference) | — |
| 51-59 | 1.36 | [1.30, 1.42] |
| 60-79 | 1.91 | [1.84, 1.98] |
| 80-99 | 2.20 | [2.09, 2.31] |
| 100-129 | 4.36 | [4.12, 4.62] |
| 130+ | 6.43 | [5.18, 7.98] |

Every band is statistically significant, monotonically increasing, matching Phase 3's EDA
exactly and even sharpening it once other factors are controlled for. A driver with the worst
BonusMalus score has, on average, **6.4× the claim frequency** of a driver with the best score.
The widest interval (130+: [5.18, 7.98]) reflects the small sample behind it (364 training
policy-years, flagged since Phase 3) — the direction and rough size are trustworthy, but not
pinned down as precisely as the other bands.

### DrivAge — a surprise that needs explaining, not just reporting

| Band | Relativity | 95% CI | Significant? |
|---|---|---|---|
| 18-22 | 1.00 (reference) | — | — |
| 23-29 | 0.68 | [0.63, 0.73] | Yes |
| 30-39 | 0.71 | [0.66, 0.76] | Yes |
| 40-49 | 1.01 | [0.94, 1.10] | **No** |
| 50-59 | 1.00 | [0.92, 1.08] | **No** |
| 60-69 | 0.97 | [0.90, 1.06] | **No** |
| 70+ | 1.04 | [0.95, 1.13] | **No** |

Phase 3's univariate EDA found a strong young-driver effect across the *whole* age range. Once
`BonusMalus` is also in the model, only the two youngest bands remain distinguishable from the
18-22 reference — everyone 40 and older is statistically indistinguishable from the youngest
band, which was not the univariate story at all.

**Why**: `BonusMalus` already encodes years of accumulated driving history — an experienced,
50-year-old driver and an experienced, 30-year-old driver can both reach the best BonusMalus
score, and once that's accounted for, `DrivAge` has little *additional* information left to
contribute for the older/middle bands. Its independent signal survives only for the youngest
bracket, who by definition haven't had time to build a BonusMalus history yet. This is not a
contradiction of Phase 3 — it's what a multivariate model is supposed to do: avoid crediting
two overlapping factors for the same underlying effect.

## Vehicle factors

### VehAge — strong and significant throughout, but not monotonic

| Band | Relativity | 95% CI |
|---|---|---|
| 0 (new) | 1.00 (reference) | — |
| 1-2 | 0.31 | [0.29, 0.32] |
| 3-5 | 0.32 | [0.31, 0.34] |
| 6-9 | 0.34 | [0.33, 0.36] |
| 10-14 | 0.29 | [0.28, 0.31] |
| 15-19 | 0.22 | [0.21, 0.24] |
| 20+ | 0.19 | [0.17, 0.22] |

Every band is significant, confirming Phase 3's finding that brand-new vehicles (`VehAge=0`)
are the highest-risk group of all — roughly 3-5× the frequency of every other age band. This
is exactly why banding, not a linear term, was the right call in Phase 4: the relationship
drops sharply then declines gently, which no single straight line could represent.

### VehPower — small per-unit effect, meaningful cumulative effect

Coefficient 0.0103 per unit of power (relativity 1.010, CI [1.004, 1.017], **significant**) —
this looked "weak" in Phase 3's univariate EDA, but the multivariate model finds it real, if
modest. Compounded across `VehPower`'s actual range (4 to 15), the cumulative effect is
**+12.1%** — comparable in size to some of the more visually obvious categorical factors.

### VehGas — modest but significant

Regular-fuel vehicles have 1.06× the frequency of Diesel (CI [1.03, 1.09], significant) — a
small, real effect.

### VehBrand

| Brand | Relativity | Significant? |
|---|---|---|
| B1 | 1.00 (reference) | — |
| B12 | 1.11 | **Yes** |
| B2, B3, B4, B5, B6, B10, B11, B13, Other | 0.86 – 1.07 | No |

Only `B12` (backed by 64,802 training policy-years — the second-largest brand) stands out as
individually significant. Every other brand, including the grouped `"Other"` category, is
statistically indistinguishable from the reference.

## Geographic factors — individually weak, jointly strong

`AreaOrdinal` (relativity 1.02, CI [0.98, 1.06]) and `LogDensity` (relativity 1.03 *per
e-fold* increase in density — i.e. density × 2.718 — CI [1.00, 1.06]; equivalently 1.02 per
plain doubling, CI [1.00, 1.04]) are each **not individually significant** — surprising, given
Phase 3 found clear, smooth univariate trends for both. Before concluding "geography doesn't
matter," this was checked properly rather than taken at face value from single p-values (the
plan is explicit that variable selection shouldn't rest on p-values alone): a **likelihood
ratio test** dropping `AreaOrdinal`, `LogDensity`, and `RegionGrouped` together (18 parameters)
gives a test statistic of 201.05 against a chi-squared(18) reference — **p ≈ 6.2×10⁻³³**.
Geography, as a block, is overwhelmingly significant.

**Why the individual pieces look weak when the group is clearly real**: `Area`, `Density`, and
`Region` are strongly correlated with each other (in France's actual geography, area type and
region are largely what determine population density in the first place). When several
correlated factors are all in the same model, it becomes hard to say *which specific one*
deserves the credit — the model can be confident that geography matters while being unable to
cleanly split that credit three ways. This is a standard statistical phenomenon
(**multicollinearity**), not a data quality problem.

Despite the individually modest coefficients, the *cumulative* range effects are real:

- **Density**: from the least dense location (1 person/km²) to the most dense (27,000/km²),
  the cumulative relativity is **1.30×** (+29.6%).
- **Area**: from A to F, the cumulative relativity is **1.10×** (+9.9%).

### Region

| Region | Relativity | Significant? |
|---|---|---|
| R11 | 1.00 (reference) | — |
| R22 | 1.15 | Yes |
| R24 | 1.16 | Yes |
| R41 | 0.89 | Yes |
| R52 | 1.08 | Yes |
| R53 | 1.17 | Yes |
| R82 | 1.14 | Yes |
| R23, R25, R26, R31, R54, R72, R73, R91, R93, Other | 0.95 – 1.10 | No |

Six of sixteen regions (plus the grouped `"Other"`) stand out as individually significant; the
rest are statistically indistinguishable from the `R11` reference once everything else in the
model is accounted for.

## Business narrative — what actually drives this portfolio's frequency

Putting the significant findings together: the biggest, clearest driver of claim frequency is
**a driver's own claims history** (`BonusMalus`) — a driver at the worst score claims over six
times as often as one at the best. **The vehicle's age** is nearly as dramatic: brand-new
vehicles claim 3-5× as often as any other age band, a genuine and unexpected finding worth
flagging to underwriting rather than assuming "newer is safer." **Being a very young driver**
(18-22) carries a real, independent penalty on top of BonusMalus — younger drivers who haven't
yet built a track record are priced up specifically for that lack of history, not just for
being young. **Geography** (region, local area type, population density) clearly matters as a
group, even though the model can't cleanly attribute that effect to any one of the three
overlapping signals. **Vehicle power and fuel type** contribute smaller, but real and
statistically defensible, adjustments.

## Caveat carried forward

All significance calls above use Negative-Binomial-corrected standard errors specifically
because the Poisson model's own are known to understate uncertainty (Phase 5, step 2). Anyone
extending this analysis should do the same, rather than trusting the Poisson model's own
`summary()` output for inference.
