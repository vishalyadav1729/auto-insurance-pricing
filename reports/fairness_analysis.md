# Phase 9, Step 1 — Fairness and Responsible-Use Analysis

## What this analysis can and can't do

This dataset carries no demographic fields — no race, gender, income, marital status, or
occupation. That is a hard constraint on what follows: **this is not, and cannot be, a formal
disparate-impact audit** in the regulatory sense, which requires exactly the fields this dataset
doesn't have. What this analysis *can* do, and does:

1. Extend the segment-calibration checks already run for `Region` and `BonusMalus`
   (`pure_premium_segment_stability.md`, Phase 7 step 4) to `DrivAge` — an explicit age variable,
   and the one rating factor here most directly analogous to a protected characteristic.
2. Check whether `Region`'s already-documented, unexplained miscalibration correlates with age
   composition — testing whether one open problem compounds another, rather than assuming either
   way.
3. State plainly what cannot be checked here, instead of implying an audit happened that didn't.

Computed in `notebooks/07_fairness_analysis.ipynb`, refitting the champion pipeline (paid-
frequency GLM × lognormal severity GLM) on training data and evaluating on validation — the same
split Phase 7 step 4 used for its own segment checks, and consistent with reserving test for the
one-time confirmation already completed in Phase 7.

## DrivAge segment calibration — a new check, and a real finding

| DrivAge band | Policies | Exposure (policy-years) | Observed (€) | Predicted (€) | O/E |
|---|---|---|---|---|---|
| 18-22 | 2,481 | 969 | 490,353 | 503,355 | 0.97 |
| 23-29 | 10,891 | 4,732 | 930,584 | 1,017,174 | 0.91 |
| 30-39 | 25,317 | 12,315 | 1,652,124 | 1,896,641 | 0.87 |
| 40-49 | 24,785 | 13,105 | 1,620,873 | 2,339,133 | **0.69** |
| 50-59 | 21,475 | 12,019 | 1,753,295 | 2,015,187 | 0.87 |
| 60-69 | 10,175 | 6,030 | 515,705 | 901,301 | **0.57** |
| 70+ | 6,578 | 4,469 | 947,327 | 711,065 | **1.33** |

**This was not previously checked, and it's a real, disclosable finding**, similar in kind to
Phase 7's Region result: O/E ranges from 0.57 to 1.33 — a ~2.3× spread. Two patterns worth
naming directly:

- **Ages 40-69 are systematically over-predicted** (the model charges more than the observed
  cost would justify), most severely for 60-69 (O/E 0.57) and 40-49 (O/E 0.69).
- **The oldest band, 70+, is under-predicted** (O/E 1.33) — the *opposite* direction from every
  other band above 40.
- **This is not a sample-size artifact**: 60-69 has 6,030 policy-years of exposure — six times
  the smallest band (18-22, 969 policy-years) — yet 18-22 is the best-calibrated band in the
  table (O/E 0.97). A larger segment being *more* miscalibrated than the smallest one rules out
  "it's just a thin-sample problem" as a full explanation, the same conclusion Phase 7 reached
  for Region.

**Why this matters for fairness, specifically**: `DrivAge` is the rating factor most directly
analogous to a legally protected characteristic (age) among those used in this project. A model
that systematically over-charges 40-69 year-olds relative to their actual claims cost, while
under-charging drivers 70 and older, is a concrete, quantified pattern — not a hypothetical
concern — that any real deployment would need to address before use, independent of whether
`DrivAge` is retained as a rating factor at all.

## Does Region's miscalibration compound with age? Checked, not confirmed

Phase 7 step 4 found `Region` O/E ranging from 0.46 to 1.83, unexplained by segment size. Given
the DrivAge finding above, the natural next question is whether regions with different age
compositions inherit some of DrivAge's own miscalibration pattern.

| Region | Mean DrivAge | Region O/E (Phase 7 step 4) |
|---|---|---|
| R41 | 48.5 (oldest) | 0.46 (most over-predicted) |
| R93 | 47.1 | 1.20 (under-predicted) |
| R53 | 46.5 | — (not separately reported in Phase 7) |
| R24 (largest region) | 46.1 | 0.76 |
| R91 | 46.0 | — |
| R72 | 45.8 | 0.55 |
| R54 | 45.8 | — |
| R25 | 45.7 | — |
| R73 | 45.4 | 0.51 |
| R26 | 45.1 | — |
| Other | 45.1 | 1.83 (most under-predicted) |
| R82 | 44.8 | 0.95 |
| R52 | 44.7 | 0.53 |
| R11 | 44.6 | 0.73 |
| R23 | 43.3 | — |
| R22 | 41.9 | — |
| R31 | 40.9 (youngest) | — |

**Checked directly and found insufficient to explain the pattern, consistent with how this
project has handled similar open questions before (Tweedie's calibration reversal, Phase 7 step
3; Region O/E vs. exposure size, Phase 7 step 4).** R41 (oldest mean age) is the most
over-predicted region — consistent with the DrivAge finding that older working-age drivers are
over-predicted. But R93 (second-oldest mean age, very close to R41) is *under*-predicted, and
`Other` (middling mean age) is the *most* under-predicted region of all. There is no clean
monotonic relationship between a region's age composition and its O/E ratio. **Region's
miscalibration is not simply age composition in disguise** — it remains its own, separately
unexplained problem, not resolved by this check.

## What cannot be checked here

- No correlation between any rating factor and a protected characteristic (race, gender, income,
  etc.) can be computed, because this dataset contains none of those fields. This is a data
  limitation, not a decision to skip the check.
- Whether `Region`, `Density`, or `Area` function as proxies for a protected characteristic is
  **not verifiable from this dataset**. It is flagged here as an open question this project
  cannot answer, not asserted as either present or absent.
- A genuine fairness audit of a production version of this model would need either direct access
  to protected-class data under appropriate legal safeguards, or an indirect proxy methodology
  (e.g., geocoding-based estimation) using external reference data this project does not have
  and does not attempt to substitute for.

## Jurisdiction and deployment considerations

Background only — none of the following was checked against any specific jurisdiction's actual
rules, and none of it should be read as legal advice:

- **Age-based rating** for adult drivers is restricted or banned in some jurisdictions. Given
  the concrete miscalibration found above, `DrivAge`'s use here would need jurisdiction-specific
  legal review before any real deployment, independent of the fairness question.
- **Territory-based rating** (`Region`/`Area`/`Density`) faces restrictions in some
  jurisdictions, partly because geography can serve as an unintended proxy for a protected
  characteristic — precisely the question this dataset cannot resolve either way.
- **`BonusMalus`** is France's own regulated encoding of a driver's claims history, not a
  demographic variable by construction — the most defensible of this project's strong rating
  factors on fairness grounds, though this framing was not independently audited here.
- **Deploying without ongoing monitoring would be a mistake regardless of the above**: this
  model carries two confirmed, unresolved segment-level miscalibration patterns (Region, and now
  DrivAge) that a real deployment would need to monitor continuously, not check once at launch.

## Summary

- `DrivAge` segment calibration, checked for the first time here: O/E ranges 0.57-1.33,
  over-predicting ages 40-69 and under-predicting 70+, not explained by sample size.
- Region's known miscalibration does not reduce to age composition — checked directly, found
  insufficient, left as its own open problem.
- What this project cannot check (any real proxy-variable analysis, absent demographic data) is
  stated plainly rather than glossed over.
- `reports/model_card.md`'s fairness section is updated (see that file) to reflect these
  findings rather than remain marked preliminary.

### Next

Step 2: consolidate every limitation documented across Phases 2-9 into one reference document,
`reports/limitations.md`.
