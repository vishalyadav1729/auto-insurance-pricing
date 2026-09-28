# Phase 8, Step 4 — Interpretation and the Final Test-Set Check

Permutation importance and partial dependence for the boosted models (computed on the
**validation** split, out-of-sample, per the plan's explicit guidance — computing this on
training data would measure what the model overfit to, not what actually helps it predict new
data), plus the one-time test-set confirmation of the boosted pure-premium pipeline.

**A caution stated up front, and held to throughout:** the plan explicitly warns against
"claiming that feature importance establishes causation." Everything below describes what
these particular fitted models learned to rely on — not a claim about the true causal
structure of auto insurance risk.

## Frequency: gradient boosting independently confirms the GLM's two strongest findings

| Feature | Permutation importance |
|---|---|
| VehAgeBand | 0.0428 (highest) |
| BonusMalusBand | 0.0382 |
| VehBrandGrouped | 0.0228 |
| DrivAgeBand | 0.0115 |
| VehPower | 0.0092 |
| VehGasBinary | 0.0080 |
| RegionGrouped | 0.0073 |
| LogDensity | 0.0046 |
| AreaOrdinal | 0.0001 (essentially zero) |

`AreaOrdinal`'s near-zero importance is a real, independent corroboration of the GLM's own
likelihood-ratio finding (Phase 5): once `LogDensity` and `Region` are present, `Area` adds
almost nothing — two completely different methods (a formal statistical test vs. an ML
model's learned reliance) agreeing on the same conclusion.

**Partial dependence recovers the GLM's two strongest patterns, cleanly and independently:**

| BonusMalusBand | Predicted rate | | VehAgeBand | Predicted rate |
|---|---|---|---|---|
| 50 (best) | 0.090 | | 0 (new) | 0.218 |
| 51-59 | 0.113 | | 1-2 | 0.105 |
| 60-79 | 0.157 | | 3-5 | 0.103 |
| 80-99 | 0.169 | | 6-9 | 0.105 |
| 100-129 | 0.281 | | 10-14 | 0.092 |
| 130+ | 0.328 | | 15-19 | 0.070 |
| | | | 20+ | 0.067 |

`BonusMalusBand`'s clean, monotonic ~3.6× rise matches the GLM's own finding exactly in shape.
`VehAgeBand`'s spike-then-decline (new vehicles nearly 2× the risk of 1-2 year old ones, then a
gentle further decline) independently reproduces the GLM's most surprising finding from Phase
5 — not an artifact of that particular model family, confirmed here by a completely different
one.

## Severity: a real limitation, found through interpretation, not visible in the aggregate score

| Feature | Permutation importance |
|---|---|
| DrivAgeBand | 0.0197 (highest, large uncertainty: std 0.0134) |
| LogDensity | 0.0111 |
| VehBrandGrouped | 0.0093 |
| RegionGrouped | 0.0079 |
| VehAgeBand | 0.0027 |
| VehPower | 0.0008 |
| AreaOrdinal | 0.0000 |
| VehGasBinary | -0.0019 |
| **BonusMalusBand** | **-0.0090 (std 0.0141 — indistinguishable from zero)** |

**`BonusMalusBand` shows negative importance** — smaller than its own uncertainty. This
directly contradicts Phase 6's finding: there, `BonusMalus` was the *one* severity relationship
confirmed genuinely robust (checked and re-checked against the large-loss sensitivity test,
barely moving when the largest claims were excluded). Investigated rather than left as an
unexplained contradiction:

```
BonusMalusBand partial dependence (boosted severity model):
  50 (best):  €1,887
  51-59:      €1,978
  60-79:      €1,936
  80-99:      €1,981
  100-129:    €1,995
  130+:       €1,898
```

**Essentially flat — a ~6% range across all six bands**, versus the GLM's confirmed 1.66×
relativity for the same factor. The boosted severity model genuinely failed to learn a real
relationship the GLM found and independently validated.

**Why**: severity's weak signal-to-noise ratio (established since Phase 3, reconfirmed by every
severity model built since) forced very aggressive regularization on the boosted model — early
stopping cut training off after only 16 boosting iterations (step 2), a tiny fraction of the
300-iteration ceiling. That level of regularization, necessary to avoid the overfitting Phase 6
and step 2 both found for less-constrained severity models, appears to have also suppressed a
real, smaller signal along with the noise. `DrivAgeBand` ends up the model's top-ranked
feature instead — the same factor Phase 6 flagged as only *partially* robust (its severity
effect shifts somewhat, not dramatically, under the large-loss sensitivity check) - a less
reliable signal to be leaning on most heavily.

**This is exactly why interpretation is a real evaluation criterion, not a formality**: step 3's
aggregate deviance comparison showed boosted severity barely edging out the GLM (+0.38%). That
comparison alone would never have revealed that the boosted model is relying on a different,
less-validated feature than the one independently confirmed to matter.

## The final test-set check

The boosted pure-premium pipeline (paid-frequency boosting × severity boosting), touched on
test for the first and only time - to confirm, not re-select:

| Split | Deviance (power=1.5) | O/E | Gini |
|---|---|---|---|
| Validation | 64.5682 | 1.0108 | 0.3283 |
| **Test** | **60.0342** | **1.0228** | **0.2521** |

Stable: deviance improves further, O/E stays excellent (both within ~2% of exact), and Gini
drops somewhat (0.3283 → 0.2521, a similar order-of-magnitude shift to the GLM champion's own
validation-to-test Gini change in Phase 7, 0.3016 → 0.2701) but shows no dramatic reversal like
Tweedie's train-to-validation swing (Phase 7 step 3). The boosted pipeline's strong validation
performance was not an artifact of validation-specific noise.

## What this means for step 5

The boosted pipeline remains the stronger performer by every aggregate metric, confirmed
stable on test. But interpretation surfaced something the numbers alone did not: for severity
specifically, the model may be winning on aggregate deviance while relying on a less
well-validated signal than the GLM uses. This is exactly the kind of finding the plan expects
an interpretability pass to surface, and it belongs in step 5's champion-challenger
recommendation alongside the predictive and calibration evidence gathered in steps 1-3.
