# Phase 8, Step 5 — Champion-Challenger Recommendation

The plan is explicit about what this recommendation is and isn't: "The goal is not to prove
that machine learning is superior. The goal is to determine whether its incremental predictive
value justifies its additional complexity," using a scorecard — predictive deviance, portfolio
calibration, segment calibration, ranking, stability, interpretability, implementation
complexity, sensitivity to large claims, governance burden — not a single metric.

## One more check, run before writing this recommendation, not after

Interpretation (step 4) found the boosted severity model's `BonusMalusBand` signal was
statistically indistinguishable from zero — a direct contradiction of the GLM's validated,
sensitivity-tested finding. Before concluding anything from that, the same large-loss
sensitivity check applied to the GLM (Phase 6) was run on the boosted model too, since it had
never been checked that way and a fair comparison requires it:

| BonusMalusBand | Partial dependence (full data) | Partial dependence (top 1% excluded) |
|---|---|---|
| 50 (best) | €1,887 | €1,359 |
| 51-59 | €1,978 | €1,373 |
| 60-79 | €1,936 | €1,433 |
| 80-99 | €1,981 | €1,416 |
| 100-129 | €1,995 | €1,445 |
| 130+ | €1,898 | €1,427 |

Both flat (a ~6% range with large claims included, ~6% without) — this **strengthens** the
step 4 concern rather than explaining it away. It isn't that large claims were masking a real
signal the trimmed model would reveal; the boosted model simply never learned the BonusMalus
relationship, regardless of large-claim influence.

## The scorecard

| Criterion | Frequency | Severity | Combined Pure Premium |
|---|---|---|---|
| **Predictive deviance** (validation) | Boosted better: 0.5766 vs. GLM 0.5942 (**-3.0%**) | Boosted marginally better: 1.5215 vs. GLM 1.5273 (**-0.38%**) | Boosted better: 64.57 vs. GLM 65.44 (**-1.3%**) |
| **Portfolio calibration** (O/E) | Both excellent (~0.99-1.00) | Both reasonable (Boosted 1.028, GLM close to 1) | **Boosted much better**: 1.011 vs. GLM's 0.843 |
| **Segment calibration** | Not separately checked | Not separately checked | **GLM comparable, Boosted worse in the extreme case** — `Other` region: GLM 1.83x, Boosted 2.54x under-prediction |
| **Ranking (Gini)** | Not separately checked | Not separately checked | Boosted better: 0.328 vs. GLM's 0.302 (**+8.9%**) |
| **Stability (train/val/test)** | Both checked, stable | Both checked, stable | Both stable on the one-time test check (GLM: 0.843→0.850; Boosted: 1.011→1.023) |
| **Interpretability** | Both agree on the same two strongest drivers (independent cross-validation) | **GLM wins decisively**: Boosted fails to learn the one factor (BonusMalus) independently validated as robust | GLM offers explicit relativities; Boosted requires permutation importance/partial dependence tooling to explain at all |
| **Sensitivity to large claims** | Not separately checked (both use the same features) | **GLM's finding validated as real; Boosted's absence of signal confirmed persistent** either way | Inherits severity's finding |
| **Implementation complexity** | GLM simpler (fewer hyperparameters, standard software) | Same | Same |
| **Governance burden** | GLM lower (standard actuarial practice, transparent to regulators) | **GLM much lower** — a model that misses a validated real signal while looking fine in aggregate deviance is exactly the kind of risk a governance review exists to catch | GLM lower |

## The recommendation: champion-challenger, not a single winner

**Operational champion: the GLM Frequency × Severity pipeline.**

Three reasons, not one:

1. **The boosted model's advantages are real but not transformative** at the component level
   (3.0% frequency, 0.38% severity) — modest gains that don't obviously justify the governance
   and interpretability cost on their own.
2. **Severity boosting has a genuine, now twice-confirmed flaw**: it doesn't use the one
   severity signal that's been independently validated as real (BonusMalus, checked stable
   under large-loss sensitivity in both Phase 6 for the GLM and now, separately, for the
   boosted model). A ~0.4% aggregate deviance edge is not worth adopting a model that has
   quietly dropped a known, true relationship — this is precisely the plan's warning against
   trusting deviance or Gini alone.
3. **The pure-premium calibration gap (O/E 0.843) is real, but it is a narrower, better-
   understood problem than switching model families** — it's a documented limitation
   (Phase 7 step 4) that could plausibly be addressed with a smaller, targeted recalibration
   rather than adopting a pipeline with its own new, different segment-level problems (the
   worse `Other`-region miscalibration).

**Challenger worth continued development: the boosted Frequency × Severity pipeline**,
specifically for its pure-premium calibration and ranking advantage — real, substantial, and
not to be dismissed. Two concrete next steps, not pursued in this phase but worth naming:

- **A GLM-corrected boosting approach** (the plan's own "advanced" suggestion): use the GLM's
  prediction as a baseline/offset and let boosting learn only the *residual* pattern beyond
  it. This would directly prevent the "boosting silently drops a validated signal" problem —
  the GLM's BonusMalus relativity would already be baked into the baseline, not something
  boosting has to rediscover — while still capturing genuine non-linear improvement on top.
- **Retune severity boosting with less aggressive regularization** now that the actual failure
  mode is understood (early stopping at 16 iterations is very likely too little for this
  factor to surface), and re-run the interpretation check to confirm whether that resolves it.

## What this recommendation deliberately avoids

Per the plan's own list of mistakes: this does **not** select a model from Gini alone (the
boosted pipeline's best metric), does **not** compare a tuned ML model against an untuned GLM
(both were properly tuned throughout Phase 8), does **not** report only training performance
(everything above is validation, with a one-time test confirmation), and does **not** treat
feature importance as establishing causation (it is used here only to check whether each
model's *learned* reliance matches an already-validated finding, which is a narrower and more
defensible claim).
