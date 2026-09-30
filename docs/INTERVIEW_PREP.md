# Interview Preparation

Likely questions an interviewer would ask about this project, with answers grounded in
what was actually found and verified while building it — not generic textbook answers.
Each answer cites the specific report or number it comes from, so it can be checked and
expanded on if pushed further. Organized by theme: modelling choices, methodology and
validation, machine-learning evaluation, communication, and limitations/critique.

## Modelling choices

**Q: You found real overdispersion in the frequency data (Pearson ratio 2.31) — why did
you still choose Poisson over Negative Binomial as the champion?**

Because point predictions were statistically indistinguishable between the two: validation
deviance was 0.594235 (Poisson) vs. 0.594267 (Negative Binomial) — a difference far smaller
than meaningful. Overdispersion affects how *confident* you can be in the coefficients
(the NB's standard errors are wider and more honest), not what the coefficients actually
predict. I used the NB's standard errors for every significance claim in
`reports/frequency_relativities.md`, but kept Poisson as the simpler, more standard, more
interpretable production model. Choosing a more complex model without a real predictive
benefit would have been complexity for its own sake.

**Q: Why lognormal over Gamma for severity, when Gamma is the more textbook choice for
positive, right-skewed cost data?**

Because on this specific data, the full Gamma GLM actually *underperformed* the trivial
baseline on validation, and I traced why rather than just reporting the worse number:
`Region` and `VehBrand` together contributed 26 of the model's 48 parameters, and Gamma's
raw-scale likelihood is highly sensitive to the handful of very large claims in this
portfolio (top 1% of claims = 41% of total value). Fitting on the log scale instead made
the lognormal model resistant to exactly that sensitivity, and the same feature set that
hurt Gamma *helped* lognormal. This is documented and compared directly in
`reports/severity_model_selection.md` — I didn't pick lognormal by default, I picked it
because I checked and it won.

**Q: Lognormal models are known to be biased when you exponentiate predictions back to the
original scale. Did you account for that?**

Yes — Duan's smearing estimator. Naively exponentiating the log-scale OLS prediction gives
you the median of the cost distribution, not the mean, because `exp(E[log Y]) != E[Y]` for
a right-skewed variable. I confirmed this wasn't a theoretical nitpick on this data: the
naive approach predicted only 40% of true total claim cost; the smearing-corrected version
predicted 99%. The correction (the mean of `exp(residual)` on training data) is baked into
`predict_lognormal_severity` so it's structurally impossible to call the model without it.

**Q: Why not just use a single Tweedie GLM to predict pure premium directly, instead of
two separate frequency and severity models?**

I built and compared both. The direct Tweedie GLM (power=1.5) was competitive but lost on
every metric that mattered — deviance (67.89 vs. the champion's 65.44), calibration, and
Gini (0.26 vs. 0.30) — and showed an unexplained calibration reversal between train (+23%
over-prediction) and validation (−32%) that I investigated (ruled out the same overfitting
mechanism that hurt Gamma severity) but couldn't fully explain. I disclosed that as an open
finding rather than force a tidier story. The frequency × severity decomposition also has a
practical advantage Tweedie doesn't: it lets you inspect and defend each component
separately, which matters for both governance and business communication.

## Methodology and validation

**Q: How did you prevent data leakage across your train/validation/test split?**

Every data-dependent decision — the rare-category pooling threshold, feature engineering
choices, model family selection, hyperparameter tuning — was fit using the training split
only, then applied identically to validation and test. The one deliberate exception is the
banding cut-points for age/vehicle-age/BonusMalus, which are fixed, domain-motivated
constants (not statistics computed from data), so I didn't re-derive them per split — I
disclosed that as a lower-risk simplification rather than pretending it's zero-risk.

**Q: You mention touching the test set "exactly once" — what does that actually mean in
practice, and why does it matter?**

Every model-selection decision — GLM vs. GLM comparisons, GLM vs. ML challenger, Tweedie
vs. the frequency×severity decomposition — was made using validation data only. Test was
used exactly once per pipeline, at the very end, purely to confirm the already-chosen
champion was stable (e.g., the GLM's O/E moved from 0.843 on validation to 0.850 on test —
no reversal). The reason this matters: if you peek at test repeatedly while iterating, you
implicitly overfit to it the same way you'd overfit to training data with enough attempts —
your final "test performance" stops being an honest estimate of generalization.

**Q: You found and fixed several real bugs while building this. Walk me through one.**

The clearest one: `statsmodels`'s own `.predict()` silently drops the exposure offset
unless you pass it again explicitly at prediction time — no error, no warning. I confirmed
it directly: calling `results.predict(train)` without re-supplying the offset overstated
total training claims by more than 2×, because it implicitly treated every policy as if it
had a full year of exposure. It recurred across four different result classes over the
project (Poisson GLM, Negative Binomial, regularized GLM, Tweedie GLM) — each time, I
re-verified rather than assumed the fix would carry over. I fixed it once, structurally, by
writing a `predict_frequency()` wrapper that's the only sanctioned way to call `.predict()`
anywhere in the codebase, rather than trusting every call site to remember.

**Q: How do you know your Poisson frequency model is well-calibrated, not just accurate on
average?**

Portfolio-level observed-to-expected ratio (0.9998) tells you the *aggregate* is right, but
a model can nail the aggregate while being badly wrong for specific risk bands. I checked
calibration by decile of predicted risk specifically to catch that, and found good
tracking across all 10 deciles (5% to 33% predicted rate) on validation. I applied the same
discipline to the pure-premium model at the segment level (by Region and by driver age)
later, and *that's* where I actually found a real, unresolved problem — see Limitations
below. The decile/segment check is what surfaces problems a single portfolio-wide number
hides.

## Machine-learning evaluation

**Q: Your gradient-boosting challenger beat the GLM on deviance. Why didn't you just switch
to it?**

Because deviance alone doesn't tell the whole story, and I checked further rather than
stopping at the win. Interpretation (permutation importance and partial dependence, both
computed out-of-sample) revealed the boosted severity model's `BonusMalus` importance was
statistically indistinguishable from zero — directly contradicting the GLM's finding that
`BonusMalus` is a genuine, robust severity driver (confirmed stable under a large-loss
sensitivity check). I traced why: severity's weak signal-to-noise ratio forced very
aggressive early stopping (16 of a possible 300 boosting iterations), which suppressed a
real but smaller signal along with the noise. A model that wins on aggregate deviance while
silently dropping a validated real relationship is exactly the kind of risk a governance
review exists to catch — so I recommended a champion-challenger approach instead of a
single winner: the GLM as the operational champion, the boosted pipeline as a challenger
worth developing further (specifically via a GLM-residual-boosting approach that would
prevent it from re-learning, and potentially dropping, signals the GLM already has right).

**Q: What's the actual scorecard you used to compare the GLM and the boosted model?**

Nine criteria, not one: predictive deviance, portfolio calibration, segment-level
calibration, ranking (Gini), stability across train/validation/test, interpretability,
implementation complexity, sensitivity to large claims, and governance burden. The boosted
model won on deviance,
portfolio calibration, and Gini; the GLM won decisively on interpretability and governance
burden, and the segment-level calibration check actually favored the GLM in the worst case
(the `Other` region: GLM under-predicted by 1.83×, boosted by 2.54×) despite the boosted
model's better *aggregate* numbers. Full table in
`reports/ml_champion_challenger_recommendation.md`. The point of using a scorecard instead
of a single metric is exactly this: it surfaced a real tradeoff a single "which deviance is
lower" comparison would have hidden.

**Q: Did the machine-learning model at least agree with the GLM on what matters?**

Partially, and I checked this explicitly rather than assumed it. For frequency, yes —
independently, via a completely different model family, gradient boosting recovered the
GLM's two strongest findings (`BonusMalus`'s effect, and the new-vehicle risk spike), which
is a genuinely reassuring cross-validation of those findings. For severity, no — it missed
`BonusMalus` entirely, which is the specific finding that drove my champion-challenger
recommendation rather than a straightforward "boosting wins."

## Communication and business framing

**Q: How would you explain a "relativity" to a non-technical stakeholder?**

A relativity says how many times more (or less) a group is expected to cost relative to a
baseline group, holding everything else equal. For example, `BonusMalus`'s worst band has
a frequency relativity of 6.4 — meaning a driver at that score is predicted to claim 6.4×
as often as a driver at the best score, after accounting for every other factor in the
model (age, vehicle, location, etc.). I'd emphasize the "holding everything else equal"
part specifically, since it's the part people most often get wrong when reading these
tables informally.

**Q: If you had to tell an actuarial pricing committee whether this model is ready for
production, what would you say?**

Not as-is, and I'd lead with why rather than bury it: two real segment-level miscalibration
problems are documented and unresolved (Region, O/E 0.46–1.83; driver age, O/E 0.57–1.33),
neither explained by sample size, and no fairness or proxy-variable audit is actually
*possible* on this dataset because it has no demographic fields at all — that's a hard data
limitation, not a completed check. I'd present the model card and consolidated limitations
document (`reports/model_card.md`, `reports/limitations.md`) as the artifacts a committee
should actually review, since a portfolio-level accuracy number alone would hide exactly
the problems that matter for a real pricing decision.

## Limitations and critique

**Q: What's the single biggest weakness of this project, in your own assessment?**

No temporal validation. The dataset is a single historical snapshot, so my train/validation/
test split is a random split, not a chronological holdout — meaning nothing in this project
demonstrates how these models would perform on genuinely *future* data, which is what a
real deployment actually needs. I disclosed this explicitly rather than let a good
validation/test agreement imply more than it does; a real deployment would need to validate
against a forward time window before I'd trust any of these numbers to hold up in
production.

**Q: If you had another two weeks, what would you do next?**

Two concrete things, in order. First, the GLM-corrected residual-boosting approach flagged
in the champion-challenger recommendation — use the GLM's prediction as a baseline offset
and let boosting learn only the residual, which should capture genuine non-linear
improvement without the "silently drops a validated signal" failure mode I found. Second,
dig further into the Region miscalibration: I checked whether it reduced to driver-age
composition (Phase 9) and found it didn't, so there's a real, still-unexplained pattern
there worth more investigation before any regional pricing decision is made from this
model.

**Q: This is a well-known public dataset. What would be different with real production
data?**

Several things I can name concretely because I ran into their absence here: real claim and
policy *dates* (for genuine temporal validation, which this dataset's single snapshot
can't support), demographic fields to actually run a fairness/proxy audit instead of just
flagging that one isn't possible, and enough history to detect drift and retrain on a
schedule rather than treating "the model" as a one-time artifact. I'd also expect
production data to force real decisions about mid-term policy changes and multi-year
customer history that this single-snapshot dataset doesn't have to represent at all.

## Cloud/infrastructure

**Q: I see you used AWS here — walk me through what you actually built and why.**

The raw data lives in S3, queried through Athena via `CREATE EXTERNAL TABLE` — no Glue
Crawlers or Glue ETL jobs, both of which bill per DPU-hour for something a single SQL DDL
statement does for free (Athena registers table metadata in the Glue Data Catalog
automatically; that catalog itself is the free part of Glue). I used it to independently
re-derive several numbers this project already had from pandas — portfolio frequency,
pure premium, the orphan-claim totals — in a completely different execution engine
(Presto/Trino SQL, not pandas). Every one matched exactly, which is a real correctness
check, not a demo for its own sake: if pandas and SQL had disagreed, that would have meant
one of them had a bug I hadn't found yet.

**Q: Why didn't you just refit the GLM in SQL too, if you were already there?**

Because that would be using the wrong tool to look busy. A GLM coefficient is a
maximum-likelihood estimate — a genuine statistical model fit, not an aggregation. SQL is
the right tool for the data-preparation and descriptive-aggregation work I actually used it
for (the BonusMalus band-level frequency table in `reports/sql_cross_validation.md` is a
naive per-band rate, and I say so explicitly — it is *not* presented as reproducing the
GLM's multivariate relativity, which holds every other factor constant and SQL's GROUP BY
does not). Conflating the two would be a real mistake, not a shortcut worth taking.

**Q: How much did this cost you?**

Under a tenth of a cent, total. Every query I ran while building this scanned a combined
~163MB, and Athena bills $5 per TB scanned — the actual dataset is tiny (34MB raw), so
every query stayed far below any meaningful cost. The two AWS services that *can* get
expensive for this kind of workload are Glue Crawlers/ETL jobs and QuickSight (a paid
subscription after a 30-day trial); I specifically avoided both and can explain exactly
why for either one if asked.

**Q: You mentioned a production monitoring recommendation in your model card. Did you
actually build it, or is it just a suggestion on paper?**

I built it. It's a scheduled Lambda (EventBridge, once a day) that publishes
observed-to-expected ratio for six segments to CloudWatch, with alarms tuned to the
already-known-bad values. I wrote an ADR before touching any infrastructure — using it
caught a real problem at the design stage: publishing every Region/DrivAge segment
combination as a separate CloudWatch custom metric would have cost about $18/year once past
the free 10-metric allotment, which isn't "negligible" by the bar I'd set for this project.
The fix was to curate the list down to the six segments already known to be worst-
calibrated, rather than monitor everything — which is also just better monitoring design,
not only the cheaper choice; alerting on every possible segment indiscriminately is how you
get alert fatigue in a real system.

**Q: How do you know the alarms actually work, rather than just existing?**

I checked, rather than assumed. After the first full daily evaluation period completed, all
six alarms transitioned to `ALARM` state — exactly as designed, since the thresholds were
set specifically to trip on the real, already-known O/E values (e.g., the `Other` region
alarms above 1.5, and its actual value is 1.83). That's a verified demonstration that this
alerting would catch the calibration problems this project already found, not a claim I'm
making without having watched it happen.

**Q: Isn't "monitoring" a stretch here, given this is a portfolio project with no live
data?**

Fair challenge, and I say so directly in the write-up rather than let it go unstated: this
Lambda republishes the same static validation-set metrics on every scheduled run, since
there's no live claims stream feeding it new data. What it demonstrates is the *operational
pattern* — scheduled computation, publishing to a metrics system, alerting on thresholds — a
real deployment would use, not genuine production telemetry. I'd rather be upfront about
that limitation than let the word "monitoring" imply something this project can't actually
back up.

**Q: Why does any of this AWS work show up on your deployed app, and not just in the
GitHub repo?**

Because the GitHub repo isn't what gets clicked from a resume — the live app link is. Work
that only exists six folders deep in `reports/` is functionally invisible to someone doing
a two-minute skim; it's an unverifiable claim on a resume rather than something they can see
in the room. So I added an "AWS Infrastructure" page to the app itself that shows the actual
current state of the CloudWatch alarms and a summary of the Athena cross-validation —
turning "I built AWS monitoring" into something demonstrable in ten seconds, not something
they have to take on faith.

**Q: Why didn't you make that page call AWS live, if the goal was to demonstrate it works?**

I considered it and decided against it, and that decision is itself the more interesting
answer. Making it live means embedding AWS credentials inside a public-facing app — even
scoped to read-only access on exactly six alarms, that's a standing, credentialed door into
a real AWS account sitting behind no authentication, reachable by anyone who finds the URL.
The benefit doesn't justify that risk, especially since the underlying data only changes
once a day at most (that's the monitoring Lambda's own schedule) — a page that queried AWS
on every load would look more real-time than the system actually is, which conflicts with
how this project has handled every other limitation: disclosed plainly, not implied away.

Instead, the page reads a snapshot file (`app/data/monitoring_snapshot.json`) that I refresh
by running a script locally, with credentials that never leave my machine, then commit to
git — the exact same pattern already used to get the trained model files into the deployed
app in the first place. The write-up for this decision is
`docs/adr/0002-monitoring-snapshot-not-live.md`. If I had more time, the natural next step
without reintroducing the credential-exposure problem would be a scheduled GitHub Actions
workflow — AWS credentials as encrypted CI secrets, never exposed to the public app itself —
running this same refresh script automatically instead of by hand.
