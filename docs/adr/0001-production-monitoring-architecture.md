# ADR-0001: Production Monitoring Architecture for Segment-Level Calibration

**Status:** Proposed
**Date:** 2026-09-29
**Deciders:** Project owner (Vishal Yadav)

## Context

`reports/model_card.md` and `reports/fairness_analysis.md` already found and disclosed two
unresolved segment-level calibration problems in the GLM pricing champion:

- **Region**: observed-to-expected (O/E) ratio ranges 0.46–1.83 on the validation split,
  not explained by segment size (`reports/pure_premium_segment_stability.md`).
- **DrivAge band**: O/E ranges 0.57–1.33, over-predicting ages 40-69 and under-predicting
  70+ (`reports/fairness_analysis.md`).

The model card's own retraining/monitoring recommendation is explicit: *"Recommend
production monitoring of observed-to-expected ratio by region and by DrivAge band ... given
the documented and unresolved segment-level miscalibration."* That recommendation has never
been implemented — only stated. This ADR designs the implementation.

**A constraint worth stating plainly**: this is a portfolio project with no live claims
stream. "Monitoring" here means demonstrating the *operational pattern* a real deployment
would use — a scheduled job computing and publishing segment metrics, with alarms on
known-risk segments — run against the static validation split as a stand-in for live data.
This is disclosed explicitly in the implementation, not presented as genuine production
telemetry.

**Hard constraints:**
- Total cost must be negligible — effectively $0/year, not just "cheap."
- Must not use AWS Glue Crawlers/ETL jobs or Amazon QuickSight (both identified in advance
  as the AWS services that could actually cost real money for a workload this size).
- Must build on what already exists: S3 bucket `riskrate-auto-pricing-data` (ca-central-1),
  admin-level IAM access, the already-fitted GLM artifacts.

## Decision

Build a **scheduled AWS Lambda function, triggered by an EventBridge rule, that reads a
precomputed segment-metrics file from S3 and publishes it to CloudWatch as custom metrics**,
with **CloudWatch Alarms on a curated subset of the already-identified worst segments** and
a **CloudWatch Dashboard** for visualization — no QuickSight, no Glue.

## Options Considered

### Option A: Lambda (EventBridge-scheduled) + CloudWatch metrics/alarms/dashboard

| Dimension | Assessment |
|---|---|
| Complexity | Low-medium |
| Cost | $0/year if metric count is kept ≤10 (CloudWatch's Always-Free custom-metric allotment) |
| Scalability | Not a concern at this data volume; the pattern itself scales to a real deployment |
| Team familiarity | Standard, widely-documented AWS serverless pattern |

**Pros:** fully serverless (no idle compute cost), a real, recognizable production
monitoring pattern, ties directly to the model card's own recommendation, alarm thresholds
can be set to the *already-known* bad values as a concrete demonstration.

**Cons:** the "monitored data" doesn't actually change between scheduled runs (no live
stream) — must be disclosed honestly rather than presented as live telemetry.

### Option B: Athena scheduled query + QuickSight dashboard

| Dimension | Assessment |
|---|---|
| Complexity | Medium |
| Cost | QuickSight: ~$9–24/user/month after a 30-day trial — **fails the cost constraint outright** |

**Pros:** reuses the Athena setup from the S3+Athena work already done.
**Cons:** disqualified by cost alone; not a close call.

### Option C: Glue ETL job (scheduled) + Glue Data Quality + SNS alerts

| Dimension | Assessment |
|---|---|
| Complexity | Medium-high |
| Cost | Glue bills per DPU-hour with minimum billing durations — real, non-trivial cost even for a tiny job — **fails the explicit "avoid Glue Crawlers/ETL jobs" constraint** |

**Pros:** purpose-built for data-quality monitoring.
**Cons:** disqualified by the constraint already set for this project; heavier than the
data volume justifies regardless.

### Option D: A "Monitoring" page added to the existing Streamlit app, no new AWS services

| Dimension | Assessment |
|---|---|
| Complexity | Low |
| Cost | $0 |

**Pros:** simplest possible option, zero new AWS surface area.
**Cons:** demonstrates nothing new — the explicit goal of this piece of work is to show
AWS monitoring/observability skill beyond what the S3+Athena work already covered. Cheapest
is not the right criterion here; fitness for the actual goal is.

## Trade-off Analysis

Options B and C are disqualified outright by this project's own cost constraints, not by a
close trade-off. Option D is the cheapest but fails the actual objective (demonstrating AWS
monitoring architecture). Option A is the only one that satisfies the cost constraint *and*
the goal — **with one important refinement**: CloudWatch bills custom metrics per unique
namespace+name+dimension-value combination past the first 10 (Always Free). Publishing all
~15 Region/DrivAge segment combinations as separate metrics would push this into
~$1.50/month (~$18/year) — not "negligible" by this project's own stated bar. The fix is to
**publish a curated subset (≤8) of the segments the model card and fairness analysis already
flagged as the worst-calibrated**, not all of them indiscriminately. This is a real,
defensible engineering trade-off in its own right — a production alerting system that
pages on every segment regardless of severity causes alert fatigue; monitoring the specific
known-risk segments is the more mature design, not just the cheaper one.

## Recommended Design

**Curated segments to monitor** (6 total, well under the free 10-metric allotment):

| Segment | Value | Known O/E | Why it's included |
|---|---|---|---|
| Region | `Other` | 1.83 | Worst under-prediction |
| Region | `R41` | 0.46 | Worst over-prediction |
| Region | `R24` | 0.76 | Largest region by exposure — miscalibration here isn't a small-sample artifact |
| DrivAgeBand | `60-69` | 0.57 | Worst over-prediction by age |
| DrivAgeBand | `70+` | 1.33 | Worst under-prediction by age |
| DrivAgeBand | `40-49` | 0.69 | Largest-exposure age band with real miscalibration |

**Data flow:**

1. **(One-time, local)** Compute each segment's O/E from the already-fitted GLM champion
   against the validation split (reusing the exact logic already in
   `app/pages/model_exploration.py`) and write a small JSON summary (6 rows: segment type,
   segment value, n_policies, exposure, observed, predicted, oe_ratio) to
   `s3://riskrate-auto-pricing-data/monitoring/segment_oe.json`.
2. **AWS Lambda** (Python, boto3 only — no pandas/numpy needed, since the model inference
   already happened locally in step 1): reads `segment_oe.json` from S3, calls
   `cloudwatch:PutMetricData` once per segment under namespace `RiskRate/Monitoring`,
   metric name `ObservedToExpectedRatio`, dimensions `{SegmentType, SegmentValue}`.
3. **EventBridge rule** (`rate(1 day)`): invokes the Lambda on a schedule — the genuine
   scheduled-job pattern a real deployment would use, even though this portfolio's
   underlying source file doesn't change between runs (disclosed limitation, not hidden).
4. **CloudWatch Alarms**: one per curated segment, threshold set so the *already-known* O/E
   value would trip it (e.g., Region=Other alarms if O/E > 1.5; Region=R41 alarms if
   O/E < 0.6) — a concrete demonstration that the alerting would have caught the problems
   this project already found.
5. **CloudWatch Dashboard**: one widget per segment showing its current metric value,
   viewable directly in the AWS console.

**IAM permissions needed:**
- Lambda execution role: `s3:GetObject` scoped to
  `arn:aws:s3:::riskrate-auto-pricing-data/monitoring/*`, `cloudwatch:PutMetricData`, plus
  `AWSLambdaBasicExecutionRole` (CloudWatch Logs only — no broader access).
- EventBridge needs `lambda:InvokeFunction` permission on the function (a resource-based
  policy statement added when the rule target is created).

**Cost estimate:**
- Lambda: ~30 invocations/month, sub-second each, minimal memory — nowhere near the
  1M-requests/400,000-GB-seconds Always-Free allotment. **$0.**
- EventBridge: a single low-frequency scheduled rule targeting one Lambda — negligible to
  free at this volume.
- CloudWatch: 6 custom metrics (under the 10 Always-Free), 6 alarms (under the 10
  Always-Free), 1 dashboard (within the free dashboard allotment). **$0.**
- **Total: $0/year**, as long as the segment list stays curated (≤10) rather than
  exhaustive. AWS free-tier terms can change — worth a spot-check against AWS's current
  pricing page before relying on this long-term, the same caveat given for every other AWS
  cost estimate in this project.

## Consequences

- **Becomes easier**: the model card's monitoring recommendation is no longer just a
  written suggestion — there's a working reference implementation an interviewer can be
  shown, with alarms that demonstrably would have caught the project's own known issues.
- **Becomes harder**: nothing structurally, but the "monitoring" is honestly a simulated
  schedule against static data, not live telemetry — this must stay disclosed everywhere
  it's described (README, this ADR, any write-up), consistent with this project's practice
  since Phase 2.
- **To revisit**: if this project ever got real production data, the Lambda's data source
  (step 1's local computation) would become a real feature-store or data-warehouse query
  instead of a one-time local export, and the segment list would need periodic review
  rather than staying fixed at the 6 currently worst-known.

## Action Items

1. [ ] Compute the 6 curated segments' O/E locally and upload `segment_oe.json` to S3.
2. [ ] Write and deploy the Lambda function (with its execution role).
3. [ ] Create the EventBridge scheduled rule targeting the Lambda.
4. [ ] Create the 6 CloudWatch Alarms.
5. [ ] Create the CloudWatch Dashboard.
6. [ ] Run the Lambda once manually to confirm metrics appear before relying on the schedule.
7. [ ] Document the whole thing in `reports/monitoring_architecture.md` and link it from
   `reports/model_card.md`'s retraining/monitoring section and the README.
