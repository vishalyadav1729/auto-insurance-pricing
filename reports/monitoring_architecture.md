# Production Monitoring: Segment-Level Calibration (AWS Lambda + CloudWatch)

`reports/model_card.md`'s retraining/monitoring recommendation was, until now, only a
written suggestion: *"Recommend production monitoring of observed-to-expected ratio by
region and by DrivAge band ... given the documented and unresolved segment-level
miscalibration."* This implements it. Full design rationale and options considered are in
`docs/adr/0001-production-monitoring-architecture.md`; this report covers what was actually
built and verified.

## What this is, and isn't

This is a portfolio project with no live claims stream, so "monitoring" here means a
**scheduled job demonstrating the operational pattern** a real deployment would use — not
genuine live production telemetry. The Lambda re-publishes the same static validation-set
metrics on each scheduled run, since there's no new data arriving between runs. This is
stated plainly here and in the ADR, not glossed over.

## Architecture

```
scripts/compute_monitoring_metrics.py  (local, one-time per model refresh)
        │  computes O/E for 6 curated segments from the GLM champion
        │  on the validation split, writes segment_oe.json
        ▼
S3 (riskrate-auto-pricing-data/monitoring/segment_oe.json)
        ▲
        │  reads via s3:GetObject
EventBridge (rate(1 day)) ──invokes──▶ Lambda (riskrate-monitoring)
                                              │  publishes via cloudwatch:PutMetricData
                                              ▼
                                    CloudWatch (namespace RiskRate/Monitoring)
                                        │                    │
                                        ▼                    ▼
                                  6 Alarms              1 Dashboard
                            (thresholds tuned to        (RiskRate-Monitoring)
                             the known-bad values)
```

All infrastructure-as-code lives in `lambda/monitoring/`: the Lambda handler
(`lambda_function.py`), its IAM trust and permissions policies, the dashboard definition
(`dashboard.json`), and `deploy.sh` — the exact sequence of AWS CLI commands used to build
all of it, kept as a record for reproducibility rather than relying on console clicks that
leave no trail.

## The 6 curated segments

Publishing all ~15 Region/DrivAgeBand combinations as separate CloudWatch custom metrics
would cost ~$18/year once past the free 10-metric allotment (see the ADR's trade-off
analysis) — not negligible by this project's own cost bar. Instead, only the segments
already flagged as worst-calibrated in Phase 7/9 are monitored:

| Segment | Value | O/E (recomputed) | Published (Phase 7/9 report) | Match |
|---|---|---|---|---|
| Region | Other | 1.8292 | 1.83 | Yes |
| Region | R41 | 0.4589 | 0.46 | Yes |
| Region | R24 (largest) | 0.7604 | 0.76 | Yes |
| DrivAgeBand | 60-69 | 0.5722 | 0.57 | Yes |
| DrivAgeBand | 70+ | 1.3323 | 1.33 | Yes |
| DrivAgeBand | 40-49 | 0.6929 | 0.69 | Yes |

Recomputed independently via `scripts/compute_monitoring_metrics.py` rather than copied
from the earlier reports — every value matched, confirming no drift between the persisted
model artifacts and what Phase 7/9 originally found.

## Verified results

- **Lambda**: manually invoked after deployment; returned `{"published_segments": 6}`.
  Confirmed via `aws cloudwatch get-metric-statistics` that the published value
  (1.8292351253076506 for Region=Other) matches the source data exactly — no transcription
  or unit error introduced by the publish step.
- **Alarms**: all 6 created with thresholds set so the already-known values would trip
  them (e.g., Region=Other alarms above 1.5; the real value is 1.83). After one full daily
  evaluation period, **all 6 transitioned to `ALARM` state**, exactly as designed — a
  concrete, verified demonstration that this alerting setup would have caught the
  calibration problems this project already found, not just a theoretical claim.
- **Dashboard**: `RiskRate-Monitoring`, showing bar-chart comparisons by segment type,
  single-value tiles per segment, and an alarm-status panel.

## Cost

- **Lambda**: ~30 invocations/month, sub-second execution, 128MB memory — far inside the
  always-free 1M-requests/400,000-GB-seconds allotment. $0.
- **EventBridge**: one low-frequency scheduled rule targeting one Lambda. Negligible to $0.
- **CloudWatch**: 6 custom metrics and 6 alarms, both under the always-free 10-per-account
  allotment; 1 dashboard, within the free dashboard allotment. $0.
- **Total: $0/year**, consistent with the ADR's estimate. AWS free-tier terms can change —
  worth a periodic spot-check against AWS's current pricing page, the same caveat given for
  every other AWS cost estimate in this project.

## Files

- `docs/adr/0001-production-monitoring-architecture.md` — the design decision and options considered.
- `scripts/compute_monitoring_metrics.py` — computes the 6 segments' O/E and uploads to S3.
- `lambda/monitoring/lambda_function.py` — the monitoring Lambda.
- `lambda/monitoring/deploy.sh` — the exact AWS CLI commands used to build everything.
- `lambda/monitoring/{iam_trust_policy,iam_permissions_policy,dashboard}.json` — supporting definitions.
