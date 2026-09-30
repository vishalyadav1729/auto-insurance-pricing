# ADR-0002: Surface AWS Monitoring State on the Website via a Committed Snapshot, Not a Live API Call

**Status:** Accepted
**Date:** 2026-09-30
**Deciders:** Project owner (Vishal Yadav)

## Context

ADR-0001 built real AWS infrastructure (S3, Athena, a scheduled Lambda, CloudWatch alarms
and a dashboard) to implement the model card's production-monitoring recommendation. None
of it was visible from the deployed Streamlit app (`riskrate.streamlit.app`) — the only
public-facing artifact most reviewers will actually open. A GitHub repo six folders deep is
not what gets clicked from a resume link; the live app is. This ADR decides how to close
that gap.

## Decision

The app displays a **snapshot** of the CloudWatch alarm states and the Athena
cross-validation summary, refreshed by running a script locally and committing its output
(`app/data/monitoring_snapshot.json`) to git — the same pattern already used for the model
artifacts. **The deployed app never holds AWS credentials and never calls AWS directly.**

## Options Considered

### Option A: Live API calls from the deployed app

| Dimension | Assessment |
|---|---|
| Complexity | Low-medium (a few boto3 calls) |
| Cost | Negligible in dollars — CloudWatch/Athena reads on this data volume cost fractions of a cent even under real traffic |
| Security | **The real problem.** Requires embedding AWS credentials in a public-facing app that any anonymous visitor's browser session ultimately triggers calls through |
| Honesty | Would visually suggest real-time freshness the underlying system doesn't have (the Lambda itself only runs once daily) |

**Pros:** genuinely live; no separate refresh step to remember.
**Cons:** the security exposure is disproportionate to the benefit. Even a read-only
credential scoped to exactly these 6 alarms and 2 Athena tables is still a permanently
open door into a real AWS account, sitting inside a public app with no authentication in
front of it. And "live" here would be cosmetic, not substantive — the data itself is only
ever as fresh as the last daily Lambda run, so a page that re-queries AWS on every load
would look more real-time than it actually is.

### Option B: Scheduled snapshot, committed to git (chosen)

| Dimension | Assessment |
|---|---|
| Complexity | Low (one script, one JSON file, one new app page) |
| Cost | $0 — the refresh script runs locally with credentials that never leave this machine |
| Security | No AWS credentials anywhere in the deployed app's surface area |
| Honesty | The app states its own refresh cadence explicitly, matching what's actually true |

**Pros:** zero credential exposure; matches the existing, already-proven
commit-artifacts-for-deployment pattern (Phase 10); the UI can be explicit about its own
staleness bound rather than implying more freshness than exists.
**Cons:** requires a manual step (run the script, commit, push) to refresh — there is no
automatic redeploy triggered by AWS state changing on its own. Acceptable here: the
underlying monitoring data changes at most once a day regardless, so an unautomated
refresh loses nothing that an automated one would have shown anyway, today.

## Trade-off Analysis

This is not a close call once the security dimension is weighed properly. Option A's only
advantage — genuine live-ness — is worth little when the underlying source data updates at
most daily; Option B gets the same practical freshness with none of the credential-exposure
risk. The manual refresh step in Option B is a real, accepted cost, not an oversight: it
was weighed against the alternative (a public app with standing AWS access) and judged the
better trade.

## Consequences

- **Becomes easier**: an interviewer can see the actual current alarm states (as of the
  last refresh) directly on the deployed site, closing the gap ADR-0001 left open.
- **Becomes harder**: the snapshot can go stale if the script isn't re-run; the UI must
  show its own `refreshed_at` timestamp prominently so staleness is visible, not hidden.
- **To revisit**: if this project ever wanted genuine automated refresh without embedding
  credentials in the public app, the correct next step is a scheduled GitHub Actions
  workflow (AWS credentials as encrypted GitHub Secrets, never exposed to the deployed
  app) that runs this same script and pushes the updated snapshot on a schedule — a
  different trust boundary (GitHub's CI, not a public Streamlit session) than either
  option above. Not built now; noted as the natural next step if wanted.

## Action Items

1. [x] `scripts/refresh_monitoring_snapshot.py` — pulls CloudWatch alarm state, writes
   `app/data/monitoring_snapshot.json`.
2. [x] A new Streamlit page (`app/pages/infrastructure.py`) displaying the snapshot, with
   its `refreshed_at` timestamp shown prominently and an explicit note on why this isn't
   live.
3. [x] Commit the initial snapshot and register the new page in `app/app.py`.
4. [x] Document the "why show this at all" reasoning in `docs/INTERVIEW_PREP.md`.
