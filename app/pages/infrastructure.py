"""RiskRate — AWS infrastructure snapshot page (ADR-0002).

Displays a snapshot of the real AWS monitoring infrastructure built for
ADR-0001 (S3 + Athena SQL cross-validation, a scheduled Lambda publishing
to CloudWatch, alarms tuned to already-known-bad segment values) - the
part of this project that otherwise would only be visible in the GitHub
repo, not from the deployed app itself.

Reads app/data/monitoring_snapshot.json, written by
scripts/refresh_monitoring_snapshot.py run locally and committed to git -
this page (and the deployed app generally) never holds AWS credentials
and never calls AWS directly. See ADR-0002 for why that's a deliberate
choice, not a limitation to apologize for.

Run via the router: streamlit run app/app.py
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT_PATH = PROJECT_ROOT / "app" / "data" / "monitoring_snapshot.json"


def load_snapshot(path: Path) -> dict | None:
    """Return the parsed snapshot, or None if it hasn't been generated yet -
    a plain function so this is unit-testable without needing Streamlit or
    a real snapshot file on disk.
    """
    if not path.exists():
        return None
    return json.loads(path.read_text())


def hours_since(refreshed_at_iso: str, now: datetime | None = None) -> float:
    """Hours elapsed since the snapshot was refreshed - separated from the
    UI so the "how stale is this" logic can be tested with a fixed `now`
    rather than depending on the real clock.
    """
    refreshed_at = datetime.fromisoformat(refreshed_at_iso)
    now = now or datetime.now(timezone.utc)
    return (now - refreshed_at).total_seconds() / 3600


st.title("AWS Infrastructure")

with st.container(border=True):
    st.markdown(
        "**Why this page exists**: the AWS work behind this project (S3 + Athena SQL "
        "cross-validation, a scheduled Lambda publishing calibration metrics to "
        "CloudWatch) is real and verified, but a GitHub repo isn't what gets opened from "
        "a resume link — this page is, so the infrastructure work is demonstrable here "
        "directly, not just documented six folders deep.\n\n"
        "**Why this isn't live**: showing this data would otherwise require embedding "
        "AWS credentials in this public app — a standing, credentialed door into a real "
        "AWS account behind no authentication. The data below is instead a snapshot, "
        "refreshed by running a script locally and committing the result, the same "
        "pattern already used for this app's model files. See "
        "[`docs/adr/0002-monitoring-snapshot-not-live.md`](https://github.com/vishalyadav1729/auto-insurance-pricing/blob/main/docs/adr/0002-monitoring-snapshot-not-live.md) "
        "for the full reasoning."
    )

snapshot = load_snapshot(SNAPSHOT_PATH)

if snapshot is None:
    st.error(
        "No snapshot found yet. Run `python scripts/refresh_monitoring_snapshot.py` "
        "and redeploy to populate this page."
    )
    st.stop()

age_hours = hours_since(snapshot["refreshed_at"])
st.caption(
    f"Snapshot refreshed {age_hours:.1f} hours ago "
    f"({snapshot['refreshed_at'][:19].replace('T', ' ')} UTC). "
    "Refreshed manually, not continuously — see the explanation above."
)

st.header("CloudWatch alarms")
st.caption(
    "6 curated segments (the worst-calibrated Region/DrivAge combinations already found "
    "in Phase 7/9), monitored by a Lambda that runs daily via EventBridge "
    "(`docs/adr/0001-production-monitoring-architecture.md`). Thresholds were set to trip "
    "on the already-known-bad values — as they correctly did."
)

with st.container(border=True):
    cols = st.columns(3)
    for i, alarm in enumerate(snapshot["alarms"]):
        with cols[i % 3]:
            state = alarm["state"]
            icon = "🔴" if state == "ALARM" else ("🟢" if state == "OK" else "⚪")
            st.markdown(f"**{icon} {alarm['segment_type']}: {alarm['segment_value']}**")
            st.caption(
                f"State: {state}  \n"
                f"Threshold: O/E {alarm['comparison_operator'].replace('Than', ' than ').lower()} "
                f"{alarm['threshold']}"
            )

st.header("Athena SQL cross-validation")
st.caption(
    "Portfolio-level figures independently re-derived in SQL (AWS Athena, Presto/Trino) "
    "against the raw data in S3, cross-checked against the original pandas-computed "
    "values (`reports/sql_cross_validation.md`). Not re-queried on every snapshot refresh "
    "— the underlying raw data never changes, so re-running would only re-confirm the "
    "same numbers at a small, pointless recurring cost."
)

with st.container(border=True):
    for check in snapshot["sql_cross_validation"]["checks"]:
        icon = "✅" if check["match"] else "❌"
        st.markdown(
            f"{icon} **{check['label']}**: pandas = `{check['pandas_value']}`, "
            f"SQL = `{check['sql_value']}`"
        )
