"""Pull the current state of the AWS monitoring infrastructure (CloudWatch
alarms/metrics from ADR-0001, plus a static summary of the Athena SQL
cross-validation) and write a snapshot the Streamlit app can display.

This is deliberately NOT called by the deployed app itself - see
docs/adr/0002-monitoring-snapshot-not-live.md for why. The deployed app on
Streamlit Community Cloud never holds AWS credentials; instead, this script
is run locally (with the AWS CLI credentials already configured in this
dev environment), and its output - app/data/monitoring_snapshot.json - is
committed to git like the model artifacts already are. Streamlit Cloud
auto-redeploys on push, so re-running this script and pushing is how the
website's displayed snapshot gets refreshed.

Usage:
    python scripts/refresh_monitoring_snapshot.py
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import boto3

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = PROJECT_ROOT / "app" / "data" / "monitoring_snapshot.json"

REGION = "ca-central-1"

# Maps each alarm (ADR-0001, lambda/monitoring/deploy.sh) to the segment it
# watches - kept here rather than parsed from CloudWatch dimensions, since
# the alarm-to-segment mapping is a design decision, not discoverable data.
ALARM_SEGMENTS = [
    {"alarm_name": "riskrate-oe-region-other-high", "segment_type": "Region", "segment_value": "Other"},
    {"alarm_name": "riskrate-oe-region-r41-low", "segment_type": "Region", "segment_value": "R41"},
    {"alarm_name": "riskrate-oe-region-r24-low", "segment_type": "Region", "segment_value": "R24"},
    {"alarm_name": "riskrate-oe-drivage-60-69-low", "segment_type": "DrivAgeBand", "segment_value": "60-69"},
    {"alarm_name": "riskrate-oe-drivage-70plus-high", "segment_type": "DrivAgeBand", "segment_value": "70+"},
    {"alarm_name": "riskrate-oe-drivage-40-49-low", "segment_type": "DrivAgeBand", "segment_value": "40-49"},
]

# Static summary of the Athena SQL cross-validation (reports/sql_cross_validation.md).
# Not re-queried here - the underlying raw data never changes, so re-running
# Athena on every snapshot refresh would just re-confirm the same numbers at
# a small, pointless recurring cost. Update this block by hand only if
# sql/cross_validation_queries.sql or the underlying data ever changes.
SQL_CROSS_VALIDATION = {
    "checked_at": "2026-09-29",
    "checks": [
        {"label": "Portfolio frequency", "pandas_value": "0.1006", "sql_value": "0.100614", "match": True},
        {"label": "Portfolio pure premium", "pandas_value": "€167.18", "sql_value": "€167.176", "match": True},
        {"label": "Total claim value", "pandas_value": "€60,697,930.68", "sql_value": "€60,697,930.68", "match": True},
        {"label": "Orphan claim total", "pandas_value": "€788,714.18", "sql_value": "€788,714.18", "match": True},
    ],
}


def fetch_alarm_snapshot() -> list[dict]:
    cloudwatch = boto3.client("cloudwatch", region_name=REGION)
    alarm_names = [a["alarm_name"] for a in ALARM_SEGMENTS]
    described = cloudwatch.describe_alarms(AlarmNames=alarm_names)
    alarms_by_name = {a["AlarmName"]: a for a in described["MetricAlarms"]}

    snapshot = []
    for entry in ALARM_SEGMENTS:
        alarm = alarms_by_name[entry["alarm_name"]]
        snapshot.append(
            {
                **entry,
                "state": alarm["StateValue"],
                "state_updated_at": alarm["StateUpdatedTimestamp"].isoformat(),
                "threshold": alarm["Threshold"],
                "comparison_operator": alarm["ComparisonOperator"],
            }
        )
    return snapshot


def main() -> None:
    snapshot = {
        "refreshed_at": datetime.now(timezone.utc).isoformat(),
        "alarms": fetch_alarm_snapshot(),
        "sql_cross_validation": SQL_CROSS_VALIDATION,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(snapshot, indent=2))
    print(f"wrote {OUT_PATH}")
    for a in snapshot["alarms"]:
        print(f"  {a['alarm_name']:<35} {a['state']}")


if __name__ == "__main__":
    main()
