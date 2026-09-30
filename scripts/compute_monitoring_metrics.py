"""Compute observed-to-expected ratio for the curated monitoring segments
(ADR-0001, docs/adr/0001-production-monitoring-architecture.md) and upload
the result to S3, where the monitoring Lambda reads it.

Curated to 6 segments - the worst Region/DrivAgeBand combinations the model
card and fairness analysis already flagged - specifically to stay under
CloudWatch's free 10-custom-metric allotment (publishing all ~15 segment
combinations would cost ~$18/year instead of $0, per the ADR).

This recomputes the same numbers already published in
reports/pure_premium_segment_stability.md and reports/fairness_analysis.md,
as a live source for the monitoring pipeline - not a new analysis. Confirmed
while building this that every value matches those reports' published
numbers (e.g. Region Other: 1.8292 here vs. 1.83 published).

Usage:
    python scripts/compute_monitoring_metrics.py [--upload]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from auto_pricing.frequency import predict_frequency
from auto_pricing.pure_premium import fit_paid_frequency_glm
from auto_pricing.severity import build_severity_table, fit_lognormal_model, predict_lognormal_severity

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

S3_BUCKET = "riskrate-auto-pricing-data"
S3_KEY = "monitoring/segment_oe.json"

# The 6 curated segments (ADR-0001) - the worst-calibrated Region and
# DrivAgeBand values already identified in Phase 7/9, not an exhaustive list.
CURATED_SEGMENTS = [
    ("Region", "RegionGrouped", "Other"),
    ("Region", "RegionGrouped", "R41"),
    ("Region", "RegionGrouped", "R24"),
    ("DrivAgeBand", "DrivAgeBand", "60-69"),
    ("DrivAgeBand", "DrivAgeBand", "70+"),
    ("DrivAgeBand", "DrivAgeBand", "40-49"),
]


def compute_segment_oe() -> list[dict]:
    pp_table = pd.read_parquet(PROCESSED_DIR / "pure_premium_table.parquet")
    train = pp_table[pp_table["split"] == "train"].copy()
    val = pp_table[pp_table["split"] == "validation"].copy()

    paid_freq_result = fit_paid_frequency_glm(train)
    paid_freq_pred_val = predict_frequency(paid_freq_result, val)

    sev = pd.read_parquet(PROCESSED_DIR / "severity_clean.parquet")
    severity_table = build_severity_table(sev, pp_table.drop(columns=["ClaimNbFromSev", "ClaimAmountSum"]))
    sev_train = severity_table[severity_table["split"] == "train"].copy()
    severity_result, smearing_factor = fit_lognormal_model(sev_train)
    severity_pred_val = predict_lognormal_severity(severity_result, smearing_factor, val)

    val_check = val.copy()
    val_check["pred"] = paid_freq_pred_val * severity_pred_val

    results = []
    for seg_type, col, value in CURATED_SEGMENTS:
        subset = val_check[val_check[col] == value]
        observed = subset["ClaimAmountSum"].sum()
        predicted = subset["pred"].sum()
        results.append(
            {
                "segment_type": seg_type,
                "segment_value": value,
                "n_policies": int(len(subset)),
                "exposure": float(subset["Exposure"].sum()),
                "observed": float(observed),
                "predicted": float(predicted),
                "oe_ratio": float(observed / predicted),
            }
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upload", action="store_true", help="upload the result to S3")
    args = parser.parse_args()

    results = compute_segment_oe()
    for r in results:
        print(f"{r['segment_type']:<12} {r['segment_value']:<8} n={r['n_policies']:>6}  O/E={r['oe_ratio']:.4f}")

    out_path = PROJECT_ROOT / "artifacts" / "monitoring" / "segment_oe.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out_path}")

    if args.upload:
        import boto3

        boto3.client("s3").upload_file(str(out_path), S3_BUCKET, S3_KEY)
        print(f"uploaded to s3://{S3_BUCKET}/{S3_KEY}")


if __name__ == "__main__":
    main()
