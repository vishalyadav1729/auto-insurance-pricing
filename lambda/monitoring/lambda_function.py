"""RiskRate monitoring Lambda (ADR-0001).

Reads the curated segment-level observed-to-expected summary from S3
(uploaded by scripts/compute_monitoring_metrics.py) and republishes it as
CloudWatch custom metrics. Triggered on a schedule by an EventBridge rule.

Deliberately boto3-only - no pandas/numpy/joblib - because the actual model
inference already happened locally when segment_oe.json was generated. This
keeps the Lambda a plain zip upload with no dependency layer needed (boto3
ships with the Lambda Python runtime already), and keeps cold starts and
execution time trivial.

Honest limitation, stated here and in docs/adr/0001: this is a portfolio
project with no live claims stream, so the S3 file this reads doesn't
actually change between scheduled runs. This Lambda demonstrates the
scheduled-publish-to-CloudWatch operational pattern a real deployment would
use, not genuine live production telemetry.
"""

from __future__ import annotations

import json

import boto3

S3_BUCKET = "riskrate-auto-pricing-data"
S3_KEY = "monitoring/segment_oe.json"
METRIC_NAMESPACE = "RiskRate/Monitoring"
METRIC_NAME = "ObservedToExpectedRatio"

s3 = boto3.client("s3")
cloudwatch = boto3.client("cloudwatch")


def handler(event, context):
    obj = s3.get_object(Bucket=S3_BUCKET, Key=S3_KEY)
    segments = json.loads(obj["Body"].read())

    metric_data = [
        {
            "MetricName": METRIC_NAME,
            "Dimensions": [
                {"Name": "SegmentType", "Value": seg["segment_type"]},
                {"Name": "SegmentValue", "Value": seg["segment_value"]},
            ],
            "Value": seg["oe_ratio"],
            "Unit": "None",
        }
        for seg in segments
    ]

    # PutMetricData accepts at most 1000 MetricDatum per call, well above the
    # 6 curated segments this project publishes - no batching needed here,
    # but the limit is worth naming for anyone extending the segment list.
    cloudwatch.put_metric_data(Namespace=METRIC_NAMESPACE, MetricData=metric_data)

    return {
        "statusCode": 200,
        "body": json.dumps({"published_segments": len(metric_data)}),
    }
