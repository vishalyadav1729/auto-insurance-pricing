"""Unit test for the RiskRate monitoring Lambda (lambda/monitoring/lambda_function.py).

This complements, rather than replaces, the real verification already done
by manually invoking the deployed Lambda and confirming via
`aws cloudwatch get-metric-statistics` that published values matched the
source data exactly (reports/monitoring_architecture.md) - that was a
stronger, integration-level check against real AWS. This test exists so a
future edit to the handler's logic gets caught immediately, without needing
AWS credentials or a real deployment to notice a regression.

`lambda/` cannot be imported as a normal Python package (`lambda` is a
reserved keyword), so this inserts `lambda/monitoring` directly onto
sys.path and imports `lambda_function` as a top-level module - the same
pattern test_app.py uses for app/pages/*.py.
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "lambda" / "monitoring"))

import lambda_function  # noqa: E402


def _sample_segments():
    return [
        {"segment_type": "Region", "segment_value": "Other", "n_policies": 3124,
         "exposure": 1417.5, "observed": 500753.19, "predicted": 273750.04, "oe_ratio": 1.8292},
        {"segment_type": "DrivAgeBand", "segment_value": "60-69", "n_policies": 10175,
         "exposure": 6029.9, "observed": 515705.18, "predicted": 901301.11, "oe_ratio": 0.5722},
    ]


def test_handler_publishes_one_metric_datum_per_segment(monkeypatch):
    segments = _sample_segments()
    mock_s3 = MagicMock()
    mock_s3.get_object.return_value = {"Body": io.BytesIO(json.dumps(segments).encode())}
    mock_cloudwatch = MagicMock()

    monkeypatch.setattr(lambda_function, "s3", mock_s3)
    monkeypatch.setattr(lambda_function, "cloudwatch", mock_cloudwatch)

    result = lambda_function.handler({}, None)

    mock_s3.get_object.assert_called_once_with(
        Bucket="riskrate-auto-pricing-data", Key="monitoring/segment_oe.json"
    )
    mock_cloudwatch.put_metric_data.assert_called_once()
    call_kwargs = mock_cloudwatch.put_metric_data.call_args.kwargs
    assert call_kwargs["Namespace"] == "RiskRate/Monitoring"
    assert len(call_kwargs["MetricData"]) == 2

    first = call_kwargs["MetricData"][0]
    assert first["MetricName"] == "ObservedToExpectedRatio"
    assert first["Value"] == 1.8292
    assert {"Name": "SegmentType", "Value": "Region"} in first["Dimensions"]
    assert {"Name": "SegmentValue", "Value": "Other"} in first["Dimensions"]

    assert result["statusCode"] == 200
    assert json.loads(result["body"]) == {"published_segments": 2}


def test_handler_propagates_s3_errors_rather_than_publishing_partial_data(monkeypatch):
    mock_s3 = MagicMock()
    mock_s3.get_object.side_effect = RuntimeError("NoSuchKey")
    mock_cloudwatch = MagicMock()

    monkeypatch.setattr(lambda_function, "s3", mock_s3)
    monkeypatch.setattr(lambda_function, "cloudwatch", mock_cloudwatch)

    try:
        lambda_function.handler({}, None)
        assert False, "expected the S3 error to propagate"
    except RuntimeError:
        pass

    mock_cloudwatch.put_metric_data.assert_not_called()
