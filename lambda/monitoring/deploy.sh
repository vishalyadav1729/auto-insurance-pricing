#!/bin/bash
# Deploys the RiskRate monitoring Lambda + EventBridge schedule + CloudWatch
# alarms/dashboard (ADR-0001, docs/adr/0001-production-monitoring-architecture.md).
#
# This is a record of the exact commands used to build the infrastructure -
# run it top to bottom on a fresh AWS account/region to reproduce the setup.
# Not idempotent by design (create-role/create-function fail if already
# present) - that's intentional so a re-run surfaces drift rather than
# silently no-op'ing over it.
#
# PREREQUISITE: run `python scripts/compute_monitoring_metrics.py --upload`
# first - step 3 below invokes the Lambda, which reads
# s3://riskrate-auto-pricing-data/monitoring/segment_oe.json and will fail
# with a NoSuchKey error if that file doesn't exist yet.
set -euo pipefail

REGION="ca-central-1"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "== 1. IAM role for the Lambda's execution =="
aws iam create-role \
  --role-name riskrate-monitoring-lambda-role \
  --assume-role-policy-document file://"$SCRIPT_DIR/iam_trust_policy.json" \
  --region "$REGION"

aws iam attach-role-policy \
  --role-name riskrate-monitoring-lambda-role \
  --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole \
  --region "$REGION"

aws iam put-role-policy \
  --role-name riskrate-monitoring-lambda-role \
  --policy-name riskrate-monitoring-permissions \
  --policy-document file://"$SCRIPT_DIR/iam_permissions_policy.json" \
  --region "$REGION"

echo "== 2. Package and create the Lambda (allow a few seconds for IAM propagation) =="
sleep 8
(cd "$SCRIPT_DIR" && zip -q /tmp/monitoring_lambda.zip lambda_function.py)

aws lambda create-function \
  --function-name riskrate-monitoring \
  --runtime python3.12 \
  --role "arn:aws:iam::${ACCOUNT_ID}:role/riskrate-monitoring-lambda-role" \
  --handler lambda_function.handler \
  --zip-file fileb:///tmp/monitoring_lambda.zip \
  --timeout 30 \
  --memory-size 128 \
  --region "$REGION"

echo "== 3. Confirm it works before wiring up the schedule =="
sleep 5
aws lambda invoke --function-name riskrate-monitoring --region "$REGION" /tmp/monitoring_invoke_result.json
cat /tmp/monitoring_invoke_result.json

echo "== 4. EventBridge daily schedule =="
aws events put-rule \
  --name riskrate-monitoring-daily \
  --schedule-expression "rate(1 day)" \
  --state ENABLED \
  --description "Triggers the RiskRate monitoring Lambda daily (ADR-0001)" \
  --region "$REGION"

aws lambda add-permission \
  --function-name riskrate-monitoring \
  --statement-id riskrate-monitoring-eventbridge \
  --action lambda:InvokeFunction \
  --principal events.amazonaws.com \
  --source-arn "arn:aws:events:${REGION}:${ACCOUNT_ID}:rule/riskrate-monitoring-daily" \
  --region "$REGION"

aws events put-targets \
  --rule riskrate-monitoring-daily \
  --targets "Id=1,Arn=arn:aws:lambda:${REGION}:${ACCOUNT_ID}:function:riskrate-monitoring" \
  --region "$REGION"

echo "== 5. CloudWatch alarms - thresholds set to trip on the already-known-bad values (ADR-0001) =="
create_alarm() {
  aws cloudwatch put-metric-alarm \
    --alarm-name "$1" --alarm-description "$2" \
    --namespace "RiskRate/Monitoring" --metric-name "ObservedToExpectedRatio" \
    --dimensions Name=SegmentType,Value="$3" Name=SegmentValue,Value="$4" \
    --statistic Average --period 86400 --evaluation-periods 1 \
    --threshold "$5" --comparison-operator "$6" \
    --treat-missing-data notBreaching --region "$REGION"
}

create_alarm "riskrate-oe-region-other-high" "Region Other O/E exceeds 1.5 (known value: 1.83, under-prediction)" "Region" "Other" 1.5 "GreaterThanThreshold"
create_alarm "riskrate-oe-region-r41-low" "Region R41 O/E below 0.6 (known value: 0.46, over-prediction)" "Region" "R41" 0.6 "LessThanThreshold"
create_alarm "riskrate-oe-region-r24-low" "Region R24 (largest region) O/E below 0.85 (known value: 0.76)" "Region" "R24" 0.85 "LessThanThreshold"
create_alarm "riskrate-oe-drivage-60-69-low" "DrivAge 60-69 O/E below 0.65 (known value: 0.57, over-prediction)" "DrivAgeBand" "60-69" 0.65 "LessThanThreshold"
create_alarm "riskrate-oe-drivage-70plus-high" "DrivAge 70+ O/E exceeds 1.2 (known value: 1.33, under-prediction)" "DrivAgeBand" "70+" 1.2 "GreaterThanThreshold"
create_alarm "riskrate-oe-drivage-40-49-low" "DrivAge 40-49 O/E below 0.8 (known value: 0.69, over-prediction)" "DrivAgeBand" "40-49" 0.8 "LessThanThreshold"

echo "== 6. CloudWatch dashboard =="
aws cloudwatch put-dashboard \
  --dashboard-name RiskRate-Monitoring \
  --dashboard-body file://"$SCRIPT_DIR/dashboard.json" \
  --region "$REGION"

echo "Done. Dashboard: https://${REGION}.console.aws.amazon.com/cloudwatch/home?region=${REGION}#dashboards/dashboard/RiskRate-Monitoring"
