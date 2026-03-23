#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="/tmp/deploy_output.txt"

set -a
source "$REPO_ROOT/.env"
set +a

echo "=== Listing App Runner services ===" > "$OUT"
aws apprunner list-services --region "$AWS_REGION" --output json >> "$OUT" 2>&1

SERVICE_ARN=$(aws apprunner list-services --region "$AWS_REGION" \
  --query 'ServiceSummaryList[0].ServiceArn' --output text 2>&1)
echo "SERVICE_ARN=$SERVICE_ARN" >> "$OUT"

echo "=== Starting deployment ===" >> "$OUT"
aws apprunner start-deployment --service-arn "$SERVICE_ARN" --region "$AWS_REGION" >> "$OUT" 2>&1

echo "=== Service status ===" >> "$OUT"
aws apprunner describe-service --service-arn "$SERVICE_ARN" --region "$AWS_REGION" \
  --query 'Service.{Status:Status,URL:ServiceUrl,Updated:UpdatedAt}' --output json >> "$OUT" 2>&1

echo "=== Done ===" >> "$OUT"
cat "$OUT"
