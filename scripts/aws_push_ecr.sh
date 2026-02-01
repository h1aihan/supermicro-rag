#!/usr/bin/env bash
set -euo pipefail

# Build + push the Docker image to AWS ECR.
# Reads configuration from repo-root .env automatically.

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -f "$REPO_ROOT/.env" ]]; then
  # Export all variables loaded from .env
  set -a
  # shellcheck disable=SC1091
  source "$REPO_ROOT/.env"
  set +a
fi

command -v aws >/dev/null 2>&1 || { echo "Missing 'aws' CLI. Install/configure AWS CLI first."; exit 1; }
command -v docker >/dev/null 2>&1 || { echo "Missing 'docker'. Install Docker first."; exit 1; }

: "${AWS_ACCOUNT_ID:?Set AWS_ACCOUNT_ID in .env (e.g. 123456789012)}"
: "${AWS_REGION:?Set AWS_REGION in .env (e.g. us-east-1)}"

ECR_REPO="${ECR_REPO:-supermicro-rag}"
IMAGE_TAG="${IMAGE_TAG:-latest}"

ECR_REGISTRY="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
ECR_IMAGE="${ECR_REGISTRY}/${ECR_REPO}:${IMAGE_TAG}"

# Create the ECR repo if it doesn't exist (idempotent).
if ! aws ecr describe-repositories --repository-names "$ECR_REPO" --region "$AWS_REGION" >/dev/null 2>&1; then
  aws ecr create-repository --repository-name "$ECR_REPO" --region "$AWS_REGION" >/dev/null
fi

aws ecr get-login-password --region "$AWS_REGION" \
  | docker login --username AWS --password-stdin "$ECR_REGISTRY"

# Build (context is repo root so Dockerfile is found reliably).
if [[ "${USE_BUILDX:-0}" == "1" ]]; then
  docker buildx build -t "${ECR_REPO}:${IMAGE_TAG}" "$REPO_ROOT"
else
  docker build -t "${ECR_REPO}:${IMAGE_TAG}" "$REPO_ROOT"
fi

docker tag "${ECR_REPO}:${IMAGE_TAG}" "$ECR_IMAGE"
docker push "$ECR_IMAGE"

echo "Pushed: $ECR_IMAGE"
