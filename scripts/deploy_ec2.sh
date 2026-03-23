#!/usr/bin/env bash
set -euo pipefail

# Deploy supermicro-rag on EC2.
# Usage: ssh into EC2, then run this script.
# Reads .env-style config from environment or uses defaults.

ECR_REGISTRY="${AWS_ACCOUNT_ID:-660341453849}.dkr.ecr.${AWS_REGION:-us-east-1}.amazonaws.com"
ECR_IMAGE="${ECR_REGISTRY}/${ECR_REPO:-supermicro-rag}:${IMAGE_TAG:-latest}"
CONTAINER_NAME="supermicro-rag"

echo "[deploy] Authenticating to ECR..."
aws ecr get-login-password --region "${AWS_REGION:-us-east-1}" \
  | docker login --username AWS --password-stdin "$ECR_REGISTRY"

echo "[deploy] Pulling latest image..."
docker pull "$ECR_IMAGE"

echo "[deploy] Stopping old container (if any)..."
docker stop "$CONTAINER_NAME" 2>/dev/null || true
docker rm "$CONTAINER_NAME" 2>/dev/null || true

# Reserve 512MB for the OS/SSH — container gets the rest
TOTAL_MEM_MB=$(free -m | awk '/^Mem:/{print $2}')
CONTAINER_MEM_MB=$(( TOTAL_MEM_MB - 512 ))

echo "[deploy] Host RAM: ${TOTAL_MEM_MB}MB, container limit: ${CONTAINER_MEM_MB}MB"

docker run -d --name "$CONTAINER_NAME" -p 8000:8000 \
  --memory="${CONTAINER_MEM_MB}m" \
  --memory-swap="${CONTAINER_MEM_MB}m" \
  --oom-kill-disable=false \
  -v ~/embeddings/primary_index:/app/embeddings/primary_index:ro \
  -v ~/embeddings/manual_index:/app/embeddings/manual_index:ro \
  -v ~/data/pages:/app/data/pages:ro \
  -v ~/data/discovered_pdfs.txt:/app/data/discovered_pdfs.txt:ro \
  -e ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:?Set ANTHROPIC_API_KEY}" \
  -e LLM_PROVIDER="${LLM_PROVIDER:-anthropic}" \
  -e ANTHROPIC_MODEL="${ANTHROPIC_MODEL:-claude-sonnet-4-5}" \
  -e PLANNER_MODEL="${PLANNER_MODEL:-claude-haiku-4-5}" \
  -e LLM_TEMPERATURE="${LLM_TEMPERATURE:-0.1}" \
  -e LLM_TOP_P="${LLM_TOP_P:-1.0}" \
  -e INDEX_DIR=/app/embeddings/primary_index \
  -e MANUAL_INDEX_DIR=/app/embeddings/manual_index \
  -e PRODUCTS_FILE=/app/data/pages/products.jsonl \
  -e TOP_K="${TOP_K:-15}" \
  -e FAISS_MMAP=1 \
  -e ENABLE_RERANKING="${ENABLE_RERANKING:-0}" \
  "$ECR_IMAGE"

echo "[deploy] Container started. Tailing logs (Ctrl+C to stop)..."
docker logs -f "$CONTAINER_NAME"
