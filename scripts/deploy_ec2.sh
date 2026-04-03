#!/usr/bin/env bash
set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# Deploy supermicro-rag on EC2 with Qdrant.
#
# Usage (run ON the EC2 instance):
#   ~/scripts/deploy_ec2.sh              # Pull image, start Qdrant + app
#   ~/scripts/deploy_ec2.sh --migrate    # Also migrate FAISS indexes into Qdrant
#
# Required env vars:
#   AWS_ACCOUNT_ID, AWS_REGION, ANTHROPIC_API_KEY
#
# The --migrate flag is for first-time deployment: it spins up Qdrant, then
# runs migrate_to_qdrant.py inside the app container to populate the
# collections from the FAISS files in ~/embeddings/.
# ─────────────────────────────────────────────────────────────────────────────

DO_MIGRATE=0
if [[ "${1:-}" == "--migrate" ]]; then
  DO_MIGRATE=1
fi

ECR_REGISTRY="${AWS_ACCOUNT_ID:?Set AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION:?Set AWS_REGION}.amazonaws.com"
ECR_IMAGE="${ECR_REGISTRY}/${ECR_REPO:-supermicro-rag}:${IMAGE_TAG:-latest}"
CONTAINER_NAME="supermicro-rag"
QDRANT_NAME="qdrant"

# ── ECR auth + pull ──────────────────────────────────────────────────────
echo "[deploy] Authenticating to ECR..."
aws ecr get-login-password --region "$AWS_REGION" \
  | docker login --username AWS --password-stdin "$ECR_REGISTRY"

echo "[deploy] Pulling latest image..."
docker pull "$ECR_IMAGE"

# ── Stop old app container ───────────────────────────────────────────────
echo "[deploy] Stopping old app container (if any)..."
docker stop "$CONTAINER_NAME" 2>/dev/null || true
docker rm "$CONTAINER_NAME" 2>/dev/null || true

# ── Memory budget ────────────────────────────────────────────────────────
TOTAL_MEM_MB=$(free -m | awk '/^Mem:/{print $2}')
CONTAINER_MEM_MB=$(( TOTAL_MEM_MB - 512 ))
echo "[deploy] Host RAM: ${TOTAL_MEM_MB}MB, container limit: ${CONTAINER_MEM_MB}MB"

# ── Start Qdrant (idempotent) ────────────────────────────────────────────
if ! docker ps --format '{{.Names}}' | grep -q "^${QDRANT_NAME}$"; then
  echo "[deploy] Starting Qdrant..."
  docker stop "$QDRANT_NAME" 2>/dev/null || true
  docker rm "$QDRANT_NAME" 2>/dev/null || true
  docker run -d --name "$QDRANT_NAME" -p 6333:6333 \
    -v ~/qdrant_data:/qdrant/storage \
    qdrant/qdrant:latest
  echo "[deploy] Waiting for Qdrant to be ready..."
  for i in $(seq 1 30); do
    if curl -sf http://localhost:6333/healthz >/dev/null 2>&1; then
      echo "[deploy] Qdrant is ready."
      break
    fi
    sleep 1
  done
fi

# ── Qdrant migration (first-time only) ──────────────────────────────────
if [[ "$DO_MIGRATE" -eq 1 ]]; then
  echo "[deploy] Running FAISS → Qdrant migration..."

  if [[ ! -f ~/embeddings/primary_index/faiss.index ]] && [[ ! -f ~/embeddings/manual_index/faiss.index ]]; then
    echo "[deploy] ERROR: No FAISS indexes found in ~/embeddings/."
    echo "         Run push_to_ec2.sh first to upload them."
    exit 1
  fi

  docker run --rm \
    --link "$QDRANT_NAME":qdrant \
    -v ~/embeddings:/app/embeddings:ro \
    -v ~/scripts/migrate_to_qdrant.py:/app/scripts/migrate_to_qdrant.py:ro \
    -e QDRANT_URL=http://qdrant:6333 \
    -e QDRANT_COLLECTION_PRIMARY="${QDRANT_COLLECTION_PRIMARY:-supermicro_primary}" \
    -e QDRANT_COLLECTION_MANUAL="${QDRANT_COLLECTION_MANUAL:-supermicro_manual}" \
    "$ECR_IMAGE" \
    python /app/scripts/migrate_to_qdrant.py \
      --primary-dir /app/embeddings/primary_index/ \
      --manual-dir /app/embeddings/manual_index/ \
      --qdrant-url http://qdrant:6333

  echo "[deploy] Migration complete. Verifying collections..."
  curl -s http://localhost:6333/collections | python3 -m json.tool 2>/dev/null || true
fi

# ── Start the app ────────────────────────────────────────────────────────
echo "[deploy] Starting app container..."
docker run -d --name "$CONTAINER_NAME" -p 8000:8000 \
  --memory="${CONTAINER_MEM_MB}m" \
  --memory-swap="${CONTAINER_MEM_MB}m" \
  --oom-kill-disable=false \
  --link "$QDRANT_NAME":qdrant \
  -v ~/data/pages:/app/data/pages:ro \
  -v ~/data/discovered_pdfs.txt:/app/data/discovered_pdfs.txt:ro \
  -v ~/embeddings/primary_index/entity_graph.json:/app/embeddings/primary_index/entity_graph.json:ro \
  -e ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:?Set ANTHROPIC_API_KEY}" \
  -e LLM_PROVIDER="${LLM_PROVIDER:-anthropic}" \
  -e ANTHROPIC_MODEL="${ANTHROPIC_MODEL:-claude-sonnet-4-5}" \
  -e PLANNER_MODEL="${PLANNER_MODEL:-claude-haiku-4-5}" \
  -e LLM_TEMPERATURE="${LLM_TEMPERATURE:-0.1}" \
  -e LLM_TOP_P="${LLM_TOP_P:-1.0}" \
  -e QDRANT_URL=http://qdrant:6333 \
  -e QDRANT_COLLECTION_PRIMARY="${QDRANT_COLLECTION_PRIMARY:-supermicro_primary}" \
  -e QDRANT_COLLECTION_MANUAL="${QDRANT_COLLECTION_MANUAL:-supermicro_manual}" \
  -e PRODUCTS_FILE=/app/data/pages/products.jsonl \
  -e TOP_K="${TOP_K:-15}" \
  -e ENABLE_RERANKING="${ENABLE_RERANKING:-0}" \
  "$ECR_IMAGE"

echo ""
echo "[deploy] Container started. Waiting for health check..."
for i in $(seq 1 60); do
  STATUS=$(curl -sf http://localhost:8000/health 2>/dev/null || true)
  if [[ -n "$STATUS" ]]; then
    echo "[deploy] Health check passed:"
    echo "$STATUS" | python3 -m json.tool 2>/dev/null || echo "$STATUS"
    break
  fi
  if [[ $i -eq 60 ]]; then
    echo "[deploy] WARNING: Health check not responding after 60s. Check logs:"
  fi
  sleep 2
done

echo ""
echo "[deploy] Tailing logs (Ctrl+C to stop)..."
docker logs -f "$CONTAINER_NAME"
