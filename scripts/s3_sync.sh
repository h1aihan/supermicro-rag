#!/usr/bin/env bash
set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# S3 sync script for supermicro-rag project
#
# Usage:
#   ./scripts/s3_sync.sh push          # Upload data + embeddings to S3
#   ./scripts/s3_sync.sh pull          # Download data + embeddings from S3
#   ./scripts/s3_sync.sh push --all    # Upload everything (code + data)
#   ./scripts/s3_sync.sh pull --all    # Download everything (code + data)
#   ./scripts/s3_sync.sh status        # Show what would be synced
#
# Prerequisites:
#   - AWS CLI v2 installed (pip install awscli or brew install awscli)
#   - AWS credentials configured (aws configure)
#
# First-time setup:
#   1. Create a bucket:  aws s3 mb s3://YOUR-BUCKET-NAME
#   2. Set the bucket below or export S3_BUCKET=your-bucket-name
# ─────────────────────────────────────────────────────────────────────────────

S3_BUCKET="${S3_BUCKET:-supermicro-rag-sync}"
S3_PREFIX="${S3_PREFIX:-supermicro-rag}"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

S3_BASE="s3://${S3_BUCKET}/${S3_PREFIX}"

CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log()  { echo -e "${CYAN}[sync]${NC} $*"; }
ok()   { echo -e "${GREEN}[done]${NC} $*"; }
warn() { echo -e "${YELLOW}[warn]${NC} $*"; }

check_aws() {
    if ! command -v aws &>/dev/null; then
        echo "Error: AWS CLI not found. Install with: pip install awscli"
        exit 1
    fi
    if ! aws sts get-caller-identity &>/dev/null; then
        echo "Error: AWS credentials not configured. Run: aws configure"
        exit 1
    fi
}

do_push() {
    local all="${1:-}"
    check_aws

    log "Pushing to ${S3_BASE}/ ..."

    if [[ "$all" == "--all" ]]; then
        log "Syncing source code..."
        aws s3 sync "$PROJECT_DIR/src/"     "${S3_BASE}/src/"     --delete
        aws s3 sync "$PROJECT_DIR/scripts/" "${S3_BASE}/scripts/" --delete --exclude "*.pyc"
        aws s3 sync "$PROJECT_DIR/static/"  "${S3_BASE}/static/"  --delete
        aws s3 sync "$PROJECT_DIR/tests/"   "${S3_BASE}/tests/"   --delete --exclude "__pycache__/*"
        aws s3 sync "$PROJECT_DIR/docs/"    "${S3_BASE}/docs/"    --delete

        for f in requirements.txt setup_rag.py Dockerfile .dockerignore README.md .env.example; do
            [[ -f "$PROJECT_DIR/$f" ]] && aws s3 cp "$PROJECT_DIR/$f" "${S3_BASE}/$f"
        done

        if [[ -f "$PROJECT_DIR/.env" ]]; then
            warn ".env contains secrets — uploading with restricted ACL"
            aws s3 cp "$PROJECT_DIR/.env" "${S3_BASE}/.env"
        fi
        ok "Source code synced"
    fi

    if [[ -d "$PROJECT_DIR/data" ]]; then
        log "Syncing data/ ..."
        aws s3 sync "$PROJECT_DIR/data/" "${S3_BASE}/data/" --delete
        ok "data/ synced"
    else
        warn "data/ not found, skipping"
    fi

    if [[ -d "$PROJECT_DIR/embeddings" ]]; then
        log "Syncing embeddings/ (entity_graph.json + legacy FAISS if present)..."
        aws s3 sync "$PROJECT_DIR/embeddings/" "${S3_BASE}/embeddings/" --delete
        ok "embeddings/ synced"
    else
        warn "embeddings/ not found, skipping"
    fi

    echo ""
    ok "Push complete → ${S3_BASE}/"
    echo "  On your other machine, run:"
    echo "    export S3_BUCKET=${S3_BUCKET}"
    echo "    ./scripts/s3_sync.sh pull --all"
}

do_pull() {
    local all="${1:-}"
    check_aws

    log "Pulling from ${S3_BASE}/ ..."

    if [[ "$all" == "--all" ]]; then
        log "Syncing source code..."
        aws s3 sync "${S3_BASE}/src/"     "$PROJECT_DIR/src/"     --delete
        aws s3 sync "${S3_BASE}/scripts/" "$PROJECT_DIR/scripts/" --delete
        aws s3 sync "${S3_BASE}/static/"  "$PROJECT_DIR/static/"  --delete
        aws s3 sync "${S3_BASE}/tests/"   "$PROJECT_DIR/tests/"   --delete
        aws s3 sync "${S3_BASE}/docs/"    "$PROJECT_DIR/docs/"    --delete

        for f in requirements.txt setup_rag.py Dockerfile .dockerignore README.md .env.example .env; do
            aws s3 cp "${S3_BASE}/$f" "$PROJECT_DIR/$f" 2>/dev/null || true
        done
        ok "Source code synced"
    fi

    log "Syncing data/ ..."
    mkdir -p "$PROJECT_DIR/data"
    aws s3 sync "${S3_BASE}/data/" "$PROJECT_DIR/data/" --delete
    ok "data/ synced"

    log "Syncing embeddings/ ..."
    mkdir -p "$PROJECT_DIR/embeddings"
    aws s3 sync "${S3_BASE}/embeddings/" "$PROJECT_DIR/embeddings/" --delete
    ok "embeddings/ synced"

    echo ""
    ok "Pull complete ← ${S3_BASE}/"
}

do_status() {
    check_aws

    echo "Bucket:  s3://${S3_BUCKET}"
    echo "Prefix:  ${S3_PREFIX}/"
    echo "Project: ${PROJECT_DIR}"
    echo ""

    log "Remote contents:"
    aws s3 ls "${S3_BASE}/" --summarize --human-readable 2>/dev/null || warn "Nothing in S3 yet"

    echo ""
    log "Local sizes:"
    for dir in data embeddings src scripts; do
        if [[ -d "$PROJECT_DIR/$dir" ]]; then
            size=$(du -sh "$PROJECT_DIR/$dir" 2>/dev/null | cut -f1)
            echo "  $dir/ → $size"
        else
            echo "  $dir/ → (not found)"
        fi
    done
}

case "${1:-help}" in
    push)   do_push "${2:-}" ;;
    pull)   do_pull "${2:-}" ;;
    status) do_status ;;
    *)
        echo "Usage: $0 {push|pull|status} [--all]"
        echo ""
        echo "  push          Upload data/ and embeddings/ to S3"
        echo "  push --all    Upload everything (code + data + embeddings)"
        echo "  pull          Download data/ and embeddings/ from S3"
        echo "  pull --all    Download everything"
        echo "  status        Show sync info"
        echo ""
        echo "Config (env vars):"
        echo "  S3_BUCKET  Bucket name  (default: supermicro-rag-sync)"
        echo "  S3_PREFIX  S3 key prefix (default: supermicro-rag)"
        exit 1
        ;;
esac
