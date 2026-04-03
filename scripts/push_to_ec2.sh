#!/usr/bin/env bash
set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# Push data, FAISS indexes, and deploy scripts to a fresh EC2 instance.
#
# Usage:
#   ./scripts/push_to_ec2.sh <EC2_HOST> [SSH_KEY]
#
# Examples:
#   ./scripts/push_to_ec2.sh ec2-user@3.95.12.100
#   ./scripts/push_to_ec2.sh ec2-user@3.95.12.100 ~/.ssh/supermicro-rag-key.pem
# ─────────────────────────────────────────────────────────────────────────────

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -f "$REPO_ROOT/.env" ]]; then
  set -a; source "$REPO_ROOT/.env"; set +a
fi

EC2="${1:?Usage: $0 <user@host> [ssh-key-path]}"
KEY="${2:-${SSH_KEY:-$HOME/.ssh/supermicro-rag-key.pem}}"
SSH_OPTS="-i $KEY -o StrictHostKeyChecking=no"

log() { echo -e "\033[0;36m[push]\033[0m $*"; }
ok()  { echo -e "\033[0;32m[done]\033[0m $*"; }
err() { echo -e "\033[0;31m[err]\033[0m $*" >&2; }

if [[ ! -f "$KEY" ]]; then
  err "SSH key not found: $KEY"
  exit 1
fi

log "Creating directories on EC2..."
ssh $SSH_OPTS "$EC2" "mkdir -p ~/embeddings/primary_index ~/embeddings/manual_index ~/data/pages ~/scripts"

# ── FAISS indexes (needed for Qdrant migration on first deploy) ──────────
if [[ -d "$REPO_ROOT/embeddings/primary_index" ]]; then
  log "Uploading primary index (FAISS + entity graph)..."
  scp $SSH_OPTS "$REPO_ROOT"/embeddings/primary_index/* "$EC2":~/embeddings/primary_index/
  ok "primary_index uploaded"
else
  err "embeddings/primary_index/ not found — Qdrant migration will skip primary"
fi

if [[ -d "$REPO_ROOT/embeddings/manual_index" ]]; then
  log "Uploading manual index..."
  scp $SSH_OPTS "$REPO_ROOT"/embeddings/manual_index/* "$EC2":~/embeddings/manual_index/
  ok "manual_index uploaded"
else
  err "embeddings/manual_index/ not found — Qdrant migration will skip manual"
fi

# ── Data files (mounted into the running container) ──────────────────────
if [[ -d "$REPO_ROOT/data/pages" ]]; then
  log "Uploading data/pages/..."
  scp $SSH_OPTS "$REPO_ROOT"/data/pages/* "$EC2":~/data/pages/
  ok "data/pages uploaded"
fi

if [[ -f "$REPO_ROOT/data/discovered_pdfs.txt" ]]; then
  log "Uploading data/discovered_pdfs.txt..."
  scp $SSH_OPTS "$REPO_ROOT/data/discovered_pdfs.txt" "$EC2":~/data/
  ok "discovered_pdfs.txt uploaded"
fi

# ── Migration script (runs inside Docker container on EC2) ───────────────
log "Uploading migration script + deploy script..."
scp $SSH_OPTS "$REPO_ROOT/scripts/migrate_to_qdrant.py" "$EC2":~/scripts/
scp $SSH_OPTS "$REPO_ROOT/scripts/deploy_ec2.sh" "$EC2":~/scripts/
ssh $SSH_OPTS "$EC2" "chmod +x ~/scripts/deploy_ec2.sh"
ok "scripts uploaded"

# ── Verify ───────────────────────────────────────────────────────────────
log "Verifying files on EC2..."
ssh $SSH_OPTS "$EC2" bash -c "'
echo \"=== embeddings/primary_index ===\"
ls -lh ~/embeddings/primary_index/ 2>/dev/null || echo \"(empty)\"
echo \"=== embeddings/manual_index ===\"
ls -lh ~/embeddings/manual_index/ 2>/dev/null || echo \"(empty)\"
echo \"=== data/pages ===\"
ls -lh ~/data/pages/ 2>/dev/null || echo \"(empty)\"
echo \"=== data/discovered_pdfs.txt ===\"
ls -lh ~/data/discovered_pdfs.txt 2>/dev/null || echo \"(missing)\"
echo \"=== scripts ===\"
ls -lh ~/scripts/ 2>/dev/null || echo \"(empty)\"
'"

echo ""
ok "All files pushed to $EC2"
echo ""
echo "Next steps on EC2:"
echo "  1. Install Docker if needed (see setup commands below)"
echo "  2. Export required env vars:"
echo "     export AWS_ACCOUNT_ID=${AWS_ACCOUNT_ID:-660341453849}"
echo "     export AWS_REGION=${AWS_REGION:-us-east-1}"
echo "     export ANTHROPIC_API_KEY='your-key-here'"
echo "  3. Run: ~/scripts/deploy_ec2.sh --migrate"
echo ""
echo "Docker install (Amazon Linux 2023):"
echo "  sudo yum install -y docker && sudo systemctl enable --now docker"
echo "  sudo usermod -aG docker \$USER && newgrp docker"
