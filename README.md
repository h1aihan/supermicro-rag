# Supermicro RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers questions about Supermicro products, solutions, and documentation using 6000+ PDF documents.

## Live Demo

**Currently hosted on AWS EC2**: The chatbot is deployed and running in the cloud. Contact for access to the live instance.

## Features

- **PDFs + web pages** — Indexes Supermicro PDFs and web content from `data/pages/` (e.g. `products.jsonl`, `rag_content.jsonl`)
- **Semantic search** using FAISS vector index (~1.5GB)
- **Local embeddings** with sentence-transformers (no API cost for search)
- **Anthropic Claude or OpenAI** for answer generation with source citations (`LLM_PROVIDER=anthropic` or `openai`)
- **Web UI** and REST API endpoints
- **Docker-ready** for easy deployment

## Quick Start (Local)

```bash
# 1. Install dependencies
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Set up API key (use one of OpenAI or Anthropic)
cp .env.example .env
# For OpenAI: OPENAI_API_KEY=sk-...
# For Anthropic: ANTHROPIC_API_KEY=sk-ant-... and LLM_PROVIDER=anthropic (optional: ANTHROPIC_MODEL=claude-opus-4-5 or claude-opus-4-6)

# 3. Process PDFs and/or data/pages and create index (if not already done)
python setup_rag.py                    # default: both PDFs and data/pages
# python setup_rag.py --source pages    # only data/pages (products.jsonl, rag_content.jsonl)
# python setup_rag.py --source pdf      # only PDFs

# 4. Run chatbot (CLI)
python src/chatbot.py --interactive

# Or run web UI
uvicorn src.server:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000` for the web UI.

## Testing (product-specific queries)

After implementing query planning, you can run product-focused tests to check listing, specs, and comparisons:

```bash
# Run all product queries and print answers (see results in terminal)
python tests/test_product_queries.py

# Only list or detail queries
python tests/test_product_queries.py --category list
python tests/test_product_queries.py --category detail

# Single test by id
python tests/test_product_queries.py --id list_1u

# Dry run: print query list only
python tests/test_product_queries.py --dry-run

# Write results to a file for later review (same content as terminal)
python tests/test_product_queries.py --output test_results.txt
```

Optional: `pip install pytest` then `pytest tests/test_product_queries.py -v` to run assertions.

## Architecture

```
PDFs (pdfs/) + Web pages (data/pages/*.jsonl)
       ↓
   pypdf (PDFs) / process_pages (pages) → raw text
       ↓
   LangChain RecursiveCharacterTextSplitter (1000 chars, 200 overlap)
       ↓
   sentence-transformers/all-MiniLM-L6-v2 (embeddings, local/free)
       ↓
   FAISS vector index (~1.5GB)
       ↓
   Query → Retrieve top-K chunks → Claude or OpenAI (answer generation)
```

## Cloud Deployment (AWS EC2 + Docker)

**This is the current production setup.** We use EC2 instead of App Runner to handle the large FAISS index (~1.5GB) without health check timeouts during index loading.

### Prerequisites
- Docker installed locally
- AWS CLI configured (`aws configure`)
- `.env` file with `AWS_ACCOUNT_ID`, `AWS_REGION`, `REPO`, `APP_NAME`

### 1) Push image to ECR

```bash
./scripts/aws_push_ecr.sh
```

### 2) Launch EC2 instance

```bash
# Load vars
set -a; source .env; set +a

# Create security group (SSH + HTTP)
VPC_ID="$(aws ec2 describe-vpcs --region "$AWS_REGION" --filters Name=isDefault,Values=true --query 'Vpcs[0].VpcId' --output text)"
SG_ID="$(aws ec2 create-security-group --region "$AWS_REGION" --vpc-id "$VPC_ID" \
  --group-name "${APP_NAME}-ec2-sg" --description "EC2 for ${APP_NAME}" --query GroupId --output text)"
aws ec2 authorize-security-group-ingress --region "$AWS_REGION" --group-id "$SG_ID" --protocol tcp --port 22 --cidr 0.0.0.0/0
aws ec2 authorize-security-group-ingress --region "$AWS_REGION" --group-id "$SG_ID" --protocol tcp --port 8000 --cidr 0.0.0.0/0

# Create IAM role for ECR access
aws iam create-role --role-name "${APP_NAME}-ec2-role" --assume-role-policy-document '{
  "Version":"2012-10-17",
  "Statement":[{"Effect":"Allow","Principal":{"Service":"ec2.amazonaws.com"},"Action":"sts:AssumeRole"}]
}'
aws iam attach-role-policy --role-name "${APP_NAME}-ec2-role" --policy-arn arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly
aws iam create-instance-profile --instance-profile-name "${APP_NAME}-ec2-profile"
aws iam add-role-to-instance-profile --instance-profile-name "${APP_NAME}-ec2-profile" --role-name "${APP_NAME}-ec2-role"

# Create key pair
aws ec2 create-key-pair --region "$AWS_REGION" --key-name "${APP_NAME}-key" --query 'KeyMaterial' --output text > ~/.ssh/${APP_NAME}-key.pem
chmod 400 ~/.ssh/${APP_NAME}-key.pem

# Launch instance (t3.medium, 4GB RAM)
AMI_ID="$(aws ec2 describe-images --region "$AWS_REGION" --owners amazon \
  --filters "Name=name,Values=al2023-ami-2023*-x86_64" "Name=state,Values=available" \
  --query 'sort_by(Images, &CreationDate)[-1].ImageId' --output text)"

INSTANCE_ID="$(aws ec2 run-instances --region "$AWS_REGION" \
  --image-id "$AMI_ID" \
  --instance-type t3.medium \
  --key-name "${APP_NAME}-key" \
  --security-group-ids "$SG_ID" \
  --iam-instance-profile Name="${APP_NAME}-ec2-profile" \
  --block-device-mappings '[{"DeviceName":"/dev/xvda","Ebs":{"VolumeSize":30,"VolumeType":"gp3"}}]' \
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=${APP_NAME}}]" \
  --query 'Instances[0].InstanceId' --output text)"

# Wait and get IP
aws ec2 wait instance-running --region "$AWS_REGION" --instance-ids "$INSTANCE_ID"
PUBLIC_IP="$(aws ec2 describe-instances --region "$AWS_REGION" --instance-ids "$INSTANCE_ID" \
  --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)"
echo "Public IP: $PUBLIC_IP"
```

### 3) Copy FAISS index + pages data to EC2

```bash
scp -i ~/.ssh/${APP_NAME}-key.pem -r embeddings/faiss_index ec2-user@${PUBLIC_IP}:~/faiss_index
scp -i ~/.ssh/${APP_NAME}-key.pem -r data/pages ec2-user@${PUBLIC_IP}:~/pages
```

### 4) SSH and run container

```bash
ssh -i ~/.ssh/${APP_NAME}-key.pem ec2-user@${PUBLIC_IP}

# On EC2:
sudo dnf install -y docker
sudo systemctl enable --now docker
sudo usermod -aG docker ec2-user
newgrp docker

# Login to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <AWS_ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com

# Pull and run
docker pull <AWS_ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/supermicro-rag:latest

# Using OpenAI:
docker run -d --name supermicro-rag \
  -p 8000:8000 \
  -v ~/faiss_index:/app/embeddings/faiss_index:ro \
  -v ~/pages:/app/data/pages:ro \
  -e OPENAI_API_KEY="sk-your-key" \
  -e LLM_PROVIDER=openai \
  -e LLM_MODEL=gpt-5.2 \
  -e INDEX_DIR=/app/embeddings/faiss_index \
  -e PRODUCTS_FILE=/app/data/pages/products.jsonl \
  -e TOP_K=15 \
  <AWS_ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/supermicro-rag:latest

# Using Anthropic (set these instead of OPENAI_API_KEY / LLM_MODEL):
#  -e ANTHROPIC_API_KEY="sk-ant-..." \
#  -e LLM_PROVIDER=anthropic \
#  -e ANTHROPIC_MODEL=claude-opus-4-5 \

# Auto-restart on reboot
docker update --restart unless-stopped supermicro-rag
```

### 5) Access

- **Web UI**: `http://<PUBLIC_IP>:8000`
- **Health check**: `http://<PUBLIC_IP>:8000/health`
- **API**: `POST http://<PUBLIC_IP>:8000/api/chat` with `{"message":"..."}`

---

## Another deployment (second EC2 / staging)

Use a **different suffix** so security groups, keys, and instance names don’t clash with the first deployment.

### 1) Push image (same as first; skip if already pushed)

```bash
./scripts/aws_push_ecr.sh
```

### 2) Set suffix and load env

```bash
set -a; source .env; set +a
SUFFIX=2   # or staging, prod2, etc.
APP="${APP_NAME:-supermicro-rag}-${SUFFIX}"
```

### 3) Create resources for this deployment

```bash
VPC_ID="$(aws ec2 describe-vpcs --region "$AWS_REGION" --filters Name=isDefault,Values=true --query 'Vpcs[0].VpcId' --output text)"
SG_ID="$(aws ec2 create-security-group --region "$AWS_REGION" --vpc-id "$VPC_ID" \
  --group-name "${APP}-sg" --description "EC2 for ${APP}" --query GroupId --output text)"
aws ec2 authorize-security-group-ingress --region "$AWS_REGION" --group-id "$SG_ID" --protocol tcp --port 22 --cidr 0.0.0.0/0
aws ec2 authorize-security-group-ingress --region "$AWS_REGION" --group-id "$SG_ID" --protocol tcp --port 8000 --cidr 0.0.0.0/0

aws iam create-role --role-name "${APP}-role" --assume-role-policy-document '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"ec2.amazonaws.com"},"Action":"sts:AssumeRole"}]}' 2>/dev/null || true
aws iam attach-role-policy --role-name "${APP}-role" --policy-arn arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly
aws iam create-instance-profile --instance-profile-name "${APP}-profile" 2>/dev/null || true
aws iam add-role-to-instance-profile --instance-profile-name "${APP}-profile" --role-name "${APP}-role" 2>/dev/null || true

aws ec2 create-key-pair --region "$AWS_REGION" --key-name "${APP}-key" --query 'KeyMaterial' --output text > ~/.ssh/${APP}-key.pem
chmod 400 ~/.ssh/${APP}-key.pem
```

### 4) Launch instance and get IP

```bash
AMI_ID="$(aws ec2 describe-images --region "$AWS_REGION" --owners amazon \
  --filters "Name=name,Values=al2023-ami-2023*-x86_64" "Name=state,Values=available" \
  --query 'sort_by(Images, &CreationDate)[-1].ImageId' --output text)"
INSTANCE_ID="$(aws ec2 run-instances --region "$AWS_REGION" \
  --image-id "$AMI_ID" --instance-type t3.medium --key-name "${APP}-key" \
  --security-group-ids "$SG_ID" --iam-instance-profile Name="${APP}-profile" \
  --block-device-mappings '[{"DeviceName":"/dev/xvda","Ebs":{"VolumeSize":30,"VolumeType":"gp3"}}]' \
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=${APP}}]" \
  --query 'Instances[0].InstanceId' --output text)"
aws ec2 wait instance-running --region "$AWS_REGION" --instance-ids "$INSTANCE_ID"
PUBLIC_IP="$(aws ec2 describe-instances --region "$AWS_REGION" --instance-ids "$INSTANCE_ID" \
  --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)"
echo "Public IP: $PUBLIC_IP"
```

### 5) Copy FAISS index + pages data and run container

```bash
scp -i ~/.ssh/${APP}-key.pem -r embeddings/faiss_index ec2-user@${PUBLIC_IP}:~/faiss_index
scp -i ~/.ssh/${APP}-key.pem -r data/pages ec2-user@${PUBLIC_IP}:~/pages

ssh -i ~/.ssh/${APP}-key.pem ec2-user@${PUBLIC_IP}
```

On the EC2 instance (replace `ACCOUNT` and `REGION` with your `.env` values, e.g. `123456789012` and `us-east-1`):

```bash
sudo dnf install -y docker && sudo systemctl enable --now docker && sudo usermod -aG docker ec2-user && newgrp docker

aws ecr get-login-password --region REGION | sudo docker login --username AWS --password-stdin ACCOUNT.dkr.ecr.REGION.amazonaws.com

sudo docker pull ACCOUNT.dkr.ecr.REGION.amazonaws.com/supermicro-rag:latest

# Use either OpenAI or Anthropic env vars (see first deployment example above).
sudo docker run -d --name supermicro-rag-2 \
  -p 8000:8000 \
  -v ~/faiss_index:/app/embeddings/faiss_index:ro \
  -v ~/pages:/app/data/pages:ro \
  -e ANTHROPIC_API_KEY="sk-ant-..." \
  -e LLM_PROVIDER=anthropic \
  -e ANTHROPIC_MODEL=claude-opus-4-5 \
  -e INDEX_DIR=/app/embeddings/faiss_index \
  -e PRODUCTS_FILE=/app/data/pages/products.jsonl \
  -e TOP_K=15 \
  ACCOUNT.dkr.ecr.REGION.amazonaws.com/supermicro-rag:latest

sudo docker update --restart unless-stopped supermicro-rag-2
```

Access: **http://&lt;PUBLIC_IP&gt;:8000**

## Project Structure

```
supermicro-rag/
├── src/
│   ├── server.py       # FastAPI web server
│   ├── chatbot.py      # RAG chatbot logic
│   ├── query_planner.py # LLM query planning (list/detail/compare/general)
│   ├── product_catalog.py # Structured product listing (reads data/pages/products.jsonl)
│   ├── index.py        # FAISS index wrapper
│   ├── query.py        # Query processing
│   ├── extract.py      # PDF text extraction
│   ├── chunk.py        # Text chunking
│   ├── process_pages.py # Web page content → chunkable format (data/pages → data/raw_pages)
│   └── embed.py        # Embedding generation
├── data/
│   └── pages/          # Web data for RAG: products.jsonl, rag_content.jsonl (used by setup_rag.py --source pages|both)
├── tests/
│   └── test_product_queries.py  # Product listing/specs/compare tests
├── static/
│   └── index.html      # Chat UI
├── embeddings/
│   └── faiss_index/    # FAISS index + metadata (~1.5GB)
├── scripts/
│   └── aws_push_ecr.sh # Push Docker image to ECR
├── Dockerfile
├── requirements.txt
└── setup_rag.py        # Setup pipeline (--source pdf|pages|both)
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `openai` | LLM provider: `openai`, `anthropic`, or `ollama` |
| `OPENAI_API_KEY` | — | OpenAI API key (required if `LLM_PROVIDER=openai`) |
| `LLM_MODEL` | `gpt-5.2` | OpenAI model name (ignored when using Anthropic) |
| `ANTHROPIC_API_KEY` | — | Anthropic API key (required if `LLM_PROVIDER=anthropic`) |
| `ANTHROPIC_MODEL` | `claude-opus-4-5` | Anthropic model (e.g. `claude-opus-4-5`, `claude-opus-4-6`) |
| `INDEX_DIR` | `embeddings/faiss_index/` | Path to FAISS index |
| `TOP_K` | `5` | Number of chunks to retrieve (use 10-15 for broad questions) |
| `FAISS_MMAP` | `1` | Memory-map FAISS index (reduces RAM usage) |

## Tips for Best Results

- **Ask specific questions** — "What GPU servers support NVIDIA HGX H100?" works better than "What are Supermicro's products?"
- **Increase TOP_K** for broad questions (set `TOP_K=15` in env vars)
- **Check sources** — the chatbot cites which PDFs it used

## Troubleshooting

### "FAISS index not found"
Run `python setup_rag.py` to create the index, or ensure `INDEX_DIR` points to the correct path.

### Slow first request
The first request loads the 1.5GB FAISS index + downloads the embedding model. This can take 1-2 minutes. Subsequent requests are fast.

### Out of memory on EC2
Use `t3.large` (8GB RAM) instead of `t3.medium` for large indexes.

## Cost Estimate (AWS)

| Resource | Cost |
|----------|------|
| t3.medium EC2 | ~$1/day |
| 30GB EBS | ~$2.40/month |
| ECR storage | ~$0.10/GB/month |

## License

MIT
