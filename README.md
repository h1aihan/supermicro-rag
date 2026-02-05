# Supermicro RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers questions about Supermicro products, solutions, and documentation using 6000+ PDF documents.

## Live Demo

**Currently hosted on AWS EC2**: The chatbot is deployed and running in the cloud. Contact for access to the live instance.

## Features

- **6000+ PDF documents** indexed from Supermicro's product documentation
- **Semantic search** using FAISS vector index (~1.5GB)
- **Local embeddings** with sentence-transformers (no API cost for search)
- **OpenAI GPT** for answer generation with source citations
- **Web UI** and REST API endpoints
- **Docker-ready** for easy deployment

## Quick Start (Local)

```bash
# 1. Install dependencies
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Set up OpenAI API key
cp .env.example .env
# Edit .env and add: OPENAI_API_KEY=sk-your-key-here

# 3. Process PDFs and create index (if not already done)
python setup_rag.py

# 4. Run chatbot (CLI)
python src/chatbot.py --interactive

# Or run web UI
uvicorn src.server:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000` for the web UI.

## Architecture

```
PDF Documents (6000+)
       ↓
   pypdf (extract text)
       ↓
   LangChain RecursiveCharacterTextSplitter (1000 chars, 200 overlap)
       ↓
   sentence-transformers/all-MiniLM-L6-v2 (embeddings, local/free)
       ↓
   FAISS vector index (~1.5GB)
       ↓
   Query → Retrieve top-K chunks → OpenAI GPT (answer generation)
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

### 3) Copy FAISS index to EC2

```bash
scp -i ~/.ssh/${APP_NAME}-key.pem -r embeddings/faiss_index ec2-user@${PUBLIC_IP}:~/faiss_index
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

docker run -d --name supermicro-rag \
  -p 8000:8000 \
  -v ~/faiss_index:/app/embeddings/faiss_index:ro \
  -e OPENAI_API_KEY="sk-your-key" \
  -e LLM_PROVIDER=openai \
  -e LLM_MODEL=gpt-5.2 \
  -e INDEX_DIR=/app/embeddings/faiss_index \
  -e TOP_K=15 \
  <AWS_ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/supermicro-rag:latest

# Auto-restart on reboot
docker update --restart unless-stopped supermicro-rag
```

### 5) Access

- **Web UI**: `http://<PUBLIC_IP>:8000`
- **Health check**: `http://<PUBLIC_IP>:8000/health`
- **API**: `POST http://<PUBLIC_IP>:8000/api/chat` with `{"message":"..."}`

## Project Structure

```
supermicro-rag/
├── src/
│   ├── server.py       # FastAPI web server
│   ├── chatbot.py      # RAG chatbot logic
│   ├── index.py        # FAISS index wrapper
│   ├── query.py        # Query processing
│   ├── extract.py      # PDF text extraction
│   ├── chunk.py        # Text chunking
│   └── embed.py        # Embedding generation
├── static/
│   └── index.html      # Chat UI
├── embeddings/
│   └── faiss_index/    # FAISS index + metadata (~1.5GB)
├── scripts/
│   └── aws_push_ecr.sh # Push Docker image to ECR
├── Dockerfile
├── requirements.txt
└── setup_rag.py        # One-command setup pipeline
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | (required) | OpenAI API key |
| `LLM_PROVIDER` | `openai` | LLM provider |
| `LLM_MODEL` | `gpt-5.2` | Model name |
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
