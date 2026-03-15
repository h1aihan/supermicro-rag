# Supermicro RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers questions about Supermicro products, solutions, and documentation using 6000+ PDF documents and structured web content.

## Features

- **Hybrid search** — FAISS semantic search + BM25 keyword matching for high recall
- **LLM query planner** — classifies intent (list / detail / compare / recommend / faq / general) and routes to catalog or RAG accordingly
- **FAQ question bank** — question-to-question cosine matching for eStore FAQ, combined with source-filtered hybrid search
- **Product catalog integration** — structured product listing from `data/pages/products.jsonl`
- **Local embeddings** with sentence-transformers (no API cost for search)
- **Anthropic Claude or OpenAI** for answer generation with source citations
- **Multi-turn conversation** — follow-up detection with context carry-over
- **Web UI** and REST API via FastAPI
- **Docker-ready** for AWS EC2 deployment via ECR

## Quick Start

```bash
# 1. Install dependencies
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Configure API keys
cp .env.example .env
# Edit .env — set ANTHROPIC_API_KEY or OPENAI_API_KEY

# 3. Build the index (PDFs + web pages)
python setup_rag.py                    # both PDFs and data/pages
# python setup_rag.py --source pages   # only web content
# python setup_rag.py --source pdf     # only PDFs

# 4. Run
python src/chatbot.py --interactive          # CLI mode
uvicorn src.server:app --host 0.0.0.0 --port 8000  # Web UI at http://localhost:8000
```

## Testing

```bash
python tests/test_product_queries.py                    # run all tests
python tests/test_product_queries.py --summary          # with quality hints
python tests/test_product_queries.py --category detail   # one category
python tests/test_product_queries.py --id rec_8gpu_h200  # single test
python tests/test_product_queries.py --dry-run           # list queries only
python tests/test_product_queries.py --output results.txt --model claude-sonnet-4-5
```

Categories: `list`, `detail`, `compare`, `general`, `conversational`, `recommendation`, `misspell`, `multi`, `followup`, `faq`.

## Architecture

```
PDFs (pdfs/) + Web pages (data/pages/*.jsonl)
       |
   Text extraction (pypdf / process_pages)
       |
   Chunking (1000 chars, 200 overlap)
       |
   Embeddings (all-MiniLM-L6-v2, local)
       |
   FAISS index + BM25 index
       |
   Query --> Query Planner (Haiku) --> Hybrid Retrieval --> Answer LLM (Opus/Sonnet/GPT)
```

## Project Structure

```
supermicro-rag/
├── src/
│   ├── server.py          # FastAPI web server + chat UI
│   ├── chatbot.py         # Main RAG chatbot (LLM calls, prompt assembly)
│   ├── query_planner.py   # Intent classification + search query generation
│   ├── query.py           # Hybrid retrieval (FAISS + BM25) + FAQ question bank
│   ├── product_catalog.py # Structured product lookup from products.jsonl
│   ├── index.py           # FAISS/BM25 index + FAQ question bank (built at load)
│   ├── embed.py           # Embedding generation
│   ├── chunk.py           # Text chunking
│   ├── extract.py         # PDF text extraction
│   ├── entity_graph.py    # Lightweight entity-relationship graph for multi-hop
│   └── process_pages.py   # Web page JSONL → chunkable text
├── scripts/
│   ├── aws_push_ecr.sh    # Build + push Docker image to ECR
│   └── ingest_faq.py      # Convert estore_faq.jsonl → rag_content.jsonl
├── data/pages/            # products.jsonl, rag_content.jsonl (includes FAQ)
├── tests/
│   └── test_product_queries.py
├── static/index.html      # Chat UI
├── embeddings/faiss_index/ # FAISS index, BM25, metadata, entity graph (~1.5GB)
├── Dockerfile
├── requirements.txt
└── setup_rag.py
```

## Configuration

Set these in `.env` (see `.env.example`):

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `openai` | `openai`, `anthropic`, or `ollama` |
| `OPENAI_API_KEY` | — | Required if provider is `openai` |
| `LLM_MODEL` | `gpt-5.2` | OpenAI model name |
| `ANTHROPIC_API_KEY` | — | Required if provider is `anthropic` |
| `ANTHROPIC_MODEL` | `claude-opus-4-5` | Anthropic model for answer generation |
| `PLANNER_MODEL` | `claude-haiku-4-5` | Cheap model for query planning |
| `LLM_TEMPERATURE` | `0.5` | Sampling temperature (0 = deterministic) |
| `LLM_TOP_P` | `1.0` | Nucleus sampling (1.0 = disabled) |
| `TOP_K` | `10` | Chunks to retrieve per query |
| `INDEX_DIR` | `embeddings/faiss_index/` | Path to FAISS index |
| `FAISS_MMAP` | `1` | Memory-map index to reduce RAM usage |
| `ENABLE_RERANKING` | `1` | Cross-encoder reranking (0 = off, faster) |

## Deployment (AWS EC2 + Docker)

EC2 is used instead of serverless to handle the ~1.5GB FAISS index without startup timeouts.

### Prerequisites

- AWS CLI configured with ECR push permissions
- `.env` must contain `AWS_ACCOUNT_ID` and `AWS_REGION`
- EC2 instance running with Docker installed and an SSH key (e.g. `~/.ssh/supermicro-rag-key.pem`)

### Step 1 — Build and push Docker image to ECR

```bash
./scripts/aws_push_ecr.sh
```

The script reads `AWS_ACCOUNT_ID`, `AWS_REGION`, and `ECR_REPO` (default `supermicro-rag`) from `.env`, creates the ECR repository if needed, builds the image, and pushes it.

### Step 2 — Copy index and data files to EC2

Use wildcards to copy **contents** into the target directories (avoids nested `dir/dir/` problems):

```bash
EC2=ec2-user@<IP>
KEY=~/.ssh/supermicro-rag-key.pem

# Create target directories on EC2
ssh -i $KEY $EC2 "mkdir -p ~/embeddings/faiss_index ~/data/pages"

# Copy FAISS index + metadata
scp -i $KEY embeddings/faiss_index/* $EC2:~/embeddings/faiss_index/

# Copy page data (products.jsonl, rag_content.jsonl, etc.)
scp -i $KEY data/pages/* $EC2:~/data/pages/
```

### Step 3 — Pull image and run on EC2

SSH into the instance first, then pull and run:

```bash
ssh -i $KEY $EC2

# On EC2:
ECR_IMAGE=<ACCOUNT>.dkr.ecr.<REGION>.amazonaws.com/supermicro-rag:latest

# Authenticate Docker to ECR
aws ecr get-login-password --region <REGION> \
  | docker login --username AWS --password-stdin <ACCOUNT>.dkr.ecr.<REGION>.amazonaws.com

# Pull the latest image
docker pull $ECR_IMAGE

# Stop any existing container
docker stop supermicro-rag 2>/dev/null; docker rm supermicro-rag 2>/dev/null

# Run
docker run -d --name supermicro-rag -p 8000:8000 \
  -v ~/embeddings/faiss_index:/app/embeddings/faiss_index:ro \
  -v ~/data/pages:/app/data/pages:ro \
  -e ANTHROPIC_API_KEY="sk-ant-..." \
  -e LLM_PROVIDER=anthropic \
  -e ANTHROPIC_MODEL=claude-sonnet-4-5 \
  -e PLANNER_MODEL=claude-haiku-4-5 \
  -e LLM_TEMPERATURE=0.5 \
  -e LLM_TOP_P=1.0 \
  -e INDEX_DIR=/app/embeddings/faiss_index \
  -e PRODUCTS_FILE=/app/data/pages/products.jsonl \
  -e TOP_K=15 \
  -e FAISS_MMAP=1 \
  -e ENABLE_RERANKING=1 \
  $ECR_IMAGE
```

### Step 4 — Verify

```bash
# Check container logs (FAQ question bank should load on startup)
docker logs -f supermicro-rag

# Health check
curl http://localhost:8000/health
```

Access: `http://<IP>:8000` (Web UI) | `POST /api/chat` (API) | `GET /health`

## Troubleshooting

- **"FAISS index not found"** — Run `python setup_rag.py` or check `INDEX_DIR` path and volume mounts.
- **Slow first request** — The FAISS index + sentence-transformers model download takes 1-2 min on first cold start.
- **Out of memory** — Use `t3.large` (8GB) instead of `t3.medium` (4GB).
- **FAQ not working** — Check for `[FAQ Bank] Built question bank: N questions` in startup logs. If missing, ensure `rag_content.jsonl` was ingested with FAQ entries via `scripts/ingest_faq.py` and the index was rebuilt.
- **Nested directory on SCP** — Use `scp files/* host:~/target/` (not `scp -r dir host:~/`) to copy contents flat.

## License

MIT
