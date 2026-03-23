# Supermicro RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers questions about Supermicro products, solutions, and documentation using 6000+ PDF documents and structured web content.

## Features

- **Hybrid search** — FAISS semantic search + BM25 keyword matching with Reciprocal Rank Fusion
- **Dual-index routing** — separate primary (datasheets, web pages, FAQ) and manual (user guides, QRGs) indices, queried based on intent
- **LLM query planner** — classifies intent (list / detail / compare / recommend / faq / general) and routes to catalog, manual index, or primary RAG
- **Entity graph** — lightweight entity-relationship graph over primary index metadata for multi-hop retrieval
- **FAQ question bank** — question-to-question cosine matching for eStore FAQ, combined with source-filtered hybrid search
- **Product catalog integration** — structured product listing from `data/pages/products.jsonl`
- **Document links** — deterministic manual PDF download links via `data/discovered_pdfs.txt` (crawler-discovered URLs), shown only for manual-type queries
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
   Split by doc type (setup_rag.py)
       |
   ┌───────────────────┬────────────────────────┐
   Primary index       Manual index             Entity graph
   (datasheets, web,   (MNL-*, QRG-*,          (primary metadata)
    FAQ, accessories)    user guides, SC*)
   └───────┬───────────┴────────┬───────────────┘
           |                    |
   Query ──> Query Planner (Haiku) ──> RoutedIndex ──> Hybrid Retrieval
           |                                               |
           |  Product Catalog (products.jsonl)              |
           └───────────────────────────────────────────────> Answer LLM
                                                            + doc links
```

## Project Structure

```
supermicro-rag/
├── src/
│   ├── server.py          # FastAPI web server + chat UI
│   ├── chatbot.py         # Main RAG chatbot (LLM calls, prompt assembly, doc links)
│   ├── query_planner.py   # Intent classification + search query generation
│   ├── query.py           # Hybrid retrieval (FAISS + BM25) + FAQ question bank
│   ├── product_catalog.py # Structured product lookup from products.jsonl
│   ├── index.py           # HybridIndex + RoutedIndex (primary/manual routing)
│   ├── embed.py           # Embedding generation
│   ├── chunk.py           # Text chunking
│   ├── extract.py         # PDF text extraction
│   ├── entity_graph.py    # Entity-relationship graph for multi-hop retrieval
│   └── process_pages.py   # Web page JSONL → chunkable text
├── scripts/
│   ├── aws_push_ecr.sh    # Build + push Docker image to ECR
│   ├── s3_sync.sh         # Push/pull data + embeddings to/from S3
│   ├── ingest_faq.py      # Convert estore_faq.jsonl → rag_content.jsonl
│   └── ingest_accessories.py
├── data/
│   ├── pages/             # products.jsonl, rag_content.jsonl (includes FAQ)
│   └── discovered_pdfs.txt # Crawler-discovered PDF URLs (for manual download links)
├── tests/
│   └── test_product_queries.py
├── static/index.html      # Chat UI
├── embeddings/
│   ├── primary_index/     # Datasheets, web pages, FAQ, accessories (~127K vectors)
│   └── manual_index/      # User guides, QRGs, installation manuals (~583K vectors)
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
| `INDEX_DIR` | `embeddings/primary_index/` | Path to primary FAISS index |
| `MANUAL_INDEX_DIR` | `embeddings/manual_index/` | Path to manual FAISS index |
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

Use wildcards to copy **contents** into the target directories (avoids nested `dir/dir/` problems).

Each index directory must contain: `faiss.index`, `bm25.pkl`, `metadata.jsonl`.
The primary index also contains `entity_graph.json` (multi-hop retrieval).

```bash
EC2=ec2-user@<IP>
KEY=~/.ssh/supermicro-rag-key.pem

# Create target directories on EC2
ssh -i $KEY $EC2 "mkdir -p ~/embeddings/primary_index ~/embeddings/manual_index ~/data/pages ~/data"

# Copy primary index (datasheets, web pages, FAQ, accessories + entity graph)
scp -i $KEY embeddings/primary_index/* $EC2:~/embeddings/primary_index/

# Copy manual index (user guides, QRGs, installation manuals)
scp -i $KEY embeddings/manual_index/* $EC2:~/embeddings/manual_index/

# Copy page data (products.jsonl, rag_content.jsonl, etc.)
scp -i $KEY data/pages/* $EC2:~/data/pages/

# Copy PDF URL map (enables manual download links in chat)
scp -i $KEY data/discovered_pdfs.txt $EC2:~/data/
```

Verify all files landed:

```bash
ssh -i $KEY $EC2 "ls -lh ~/embeddings/primary_index/ ~/embeddings/manual_index/ ~/data/pages/ ~/data/discovered_pdfs.txt"
```

Expected in `primary_index/`: `faiss.index`, `bm25.pkl`, `metadata.jsonl`, `entity_graph.json`.
Expected in `manual_index/`: `faiss.index`, `bm25.pkl`, `metadata.jsonl`.

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
  -v ~/embeddings/primary_index:/app/embeddings/primary_index:ro \
  -v ~/embeddings/manual_index:/app/embeddings/manual_index:ro \
  -v ~/data/pages:/app/data/pages:ro \
  -v ~/data/discovered_pdfs.txt:/app/data/discovered_pdfs.txt:ro \
  -e ANTHROPIC_API_KEY="sk-ant-..." \
  -e LLM_PROVIDER=anthropic \
  -e ANTHROPIC_MODEL=claude-sonnet-4-5 \
  -e PLANNER_MODEL=claude-haiku-4-5 \
  -e LLM_TEMPERATURE=0.5 \
  -e LLM_TOP_P=1.0 \
  -e INDEX_DIR=/app/embeddings/primary_index \
  -e MANUAL_INDEX_DIR=/app/embeddings/manual_index \
  -e PRODUCTS_FILE=/app/data/pages/products.jsonl \
  -e TOP_K=15 \
  -e FAISS_MMAP=1 \
  -e ENABLE_RERANKING=1 \
  $ECR_IMAGE
```

### Step 4 — Verify

```bash
# Check container logs
docker logs -f supermicro-rag

# Health check
curl http://localhost:8000/health
```

Startup logs should show all components loading:

```
[RoutedIndex] Primary index: 127,xxx vectors
[RoutedIndex] Manual index: 583,xxx vectors
[EntityGraph] Loaded graph: X,xxx entities
[FAQ Bank] Built question bank: N questions
[ProductCatalog] Loaded N products
[PDF URL Map] Loaded N PDF download links
```

If any line is missing or shows a warning, the corresponding data file was not mounted correctly.

Access: `http://<IP>:8000` (Web UI) | `POST /api/chat` (API) | `GET /health`

## Troubleshooting

- **"FAISS index not found"** — Run `python setup_rag.py` or check `INDEX_DIR` / `MANUAL_INDEX_DIR` paths and volume mounts.
- **"No entity_graph.json found"** — Rebuild with `python src/entity_graph.py --metadata embeddings/primary_index/metadata.jsonl --output embeddings/primary_index/entity_graph.json`, then re-SCP `primary_index/*` to EC2.
- **Manual queries not using manual index** — Verify `MANUAL_INDEX_DIR` is set and the manual index volume is mounted. Logs should show `[RoutedIndex] Manual index:`.
- **No manual download links** — Check that `data/discovered_pdfs.txt` is mounted. Logs should show `[PDF URL Map] Loaded N PDF download links`.
- **Product catalog empty** — Check that `data/pages/products.jsonl` is mounted and `PRODUCTS_FILE` points to it.
- **Slow first request** — The FAISS index + sentence-transformers model download takes 1-2 min on first cold start.
- **Out of memory** — Use `t3.large` (8GB) instead of `t3.medium` (4GB).
- **FAQ not working** — Check for `[FAQ Bank] Built question bank: N questions` in startup logs. If missing, ensure `rag_content.jsonl` was ingested with FAQ entries via `scripts/ingest_faq.py` and the index was rebuilt.
- **Nested directory on SCP** — Use `scp files/* host:~/target/` (not `scp -r dir host:~/`) to copy contents flat.

## License

MIT
