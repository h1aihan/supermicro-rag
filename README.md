# Supermicro RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers questions about Supermicro products, solutions, and documentation using 6000+ PDF documents and structured web content.

## Features

- **Hybrid search** — FAISS semantic search + BM25 keyword matching for high recall
- **LLM query planner** — classifies intent (list / detail / compare / recommend / general) and routes to catalog or RAG accordingly
- **Product catalog integration** — structured product listing from `data/pages/products.jsonl`
- **Local embeddings** with sentence-transformers (no API cost for search)
- **Anthropic Claude or OpenAI** for answer generation with source citations
- **Multi-turn conversation** — follow-up detection with context carry-over
- **Web UI** and REST API via FastAPI
- **Docker-ready** for AWS EC2 deployment

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

Categories: `list`, `detail`, `compare`, `general`, `conversational`, `recommendation`, `misspell`, `multi`, `followup`.

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
│   ├── query.py           # Hybrid retrieval (FAISS + BM25)
│   ├── product_catalog.py # Structured product lookup from products.jsonl
│   ├── index.py           # FAISS index build + load
│   ├── embed.py           # Embedding generation
│   ├── chunk.py           # Text chunking
│   ├── extract.py         # PDF text extraction
│   └── process_pages.py   # Web page JSONL → chunkable text
├── data/pages/            # products.jsonl, rag_content.jsonl
├── tests/
│   └── test_product_queries.py
├── static/index.html      # Chat UI
├── embeddings/faiss_index/ # FAISS index + metadata (~1.5GB)
├── scripts/aws_push_ecr.sh
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

```bash
# 1. Push Docker image to ECR
./scripts/aws_push_ecr.sh

# 2. Launch EC2 (t3.medium+), install Docker, pull image

# 3. Copy index and data to EC2
scp -r embeddings/faiss_index ec2-user@<IP>:~/faiss_index
scp -r data/pages ec2-user@<IP>:~/pages

# 4. Run container
docker run -d --name supermicro-rag -p 8000:8000 \
  -v ~/faiss_index:/app/embeddings/faiss_index:ro \
  -v ~/pages:/app/data/pages:ro \
  -e ANTHROPIC_API_KEY="sk-ant-..." \
  -e LLM_PROVIDER=anthropic \
  -e ANTHROPIC_MODEL=claude-opus-4-5 \
  -e INDEX_DIR=/app/embeddings/faiss_index \
  -e TOP_K=15 \
  <ACCOUNT>.dkr.ecr.<REGION>.amazonaws.com/supermicro-rag:latest
```

Access: `http://<IP>:8000` (Web UI) | `POST /api/chat` (API) | `GET /health`

## Troubleshooting

- **"FAISS index not found"** — Run `python setup_rag.py` or check `INDEX_DIR` path.
- **Slow first request** — The 1.5GB FAISS index + embedding model download takes 1-2 min on first load.
- **Out of memory** — Use `t3.large` (8GB) instead of `t3.medium` (4GB).

## License

MIT
