# Enterprise RAG Q&A Platform

A production-style **retrieval-augmented generation (RAG)** stack for answering questions over large technical document corpora, **structured product catalogs**, and FAQ content. It combines semantic and lexical search, intent-aware routing, and multi-turn dialogue—exposed through a **REST API**, **web chat UI**, and **containerized** cloud deployment.

> **Sample corpus:** This repository is configured for enterprise server hardware documentation (thousands of PDFs plus scraped web/FAQ data). The architecture is domain-agnostic: swap `data/`, indexes, and taxonomy prompts for your own knowledge base.

---

## Highlights (what this project demonstrates)

- **Hybrid retrieval** — dense (FAISS) + sparse (BM25) search with reciprocal rank fusion; optional cross-encoder reranking  
- **Intent-aware query planning** — fast LLM step classifies user intent (e.g. list, detail, compare, FAQ, follow-up) and produces retrieval plans  
- **Multi-collection routing** — separate vector indexes for *spec/marketing/FAQ* vs *long-form manuals*, selected by query type  
- **Structured + unstructured fusion** — JSONL product catalog for filters and listings; vector index for narrative docs  
- **Graph-assisted retrieval** — entity graph over index metadata for related-document expansion  
- **FAQ specialization** — question-to-question matching layered with hybrid search for policy-style Q&A  
- **Operational readiness** — health endpoints, env-based configuration, Docker, and documented AWS (ECR/EC2) rollout  

---

## Features

- **Dual-index routing** — primary pool (datasheets, web-derived content, FAQ, accessories) vs manual pool (user guides, installation docs), chosen by planner and doc-type hints  
- **LLM query planner** — outputs structured plan: intent, filters, rewritten search queries, catalog/RAG flags, manual vs primary scope  
- **Entity graph** — lightweight relationship layer on primary metadata for multi-hop context  
- **FAQ question bank** — cosine match on FAQ titles/questions plus source-filtered hybrid retrieval  
- **Product catalog** — structured listings from `data/pages/products.jsonl` (replace with your own schema)  
- **Document links** — deterministic PDF URLs from `data/discovered_pdfs.txt` for manual-style answers  
- **Local embeddings** — sentence-transformers for indexing and search (no embedding API cost at query time)  
- **Answer generation** — OpenAI or Anthropic (or Ollama) with source-backed responses  
- **Multi-turn chat** — follow-up detection and context carry-over  
- **FastAPI** — Web UI (`static/`) and JSON API  

---

## Quick Start

```bash
# 1. Install dependencies
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Configure API keys
cp .env.example .env
# Edit .env — set ANTHROPIC_API_KEY and/or OPENAI_API_KEY

# 3. Build indexes (PDFs + web-derived JSONL)
python setup_rag.py                    # both PDFs and data/pages
# python setup_rag.py --source pages   # web content only
# python setup_rag.py --source pdf     # PDFs only

# 4. Run
python src/chatbot.py --interactive                    # CLI
uvicorn src.server:app --host 0.0.0.0 --port 8000      # http://localhost:8000
```

---

## Testing

```bash
python tests/test_product_queries.py                    # full suite
python tests/test_product_queries.py --summary          # quality hints
python tests/test_product_queries.py --category detail  # one category
python tests/test_product_queries.py --id rec_8gpu_h200 # single case
python tests/test_product_queries.py --dry-run          # list only
python tests/test_product_queries.py --output results.txt --model claude-sonnet-4-5
python -m pytest tests/test_form_factors.py -v         # form-factor helpers
```

Example categories in the eval harness: `list`, `detail`, `compare`, `general`, `conversational`, `recommendation`, `misspell`, `multi`, `followup`, `faq`.

---

## Architecture

```
PDFs + Web JSONL (data/pages)
       |
   Extract / normalize text
       |
   Chunking
       |
   Index build (setup_rag.py)
       |
   ┌────────────────────────┬─────────────────────────┐
   Primary vector index     Manual vector index       Entity graph
   (specs, web, FAQ, …)     (guides, procedures)      (on primary metadata)
   └───────────┬────────────┴────────────┬────────────┘
               |                         |
   User query → Query planner (small LLM) → Routed hybrid retrieval
               |                                    |
               +── Structured catalog (JSONL) ──────┴──→ Answer LLM (+ links)
```

---

## Repository Layout

```
├── src/
│   ├── server.py           # FastAPI app + chat UI
│   ├── chatbot.py          # Orchestration, prompts, citations
│   ├── query_planner.py    # Intent + plan JSON
│   ├── query.py            # Hybrid retrieval + FAQ bank
│   ├── product_catalog.py  # Structured catalog filters
│   ├── form_factors.py     # Shared form-factor vocabulary
│   ├── index.py            # HybridIndex + multi-index routing
│   ├── embed.py            # Embeddings
│   ├── chunk.py            # Chunking
│   ├── extract.py          # PDF text extraction
│   ├── entity_graph.py     # Graph build / traversal
│   └── process_pages.py    # JSONL pages → text chunks
├── scripts/                # ingest helpers, AWS/S3 utilities
├── data/pages/             # products.jsonl, rag_content.jsonl, …
├── data/discovered_pdfs.txt
├── tests/
├── static/index.html
├── embeddings/
│   ├── primary_index/
│   └── manual_index/
├── Dockerfile
├── requirements.txt
└── setup_rag.py
```

---

## Configuration

Key environment variables (see `.env.example`):

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `openai` | `openai`, `anthropic`, or `ollama` |
| `OPENAI_API_KEY` | — | If using OpenAI |
| `LLM_MODEL` | (see `.env.example`) | Answer model |
| `ANTHROPIC_API_KEY` | — | If using Anthropic |
| `PLANNER_MODEL` | (see `.env.example`) | Small/fast model for planning |
| `TOP_K` | `10` | Chunks retrieved per search pass |
| `INDEX_DIR` | `embeddings/primary_index/` | Primary FAISS index path |
| `MANUAL_INDEX_DIR` | `embeddings/manual_index/` | Manual FAISS index path |
| `FAISS_MMAP` | `1` | Memory-map indexes |
| `ENABLE_RERANKING` | `1` | Cross-encoder reranker on/off |

---

## Deployment (Docker + AWS EC2)

Indexes are large (on the order of **~1GB+** FAISS + metadata); a small **EC2** instance with Docker is a practical pattern (avoids cold-start limits of some serverless options).

### Build and push (ECR)

```bash
./scripts/aws_push_ecr.sh
```

Requires `AWS_ACCOUNT_ID`, `AWS_REGION`, and `ECR_REPO` in `.env` (default repo name is set for this project; change to match your registry).

### Ship indexes and data to the host

Each index directory should contain at least: `faiss.index`, `bm25.pkl`, `metadata.jsonl`. Primary also uses `entity_graph.json` when present.

```bash
EC2=user@<INSTANCE_IP>
KEY=~/.ssh/your-key.pem

ssh -i $KEY $EC2 "mkdir -p ~/embeddings/primary_index ~/embeddings/manual_index ~/data/pages ~/data"

scp -i $KEY embeddings/primary_index/* $EC2:~/embeddings/primary_index/
scp -i $KEY embeddings/manual_index/* $EC2:~/embeddings/manual_index/
scp -i $KEY data/pages/* $EC2:~/data/pages/
scp -i $KEY data/discovered_pdfs.txt $EC2:~/data/
```

### Run container on EC2

Authenticate Docker to ECR, set your image URI, then:

```bash
docker run -d --name rag-app -p 8000:8000 \
  -v ~/embeddings/primary_index:/app/embeddings/primary_index:ro \
  -v ~/embeddings/manual_index:/app/embeddings/manual_index:ro \
  -v ~/data/pages:/app/data/pages:ro \
  -v ~/data/discovered_pdfs.txt:/app/data/discovered_pdfs.txt:ro \
  -e ANTHROPIC_API_KEY="..." \
  -e LLM_PROVIDER=anthropic \
  -e TOP_K=15 \
  -e FAISS_MMAP=1 \
  <YOUR_ECR_IMAGE>:latest
```

Verify: `GET /health`, `docker logs rag-app`. UI: `http://<IP>:8000` · API: `POST /api/chat`.

---

## Troubleshooting

- **Missing FAISS index** — Run `python setup_rag.py` or fix `INDEX_DIR` / mounts.  
- **No `entity_graph.json`** — Rebuild per `src/entity_graph.py` CLI and copy into `primary_index/`.  
- **Manual questions hit wrong index** — Confirm `MANUAL_INDEX_DIR` and logs show the manual index loaded.  
- **No PDF download links** — Mount `data/discovered_pdfs.txt`.  
- **Empty catalog** — Mount `products.jsonl` and check `PRODUCTS_FILE`.  
- **Cold start slow** — First load of models + mmap can take minutes on small instances.  
- **OOM** — Prefer **8GB+** RAM for dual indexes + reranker.  
- **SCP nested folders** — Copy with `scp dir/* host:~/target/` for flat contents.  

---

## License

MIT
