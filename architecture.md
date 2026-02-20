# Supermicro RAG System — Architecture

## High-Level Overview

```
User Query
    │
    ▼
┌──────────────────────────────────────────────────────────────┐
│  chatbot.py — Orchestration Layer                            │
│                                                              │
│  1. Follow-up Detection (regex-based)                        │
│  2. Query Planning (LLM call via query_planner.py)           │
│  3. Catalog Retrieval (product_catalog.py)                   │
│  4. RAG Retrieval (query.py → index.py)                      │
│     - Single or Split retrieval (planner-driven)             │
│  5. Context Assembly (catalog + RAG chunks)                  │
│  6. LLM Response Generation                                  │
└──────────────────────────────────────────────────────────────┘
    │
    ▼
 Response + Sources
```

---

## Data Pipeline

```
PDF Files (data/pdfs/)           Web Content (data/pages/)
        │                                │
        ▼                                ▼
   extract.py                      process_pages.py
   (pypdf extraction)              (boilerplate removal, dedup)
        │                                │
        ▼                                ▼
   data/raw_text/*.json            data/raw_pages/*.json
        │                                │
        └──────────┬─────────────────────┘
                   ▼
              chunk.py
    (RecursiveCharacterTextSplitter)
                   │
                   ▼
          data/chunks.jsonl
                   │
                   ▼
              embed.py
    (Sentence Transformers + BM25)
                   │
                   ▼
     embeddings/faiss_index/
        ├── faiss.index      (FAISS vector index)
        ├── bm25.pkl         (BM25 keyword index)
        └── metadata.jsonl   (chunk text + metadata)
```

**Orchestrated by:** `setup_rag.py --source pdf|pages|both --filter datasheet|all`

### Chunking Parameters

| Parameter | Value | File |
|---|---|---|
| Chunk size | 1000 characters | `src/chunk.py` |
| Chunk overlap | 200 characters | `src/chunk.py` |
| Text splitter | `RecursiveCharacterTextSplitter` (langchain) | `src/chunk.py` |
| Separators | `["\n\n", "\n", ". ", " ", ""]` | `src/chunk.py` |
| Length function | `len()` (character count) | `src/chunk.py` |

### Chunk Metadata Fields

Each chunk in `chunks.jsonl` contains:

| Field | Example |
|---|---|
| `chunk_id` | `"datasheet_SYS-521GE-TNRT.pdf_chunk_3"` |
| `source_file` | `"datasheet_SYS-521GE-TNRT.pdf"` |
| `chunk_index` | `3` |
| `text` | The actual chunk content |
| `total_chunks` | Total chunks from this document |

### Embedding

| Parameter | Value |
|---|---|
| Model | `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions) |
| Batch size | 32 |
| Normalization | L2 normalized before indexing |
| FAISS index type | `IndexFlatIP` (inner product = cosine similarity after L2 norm) |
| Memory mapping | Enabled by default (`FAISS_MMAP=1`) |

**Text enrichment at embed time** (`embed.py:enrich_text`): Each chunk's text is prepended with metadata tags derived from the source filename:

```
[Document: SYS 521GE TNRT] [Platform: X14 H14] [Products: SYS-521GE-TNRT]
<original chunk text>
```

This enrichment is applied to the text stored in `metadata.jsonl` (used by BM25), but the FAISS vectors are generated from the raw `chunk["text"]` without enrichment.

### BM25 Index

- Implementation: `rank_bm25.BM25Okapi`
- Custom tokenizer splits hyphenated product codes into parts:
  - `"SYS-521GE-TNRT"` → `["sys-521ge-tnrt", "sys", "521ge", "tnrt"]`
- Stored as `bm25.pkl` (pickled)

---

## Query Processing Pipeline

### Step 1: Follow-up Detection (`chatbot.py`)

Determines whether a query is a new question or a follow-up to the previous conversation:

1. **Affirmative continuation**: `"yes"`, `"sure"`, `"please"`, etc. → uses last assistant message as retrieval query
2. **Product code present**: regex detects `SYS-*`, `AS-*`, partial codes → always treated as a **new** query
3. **Referential language**: `"it"`, `"this"`, `"tell me more"`, `"what about its..."` → confirmed follow-up, injects most recent product code from conversation
4. **No signals**: treated as new query (conversation context suppressed)

### Step 2: Query Planning (`query_planner.py`)

An LLM-based planner analyzes the query and outputs a structured `QueryPlan`.

#### Planner Configuration

| Parameter | Value |
|---|---|
| OpenAI model | `gpt-4o-mini` (default, via `PLANNER_MODEL` env) |
| Anthropic model | `claude-haiku-4-5` (default, via `PLANNER_MODEL` env) |
| Temperature | `0.0` (deterministic) |
| Max tokens | `300` |

#### QueryPlan Fields

| Field | Type | Description |
|---|---|---|
| `intent` | `str` | `"list"` \| `"detail"` \| `"compare"` \| `"general"` \| `"follow_up"` |
| `product_codes` | `List[str]` | Normalized product codes (e.g., `["SYS-521GE-TNRT", "SYS-421GE-TNRT"]`) |
| `form_factor` | `str \| None` | `"1U"` \| `"2U"` \| `"4U"` \| `"8U"` \| `"Mid-Tower"` |
| `tags` | `List[str]` | Product family/category tags |
| `keywords` | `List[str]` | Free-text terms (e.g., `["NVIDIA", "NVMe"]`) |
| `search_queries` | `List[str]` | Optimized search queries for RAG retrieval |
| `use_catalog` | `bool` | Whether to query the structured product catalog |
| `use_rag` | `bool` | Whether to perform document retrieval |

#### Valid Tags

```
Gold Series, CloudDC, Hyper, Edge, Storage, GPU, GPU-capable,
Blade, Workstation, Mainstream, WIO, Twin, BigTwin, FatTwin, MicroCloud
```

#### Multi-Query Logic

The planner outputs **multiple** `search_queries` when the user asks about distinct topics:

- `"Compare SYS-521GE and SYS-421GE"` → `["SYS-521GE-TNRT specifications datasheet", "SYS-421GE-TNRT specifications datasheet"]`
- `"List MicroCloud and BigTwin servers"` → `["Supermicro MicroCloud multi-node...", "Supermicro BigTwin multi-node..."]`

Single-topic queries get a **single** search query.

#### Fallback

If the planner LLM call fails, a rule-based heuristic (`_fallback_plan`) uses regex to detect intents and tags.

### Step 3: Product Catalog Retrieval (`product_catalog.py`)

A structured in-memory product database loaded from `data/pages/products.jsonl`.

#### Product Fields

| Field | Description |
|---|---|
| `name` | Product name |
| `model` | Model identifier/URL |
| `chassis` | Chassis information |
| `category` | Product category |
| `key_features` | Key features/applications |
| `cpu` | CPU specifications |
| `gpu` | GPU specifications |
| `memory` | Memory specifications |
| `storage` | Storage specifications |
| `network` | Network specifications |
| `price_range` | Price range |
| `url` | Source URL |
| `form_factor` | Derived from chassis/name |
| `sku` | Extracted from name |
| `tags` | Set of product tags |

#### Search Methods

| Method | Description |
|---|---|
| `filter_structured()` | Exact field matching on form_factor, tags, keywords |
| `search()` | Keyword search with stemming and tag matching |

Scoring: text matches = `1.0` (exact) or `0.8` (stem variant); tag matches = `1.0` + `0.3` bonus. Results must exceed `0.5` relevance threshold.

### Step 4: RAG Retrieval (`query.py` → `index.py`)

#### Query Preprocessing (`query.py:preprocess_query`)

1. **Normalize product codes**: `"521 ge"` → `"521ge"`, `"sys 521ge tnrt"` → `"sys-521ge-tnrt"`
2. **Expand partial codes**: `"521GE"` → `"521GE SYS-521GE AS-521GE"`
3. **Platform aliasing**: `"X13"` ↔ `"H13"`, `"X14"` ↔ `"H14"`, `"X12"` ↔ `"H12"`

#### Hybrid Search (`index.py:search_hybrid`)

Three search channels combined via **Reciprocal Rank Fusion (RRF)**:

| Channel | Method | Description |
|---|---|---|
| Semantic | FAISS (cosine similarity) | Dense vector search |
| Keyword | BM25Okapi | Sparse keyword search with custom tokenizer |
| Filename | Inverted index | Token matching against source filenames |

**Fetch K**: `max(top_k * 5, 30)` — each channel retrieves 5x the final count for better fusion coverage.

#### Adaptive Weights (Auto-detected)

| Query Type | Detection | Semantic Weight | Keyword Weight |
|---|---|---|---|
| Product code | Contains `SYS-*`, `AS-*`, or short alphanumeric codes | 0.2 | 0.8 |
| Keyword-heavy | ≤ 2 words | 0.25 | 0.75 |
| Natural language | Everything else | 0.5 | 0.5 |

#### Filename Channel Weight

| Condition | Weight |
|---|---|
| Best match score ≥ 2 terms | 1.0 |
| Otherwise | 0.3 |

#### RRF Score Calculation

```
rrf_score = weight × (1.0 / (rrf_k + rank + 1))
```

| Parameter | Value |
|---|---|
| `rrf_k` | 60 |

Scores from all three channels are summed per chunk.

#### Post-RRF Boosting & Penalties

| Rule | Multiplier | Condition |
|---|---|---|
| Boilerplate penalty | 0.3× | Chunk contains "global leader in high performance" AND "broad range of skus" |
| PDF boost | 1.05× | Source file ends with `.pdf` |
| Product page boost | 1.02× | Source file starts with `web_product_` |
| Web page | 1.0× (no boost) | Source file starts with `web_page_` |

#### Source Diversity (`max_per_source`)

Controls how many chunks from a single source document are returned. Prevents one large document from dominating results.

| Intent | `max_per_source` | Behavior |
|---|---|---|
| `list` | 2 | Source diversity enforced |
| `compare` | 3 | Source diversity enforced |
| `detail` | None (no cap) | Context expansion enabled |
| `follow_up` | None (no cap) | Context expansion enabled |
| `general` | None (no cap) | Context expansion enabled |

#### Context Expansion (Concentrated Path)

When `max_per_source` is `None`, the system checks if the top-scoring document should have its sibling chunks pulled in:

**Trigger conditions** (both must be true):
1. Top source has `> 5` total chunks (multi-page document)
2. Top score is `> 1.03×` the second result's score

**Behavior**: Adds up to `top_k // 2` sibling chunks from the top source, sorted by `chunk_index` to maintain document order. Siblings receive `0.95×` the top chunk's score.

### Step 5: Split Retrieval (`chatbot.py`)

When the planner outputs ≥ 2 `search_queries`, retrieval is performed separately for each query.

**Safety net**: If the planner returned ≥ 2 `product_codes` but only 1 `search_query`, the system auto-splits into per-product queries:
```
"SYS-521GE-TNRT specifications datasheet"
"SYS-421GE-TNRT specifications datasheet"
```

**Product code injection**: Only performed for **single** queries (to avoid cross-contamination in split queries). Missing product codes are appended to the search query.

**Per-query allocation**: `per_k = max(5, rag_top_k // len(search_queries))`

**Interleaving**: Round-robin across queries with deduplication by `chunk_id` to ensure balanced representation of each topic.

**Product code safety net**: After retrieval, the system checks whether chunks from each identified `product_code` actually appear in the results. If a product's documents are missing (common when topic terms like "GPU" dominate the search), a focused rescue retrieval is performed using just the product code. Rescued chunks are appended up to `max(3, rag_top_k // 3)` per missing product.

### Step 6: Intent-Based Retrieval Parameters

| Intent | `rag_top_k` | `catalog_max` | `max_per_source` |
|---|---|---|---|
| `list` (with catalog) | `max(top_k, 10)` | 30 | 2 |
| `list` (no catalog) | `max(top_k, 15)` | 0 | 2 |
| `detail` | `top_k` | min(catalog_results, 5) | None |
| `follow_up` | `top_k` | min(catalog_results, 5) | None |
| `compare` | `max(int(top_k * 1.5), 15)` | min(catalog_results, 10) | 3 |
| `general` | `top_k` | 0 | None |

Default `top_k`: **10**

---

## Cross-Encoder Reranking (`query.py`)

| Parameter | Value |
|---|---|
| Model | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Enabled by default | Yes (`ENABLE_RERANKING=1`) |
| Lazy loaded | Yes (on first use) |

### Score Combination

Cross-encoder scores are normalized to 0–1 range, then blended with the original ranking position:

| Query Type | Original Rank Weight | Cross-Encoder Weight |
|---|---|---|
| Product code queries | 0.6 | 0.4 |
| Natural language queries | 0.3 | 0.7 |

> **Note**: Reranking is implemented and available but is not currently called in the main retrieval flow. The hybrid search pipeline (`query.py:retrieve`) calls `search_hybrid()` directly without a reranking step.

---

## LLM Response Generation

### Models

| Provider | Default Model | Environment Variable |
|---|---|---|
| OpenAI | `gpt-5.2` | `LLM_MODEL` |
| Anthropic | `claude-opus-4-5` | `ANTHROPIC_MODEL` |
| Ollama | `llama3` | `OLLAMA_MODEL` |

Provider is selected via `LLM_PROVIDER` env var (default: `openai`).

### Parameters

| Parameter | Value |
|---|---|
| Temperature | 0.5 |
| Max tokens (Anthropic) | 2048 |

### System Prompt Highlights

- Domain knowledge for Supermicro product naming conventions (SYS-, AS-, X-series, SC-, PWS-, SBI-, AOC-)
- Instructions to synthesize information from multiple sources
- Target response length: 200–350 words
- Use tables for comparisons
- When data is incomplete: state briefly, provide what is known, suggest checking Supermicro website, then stop
- Anti-hallucination guardrails: avoid listing missing information, avoid over-hedging

### User Prompt Structure

```
## CONVERSATION HISTORY       (only if confirmed follow-up)
<history>
---

## RETRIEVED CONTEXT
The following excerpts were retrieved from Supermicro documentation.
Sources: <top 5 source files>

---
<PRODUCT CATALOG DATA>        (if catalog results exist)
<DOCUMENTATION CONTEXT>       (RAG chunks)
---

## USER QUESTION
<question>

## INSTRUCTIONS
1. Use retrieved context as primary source
2. Supplement with general knowledge when context is incomplete
3. Cite source documents
4. Provide key specs for product questions
5. Refer to conversation history for follow-ups
6. Be helpful and informative
```

---

## Server / API (`server.py`)

**Framework**: FastAPI

### Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Health check (`{"ok": true}`) |
| `GET` | `/` | Serves static UI (`static/index.html`) |
| `POST` | `/api/chat` | Chat endpoint |
| `GET` | `/api/document/{filename}` | Serve PDF documents |
| `GET` | `/api/documents/search` | Search documents by filename |

### Chat Request/Response

**Request** (`POST /api/chat`):
```json
{
  "message": "What are the specs of SYS-521GE-TNRT?",
  "history": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
```

**Response**:
```json
{
  "answer": "...",
  "sources": ["datasheet_SYS-521GE-TNRT.pdf", "..."]
}
```

### Conversation History

- Keeps last **6 messages** (3 turns) for context
- History is formatted as `"User: ...\nAssistant: ..."` pairs

### CORS

Configurable via `CORS_ORIGINS` environment variable (comma-separated origins).

---

## BM25 Query Expansion (`index.py`)

Before BM25 scoring, queries are expanded with stemmed variants using suffix rules:

| Rule | Example |
|---|---|
| `-ies` → `-y` | `categories` → `category` |
| `-ses` → drop `-es` | `processes` → `process` |
| `-es` → drop `-es` or `-s` | `switches` → `switch` / `switche` |
| `-s` → drop `-s` | `servers` → `server`, `skus` → `sku` |
| `-en` → drop `-en` | `golden` → `gold` |
| `-ing` → drop `-ing` | `computing` → `comput` / `compute` |
| `-ed` → drop `-ed` or `-d` | `configured` → `configur` / `configure` |

Stems shorter than 3 characters are discarded as noise.

---

## Environment Variables Reference

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `"openai"` | LLM provider: `openai`, `anthropic`, `ollama` |
| `LLM_MODEL` | `"gpt-5.2"` | Main LLM model |
| `ANTHROPIC_API_KEY` | — | Anthropic API key |
| `ANTHROPIC_MODEL` | `"claude-opus-4-5"` | Anthropic model |
| `OPENAI_API_KEY` | — | OpenAI API key |
| `OLLAMA_BASE_URL` | `"http://localhost:11434"` | Ollama server URL |
| `OLLAMA_MODEL` | `"llama3"` | Ollama model |
| `PLANNER_MODEL` | `"gpt-4o-mini"` / `"claude-haiku-4-5"` | Query planner model (depends on provider) |
| `EMBEDDING_MODEL` | `"sentence-transformers/all-MiniLM-L6-v2"` | Embedding model |
| `INDEX_DIR` | `"embeddings/faiss_index/"` | FAISS index directory |
| `FAISS_MMAP` | `"1"` | Enable FAISS memory mapping |
| `TOP_K` | `"10"` | Default number of chunks to retrieve |
| `ENABLE_RERANKING` | `"1"` | Enable cross-encoder reranking |
| `PRODUCTS_FILE` | `"data/pages/products.jsonl"` | Product catalog file |
| `CORS_ORIGINS` | — | Comma-separated allowed CORS origins |

---

## File Structure

```
supermicro-rag/
├── src/
│   ├── chatbot.py          # Main orchestration, follow-up detection, LLM calls
│   ├── query_planner.py    # LLM-based intent classification and query routing
│   ├── query.py            # Query preprocessing, hybrid retrieval, reranking
│   ├── index.py            # FAISS + BM25 + filename hybrid search, RRF, context expansion
│   ├── embed.py            # Embedding generation, FAISS/BM25 index creation
│   ├── chunk.py            # Document chunking with RecursiveCharacterTextSplitter
│   ├── extract.py          # PDF text extraction (pypdf)
│   ├── process_pages.py    # Web content processing
│   ├── product_catalog.py  # Structured product database
│   └── server.py           # FastAPI server
├── setup_rag.py            # Pipeline orchestration script
├── data/
│   ├── pdfs/               # Source PDF documents
│   ├── pages/              # Web content (products.jsonl, rag_content.jsonl)
│   ├── raw_text/           # Extracted PDF text (JSON)
│   ├── raw_pages/          # Processed web pages (JSON)
│   └── chunks.jsonl        # Chunked documents
├── embeddings/
│   └── faiss_index/
│       ├── faiss.index     # FAISS vector index
│       ├── bm25.pkl        # BM25 keyword index
│       └── metadata.jsonl  # Chunk metadata
├── static/
│   └── index.html          # Chat UI
└── tests/
    ├── test_product_queries.py
    └── super-tests.py
```

---

## Key Design Decisions

1. **Hybrid search over pure semantic**: Product codes like `SYS-521GE-TNRT` are better matched by BM25 keyword search than by semantic similarity. The three-channel RRF approach (semantic + BM25 + filename) ensures both semantic understanding and exact-match reliability.

2. **LLM-driven query planning**: Instead of hardcoded regex for intent detection and query splitting, a fast LLM (`gpt-4o-mini` / `claude-haiku-4-5` at temperature=0.0) classifies intent, normalizes product codes, and generates optimized search queries. This handles misspellings, partial codes, and multi-product queries.

3. **Split retrieval for multi-topic queries**: When comparing products or listing multiple families, separate retrieval calls are made per topic with round-robin interleaving. This prevents one dominant topic from drowning out others.

4. **Context expansion for detail queries**: For single-product questions, if the top document scores significantly higher than the rest, sibling chunks are pulled in to give the LLM complete context from that document.

5. **Source diversity for broad queries**: `max_per_source` caps for `list` and `compare` intents prevent a single large document from consuming all retrieval slots.

6. **Dual data sources**: The product catalog provides structured product metadata (specs, tags, form factors), while RAG provides unstructured documentation context. Both are sent to the LLM when relevant.
