# Supermicro RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers questions about Supermicro products, solutions, and documentation using the downloaded PDF collection.

## 🚀 Quick Start

```bash
# 1. Install dependencies
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Set up OpenAI API key
cp .env.example .env
# Edit .env and add: OPENAI_API_KEY=sk-your-key-here

# 3. Process PDFs and create index
python setup_rag.py

# 4. Run chatbot
python src/chatbot.py --interactive
```

**That's it!** See [Quick Start Guide](#quick-start-guide) below for detailed instructions.

## Overview

This project implements a RAG-based question-answering system that:
- Processes Supermicro PDF documents (manuals, datasheets, white papers, case studies, etc.)
- Extracts and chunks text content for efficient retrieval
- Embeds documents into a vector database using local models (FREE)
- Answers user questions by retrieving relevant context and generating responses using an LLM

---

## Architecture

### Implemented Architecture: Simple Local RAG

**Tech Stack:**
- **PDF Processing**: `pypdf` for text extraction
- **Text Chunking**: `RecursiveCharacterTextSplitter` (via LangChain / `langchain-text-splitters`)
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` - free, local
- **Vector Store**: `FAISS` (Facebook AI Similarity Search) - disk-based
- **LLM**: OpenAI GPT-5.2 (via API)

**Key Features:**
- ✅ No API costs for embeddings (local models)
- ✅ Fast setup and processing
- ✅ Privacy-friendly (embeddings and search are local)
- ✅ Parallel PDF processing for speed
- ✅ Resume capability (skips already processed files)

**How It Works:**
1. Extract text from all PDFs (parallelized)
2. Split into chunks (1000 chars with 200 char overlap)
3. Generate embeddings for each chunk (local sentence-transformers)
4. Store in FAISS index
5. Query: embed question → find top-k similar chunks → send to OpenAI LLM with context

---

## Recommended Tech Stack (Starting Point)

### Current Dependencies

```python
# Core dependencies (see requirements.txt)
pypdf>=3.0.0                    # PDF text extraction
langchain>=1.0.0                 # RAG orchestration
langchain-text-splitters>=1.1.0  # Text splitting utilities
sentence-transformers>=2.2.0     # Local embeddings (FREE)
faiss-cpu>=1.7.4                # Vector search (or faiss-gpu if GPU available)
openai>=2.15.0                   # LLM API (for GPT-5.2)
python-dotenv>=1.0.0            # Environment variables
tqdm>=4.65.0                     # Progress bars
numpy>=1.24.0                    # Numerical operations
```

---

## Project Structure

```
supermicro-rag/
├── pdfs/                    # PDF files (6000+)
├── data/                    # Extracted text and chunks
├── embeddings/faiss_index/  # FAISS vector index
├── src/                     # Core modules (extract, chunk, embed, query, chatbot)
├── config/                  # Configuration files
├── setup_rag.py            # Complete setup script
└── requirements.txt         # Dependencies
```

---

## Implementation Details

- **Chunking**: 1000 characters with 200 character overlap using `RecursiveCharacterTextSplitter`
- **Retrieval**: Top-k similarity search (default: 5 chunks) using FAISS
- **Interface**: Command-line only (CLI)
- **Performance**: Parallel PDF processing using multiprocessing

## Quick Start Guide

### Prerequisites

- Python 3.10 or higher
- PDF files in the `pdfs/` directory (already included: 6000+ PDFs)
- OpenAI API key for GPT-5.2

### Step 1: Install Dependencies

```bash
cd supermicro-rag
python3 -m venv venv
source venv/bin/activate          # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Step 2: Configure API Keys (Required for OpenAI)

**Option A: Use OpenAI (Recommended for best quality)**

1. **Get your OpenAI API key:**
   - Go to https://platform.openai.com/api-keys
   - Sign in or create an account
   - Click "Create new secret key"
   - Copy the key (starts with `sk-`)

2. **Create `.env` file:**
   ```bash
   cp .env.example .env
   ```

3. **Edit `.env` and add your API key:**
   ```bash
   OPENAI_API_KEY=sk-your-actual-api-key-here
   LLM_PROVIDER=openai
   LLM_MODEL=gpt-5.2
   ```


**⚠️ Important:** The `.env` file is already in `.gitignore` - never commit your API keys!

### Step 3: Extract and Index PDFs

**Recommended: Run the complete pipeline (fastest)**

```bash
python setup_rag.py
```

This will:
1. Extract text from all PDFs (uses parallel processing - much faster!)
2. Chunk the extracted text
3. Generate embeddings and create FAISS index

**Or run steps manually:**

```bash
# Step 1: Extract text from all PDFs (parallelized - uses all CPU cores by default)
python src/extract.py --input pdfs/ --output data/raw_text/ --workers 8

# Step 2: Chunk the text
python src/chunk.py --input data/raw_text/ --output data/chunks.jsonl

# Step 3: Generate embeddings and create FAISS index
python src/embed.py --input data/chunks.jsonl --output embeddings/faiss_index/
```

**Note:** The first run processes all PDFs and may take 10-20 minutes. Subsequent runs skip already processed files.

### Step 4: Run the Chatbot

**Interactive mode (recommended):**
```bash
python src/chatbot.py --interactive
```

**Single query:**
```bash
python src/chatbot.py --query "What are the power requirements for X13 servers?"
```

**With custom settings:**
```bash
# Use specific model
python src/chatbot.py --interactive --llm-model gpt-5.2

# Retrieve more chunks for better context
python src/chatbot.py --interactive --top-k 10
```

---

## Architecture Overview

### What Uses APIs vs. Local Processing

```
┌─────────────────────────────────────────────────────────┐
│                    RAG Pipeline                         │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  1. PDF Extraction    →  Local (pypdf) - FREE           │
│  2. Text Chunking     →  Local (langchain) - FREE       │
│  3. Embeddings        →  Local (sentence-transformers)  │
│                        →  FREE, no API calls             │
│  4. Vector Search     →  Local (FAISS) - FREE           │
│  5. LLM Answer        →  OpenAI API (GPT-5.2)          │
│                        →  Only this step uses API/key    │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

**Key Points:**
- ✅ **Embeddings are FREE** - uses local `sentence-transformers` (no API costs)
- ✅ **Vector search is FREE** - uses local FAISS index
- ✅ **Only LLM uses API** - OpenAI GPT-5.2
- ✅ **PDF processing is FREE** - all local processing

### Current Implementation

This project uses **Option 1: Simple Local RAG** with:
- **PDF Processing**: `pypdf` (local, free)
- **Text Chunking**: `RecursiveCharacterTextSplitter` from LangChain
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` (local, free)
- **Vector Store**: FAISS (local, free)
- **LLM**: GPT-5.2 (OpenAI API)

---

## Future Enhancements

1. **Web Interface**: Gradio or Streamlit for browser-based interaction
2. **Multi-modal RAG**: Extract images/diagrams from PDFs, use vision models
3. **Table Extraction**: Better handling of datasheet tables
4. **Citation Links**: Clickable links back to original PDF pages
5. **Conversation Memory**: Remember previous questions in a session
6. **Hybrid Search**: Combine semantic search with keyword search (BM25)
7. **Evaluation**: Test accuracy with a question-answer dataset
8. **Streaming Responses**: Stream LLM responses token-by-token

---

## Resources

- **LangChain RAG Tutorial**: https://python.langchain.com/docs/use_cases/question_answering/
- **LlamaIndex**: https://www.llamaindex.ai/
- **FAISS**: https://github.com/facebookresearch/faiss
- **Sentence Transformers**: https://www.sbert.net/

---

## Configuration

**Environment Variables** (`.env` file):
- `OPENAI_API_KEY`: Your OpenAI API key (required)
- `LLM_MODEL`: Model name (default: `gpt-5.2`)

**Key Command Options:**
- `chatbot.py --interactive`: Interactive mode
- `chatbot.py --query "question"`: Single query
- `extract.py --workers 8`: Parallel processing (faster)
- `chatbot.py --top-k 10`: Retrieve more chunks

See `USAGE.md` for complete command reference.

---

## Troubleshooting

### "No module named 'src'"
Run from the project root directory:
```bash
cd /home/h1aihan/supermicro-rag
python src/chatbot.py --interactive
```

Or use module mode:
```bash
python -m src.chatbot --interactive
```

### "FAISS index not found"
Run the setup pipeline first:
```bash
python setup_rag.py
```

### "Missing OPENAI_API_KEY" or "Error calling OpenAI API"
1. **Check `.env` file exists:**
   ```bash
   ls -la .env
   ```

2. **Verify API key is set:**
   ```bash
   cat .env | grep OPENAI_API_KEY
   ```

3. **Get a new API key:**
   - Visit https://platform.openai.com/api-keys
   - Create a new secret key
   - Add it to `.env`: `OPENAI_API_KEY=sk-...`


### Extraction is slow
Use parallel processing (default uses all CPU cores):
```bash
python src/extract.py --workers 8  # Adjust based on your CPU cores
```


---

## Additional Resources

- **USAGE.md**: Detailed usage guide with all command options
- **ENV_SETUP.md**: Complete guide to environment variables and API keys
- **RAG_BEHAVIOR.md**: Explanation of how the RAG system uses knowledge
- **QUICKSTART.md**: Quick reference for common tasks
