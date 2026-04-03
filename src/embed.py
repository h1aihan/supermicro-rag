#!/usr/bin/env python3
"""
Generate embeddings for text chunks and index them.

Supports two backends:
  - Qdrant (production): dense + sparse vectors in Qdrant collections
  - FAISS + BM25 (legacy): file-based indexes for local development

Backend selection: Qdrant is used when --qdrant-url is provided or the
QDRANT_URL environment variable is set; otherwise falls back to FAISS.
"""

import json
import math
import os
import pickle
import re
import zlib
import argparse
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


# ── BM25 parameters ──────────────────────────────────────────────────────

_BM25_K1 = 1.5
_BM25_B = 0.75


# ── Regex constants (used by enrich_text) ─────────────────────────────────

_PLATFORM_ALIASES = {
    'x12': 'H12', 'h12': 'X12',
    'x13': 'H13', 'h13': 'X13',
    'x14': 'H14', 'h14': 'X14',
}

_PREFIX_RE = re.compile(
    r'^(datasheet|web_product|web_page|product[-_]brief|solution[-_]brief)[-_]',
    re.IGNORECASE,
)
_HASH_RE = re.compile(r'__[a-f0-9]{8,}$')
_EXT_RE = re.compile(r'\.(pdf|txt|json|jsonl)$', re.IGNORECASE)
_PRODUCT_CODE_RE = re.compile(
    r'\b(?:SYS|AS|SSG|SBI|AOC|MBD|PWS)-[A-Za-z0-9](?:[-A-Za-z0-9]*[A-Za-z0-9])?',
    re.IGNORECASE,
)
_PLATFORM_RE = re.compile(r'\b([XH]1[0-9])\b', re.IGNORECASE)


def enrich_text(chunk: Dict) -> str:
    """Prepend source-file metadata so embeddings encode product identity."""
    source = chunk.get('source_file', '')
    text = chunk.get('text', '')
    if not source:
        return text

    product_codes = [pc.upper() for pc in _PRODUCT_CODE_RE.findall(source)]

    name = _EXT_RE.sub('', source)
    name = _HASH_RE.sub('', name)
    name = _PREFIX_RE.sub('', name)
    label = re.sub(r'[_|]+', ' ', name).replace('-', ' ')
    label = re.sub(r'\s+', ' ', label).strip()

    platforms = _PLATFORM_RE.findall(label)
    platform_tags: List[str] = []
    for p in platforms:
        pu = p.upper()
        if pu not in platform_tags:
            platform_tags.append(pu)
        alias = _PLATFORM_ALIASES.get(p.lower())
        if alias and alias not in platform_tags:
            platform_tags.append(alias)

    parts = [f'[Document: {label}]']
    if platform_tags:
        parts.append(f'[Platform: {" ".join(platform_tags)}]')
    if product_codes:
        parts.append(f'[Products: {", ".join(product_codes)}]')

    return f'{" ".join(parts)}\n{text}'


# ── BM25 tokenization ────────────────────────────────────────────────────

def tokenize_for_bm25(text: str) -> List[str]:
    """
    Tokenizer for BM25 optimized for Supermicro product codes.

    For product codes like SYS-521GE-TNRT, generates both:
    - The full hyphenated token: sys-521ge-tnrt
    - Individual parts: sys, 521ge, tnrt
    """
    text = text.lower()

    hyphenated_tokens = re.findall(r'\b[\w]+-[\w-]+\b', text)
    simple_tokens = re.findall(r'\b\w+\b', text)

    tokens: List[str] = []
    for token in hyphenated_tokens:
        tokens.append(token)
        tokens.extend(token.split('-'))

    tokens.extend(simple_tokens)

    seen: set = set()
    unique: List[str] = []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            unique.append(t)
    return unique


# ── Sparse vector helpers ─────────────────────────────────────────────────

def token_to_index(token: str) -> int:
    """Deterministic hash from token string to sparse-vector index."""
    return zlib.crc32(token.encode('utf-8')) & 0xFFFFFFFF


def compute_corpus_stats(tokenized_corpus: List[List[str]]) -> Dict:
    """Compute BM25 IDF and average document length."""
    num_docs = len(tokenized_corpus)
    df: Counter = Counter()
    total_len = 0

    for tokens in tokenized_corpus:
        total_len += len(tokens)
        for t in set(tokens):
            df[t] += 1

    avg_dl = total_len / max(num_docs, 1)
    idf: Dict[str, float] = {}
    for t, freq in df.items():
        idf[t] = max(0.0, math.log((num_docs - freq + 0.5) / (freq + 0.5)))

    return {"num_docs": num_docs, "avg_doc_len": avg_dl, "idf": idf}


def build_sparse_vector(
    tokens: List[str], corpus_stats: Dict
) -> Tuple[List[int], List[float]]:
    """Build a BM25-weighted sparse vector for a single document.

    IDF is baked into the document vector so that query vectors only
    need unit weights (1.0 per unique query term).
    """
    idf = corpus_stats["idf"]
    avg_dl = corpus_stats["avg_doc_len"]
    dl = len(tokens)

    tf = Counter(tokens)
    indices: List[int] = []
    values: List[float] = []

    for t, freq in tf.items():
        t_idf = idf.get(t, 0.0)
        if t_idf <= 0:
            continue
        bm25_w = t_idf * (freq * (_BM25_K1 + 1)) / (
            freq + _BM25_K1 * (1 - _BM25_B + _BM25_B * dl / max(avg_dl, 1))
        )
        indices.append(token_to_index(t))
        values.append(bm25_w)

    return indices, values


def build_query_sparse_vector(tokens: List[str]) -> Tuple[List[int], List[float]]:
    """Build a sparse vector for a search query.

    IDF is already encoded in document vectors, so each unique query
    term gets weight 1.0.
    """
    seen: set = set()
    indices: List[int] = []
    values: List[float] = []
    for t in tokens:
        if t in seen:
            continue
        seen.add(t)
        indices.append(token_to_index(t))
        values.append(1.0)
    return indices, values


# ── Chunk loading & dense embeddings ──────────────────────────────────────

def load_chunks(chunks_file: str) -> List[Dict]:
    """Load chunks from a JSONL file."""
    chunks: List[Dict] = []
    with open(chunks_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
    return chunks


def generate_embeddings(
    chunks: List[Dict],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> np.ndarray:
    """Generate dense embeddings for all chunks."""
    print(f"Loading embedding model: {model_name}...")
    model = SentenceTransformer(model_name)
    texts = [chunk["text"] for chunk in chunks]
    print(f"Generating embeddings for {len(texts)} chunks...")
    return model.encode(texts, show_progress_bar=True, batch_size=32)


def _tokenize_corpus(chunks: List[Dict]) -> List[List[str]]:
    """Tokenize all chunks for BM25 sparse vectors.

    Matches the legacy BM25 indexing behaviour: source-file tokens are
    prepended so product-identity terms carry extra weight.
    """
    tokenized: List[List[str]] = []
    for chunk in tqdm(chunks, desc="Tokenizing for sparse vectors"):
        text = chunk.get("text", "")
        source = chunk.get("source_file", "")
        tokens = tokenize_for_bm25(text)
        if source:
            src_tokens = tokenize_for_bm25(
                source.replace("_", " ").replace("|", " ")
            )
            tokens = src_tokens + tokens
        tokenized.append(tokens)
    return tokenized


# ── Qdrant backend ────────────────────────────────────────────────────────

def get_qdrant_client(url: str, api_key: str = None):
    """Create a QdrantClient instance."""
    from qdrant_client import QdrantClient

    kwargs: Dict = {"url": url, "timeout": 120}
    if api_key:
        kwargs["api_key"] = api_key
    return QdrantClient(**kwargs)


def create_qdrant_collection(client, collection_name: str, dense_dim: int):
    """Create (or recreate) a Qdrant collection with dense + sparse vectors
    and payload indexes for filtering."""
    from qdrant_client.models import (
        Distance,
        PayloadSchemaType,
        SparseVectorParams,
        TextIndexParams,
        TokenizerType,
        VectorParams,
    )

    client.recreate_collection(
        collection_name=collection_name,
        vectors_config={
            "dense": VectorParams(size=dense_dim, distance=Distance.COSINE),
        },
        sparse_vectors_config={
            "sparse": SparseVectorParams(),
        },
    )

    client.create_payload_index(
        collection_name=collection_name,
        field_name="source_file",
        field_schema=PayloadSchemaType.KEYWORD,
    )
    client.create_payload_index(
        collection_name=collection_name,
        field_name="source_file_text",
        field_schema=TextIndexParams(
            type="text",
            tokenizer=TokenizerType.WORD,
            min_token_len=2,
            max_token_len=40,
            lowercase=True,
        ),
    )
    client.create_payload_index(
        collection_name=collection_name,
        field_name="chunk_index",
        field_schema=PayloadSchemaType.INTEGER,
    )
    client.create_payload_index(
        collection_name=collection_name,
        field_name="chunk_id",
        field_schema=PayloadSchemaType.KEYWORD,
    )
    print(
        f"Created Qdrant collection '{collection_name}' "
        f"(dense={dense_dim}, sparse, indexed payloads)"
    )


def upsert_to_qdrant(
    client,
    collection_name: str,
    chunks: List[Dict],
    dense_embeddings: np.ndarray,
    sparse_vectors: List[Tuple[List[int], List[float]]],
    batch_size: int = 100,
):
    """Batch-upsert points with dense + sparse vectors and metadata."""
    from qdrant_client.models import PointStruct, SparseVector

    total = len(chunks)
    for start in tqdm(range(0, total, batch_size), desc="Upserting to Qdrant"):
        end = min(start + batch_size, total)
        points: List[PointStruct] = []
        for i in range(start, end):
            chunk = chunks[i]
            s_idx, s_val = sparse_vectors[i]
            source = chunk.get("source_file", "")
            points.append(
                PointStruct(
                    id=i,
                    vector={
                        "dense": dense_embeddings[i].tolist(),
                        "sparse": SparseVector(indices=s_idx, values=s_val),
                    },
                    payload={
                        "chunk_id": chunk.get("chunk_id", ""),
                        "source_file": source,
                        "source_file_text": source,
                        "chunk_index": chunk.get("chunk_index", 0),
                        "text": chunk.get("text", ""),
                        "total_chunks": chunk.get("total_chunks", 1),
                    },
                )
            )
        client.upsert(collection_name=collection_name, points=points)

    print(f"Upserted {total} points to '{collection_name}'")


def embed_and_index_qdrant(
    chunks_file: str,
    collection_name: str,
    qdrant_url: str,
    qdrant_api_key: str = None,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
):
    """Generate embeddings and upsert to a Qdrant collection."""
    print(f"Loading chunks from '{chunks_file}'...")
    chunks = load_chunks(chunks_file)
    if not chunks:
        print("No chunks found to embed.")
        return

    print(f"Found {len(chunks)} chunks")

    embeddings = generate_embeddings(chunks, model_name)

    tokenized = _tokenize_corpus(chunks)
    stats = compute_corpus_stats(tokenized)
    sparse_vecs = [
        build_sparse_vector(toks, stats)
        for toks in tqdm(tokenized, desc="Building sparse vectors")
    ]

    client = get_qdrant_client(qdrant_url, qdrant_api_key)
    dense_dim = embeddings.shape[1]
    create_qdrant_collection(client, collection_name, dense_dim)
    upsert_to_qdrant(
        client, collection_name, chunks,
        embeddings.astype("float32"), sparse_vecs,
    )

    info = client.get_collection(collection_name)
    print(f"\n{'=' * 60}")
    print("Qdrant indexing complete!")
    print(f"{'=' * 60}")
    print(f"  Collection: {collection_name}")
    print(f"  Points:     {info.points_count}")
    print(f"  Dense dim:  {dense_dim}")
    print(f"  Sparse vocab: {len(stats['idf']):,} tokens")


# ── Legacy FAISS + BM25 backend (kept for migration) ─────────────────────

def create_faiss_index(
    embeddings: np.ndarray, chunks: List[Dict], output_dir: str,
):
    """Create FAISS index and save with metadata."""
    import faiss

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    faiss.normalize_L2(embeddings)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)

    print(f"Adding {len(embeddings)} vectors to FAISS index...")
    index.add(embeddings.astype('float32'))

    index_file = output_path / "faiss.index"
    faiss.write_index(index, str(index_file))
    print(f"Saved FAISS index to {index_file}")

    metadata_file = output_path / "metadata.jsonl"
    with open(metadata_file, 'w', encoding='utf-8') as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + '\n')
    print(f"Saved metadata to {metadata_file}")
    return index


def create_bm25_index(chunks: List[Dict], output_dir: str):
    """Create BM25 index for keyword search."""
    from rank_bm25 import BM25Okapi

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    print("Building BM25 index for keyword search...")

    tokenized_corpus: List[List[str]] = []
    for chunk in tqdm(chunks, desc="Tokenizing for BM25"):
        text = chunk.get("text", "")
        source = chunk.get("source_file", "")
        text_tokens = tokenize_for_bm25(text)
        if source:
            source_tokens = tokenize_for_bm25(
                source.replace("_", " ").replace("|", " ")
            )
            text_tokens = source_tokens + text_tokens
        tokenized_corpus.append(text_tokens)

    bm25 = BM25Okapi(tokenized_corpus)
    bm25_file = output_path / "bm25.pkl"
    with open(bm25_file, 'wb') as f:
        pickle.dump({'bm25': bm25, 'tokenized_corpus': tokenized_corpus}, f)
    print(f"Saved BM25 index to {bm25_file}")
    return bm25


def embed_and_index(
    chunks_file: str,
    output_dir: str,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
):
    """Legacy FAISS + BM25 indexing (used when Qdrant is not configured)."""
    print(f"Loading chunks from '{chunks_file}'...")
    chunks = load_chunks(chunks_file)
    if not chunks:
        print("No chunks found to embed.")
        return

    print(f"Found {len(chunks)} chunks")
    embeddings = generate_embeddings(chunks, model_name)
    create_faiss_index(embeddings, chunks, output_dir)
    create_bm25_index(chunks, output_dir)

    print(f"\n{'=' * 60}")
    print("Hybrid indexing complete!")
    print(f"{'=' * 60}")
    print(f"  Total chunks: {len(chunks)}")
    print(f"  FAISS index: {output_dir}/faiss.index")
    print(f"  BM25 index:  {output_dir}/bm25.pkl")
    print(f"  Metadata:    {output_dir}/metadata.jsonl")


# ── CLI ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate embeddings and create search index"
    )
    parser.add_argument(
        "--input", default="data/chunks.jsonl",
        help="Input JSONL file with chunks (default: data/chunks.jsonl)",
    )
    parser.add_argument(
        "--output", default="embeddings/faiss_index/",
        help="Output dir for legacy FAISS index (ignored with --qdrant-url)",
    )
    parser.add_argument(
        "--model", default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence transformer model name",
    )
    parser.add_argument(
        "--qdrant-url", default=os.getenv("QDRANT_URL"),
        help="Qdrant server URL (default: $QDRANT_URL)",
    )
    parser.add_argument(
        "--qdrant-api-key", default=os.getenv("QDRANT_API_KEY"),
        help="Qdrant API key (default: $QDRANT_API_KEY)",
    )
    parser.add_argument(
        "--collection",
        default=os.getenv("QDRANT_COLLECTION_PRIMARY", "supermicro_primary"),
        help="Qdrant collection name",
    )

    args = parser.parse_args()

    if args.qdrant_url:
        embed_and_index_qdrant(
            args.input, args.collection,
            args.qdrant_url, args.qdrant_api_key, args.model,
        )
    else:
        embed_and_index(args.input, args.output, args.model)
