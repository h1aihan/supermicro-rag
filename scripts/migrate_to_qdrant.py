#!/usr/bin/env python3
"""
One-time migration: read existing FAISS indexes + metadata and upload
everything to Qdrant collections.

Usage:
    # Start Qdrant first
    docker compose up -d qdrant

    # Run migration (uses QDRANT_URL from .env or defaults to localhost)
    python scripts/migrate_to_qdrant.py

    # Or specify paths explicitly
    python scripts/migrate_to_qdrant.py \
        --primary-dir embeddings/primary_index/ \
        --manual-dir  embeddings/manual_index/ \
        --qdrant-url  http://localhost:6333
"""

import json
import os
import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

import numpy as np
from tqdm import tqdm

from src.embed import (
    build_sparse_vector,
    compute_corpus_stats,
    create_qdrant_collection,
    get_qdrant_client,
    load_chunks,
    tokenize_for_bm25,
    upsert_to_qdrant,
)


def _load_faiss_vectors(index_dir: Path) -> np.ndarray:
    """Extract all vectors from an existing FAISS index."""
    import faiss

    index_path = index_dir / "faiss.index"
    if not index_path.exists():
        raise FileNotFoundError(f"FAISS index not found: {index_path}")

    index = faiss.read_index(str(index_path))
    n = index.ntotal
    dim = index.d
    print(f"  FAISS index: {n:,} vectors x {dim} dims")

    vectors = np.zeros((n, dim), dtype="float32")
    for i in range(n):
        vectors[i] = index.reconstruct(i)
    return vectors


def _load_metadata(index_dir: Path):
    """Load metadata.jsonl from an index directory."""
    meta_path = index_dir / "metadata.jsonl"
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata not found: {meta_path}")
    return load_chunks(str(meta_path))


def _tokenize_chunks(chunks):
    """Tokenize chunks for sparse vector generation."""
    tokenized = []
    for chunk in tqdm(chunks, desc="  Tokenizing"):
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


def migrate_index(
    index_dir: Path,
    collection_name: str,
    client,
):
    """Migrate a single FAISS index directory to a Qdrant collection."""
    print(f"\n{'=' * 60}")
    print(f"Migrating: {index_dir} -> '{collection_name}'")
    print(f"{'=' * 60}")

    vectors = _load_faiss_vectors(index_dir)
    chunks = _load_metadata(index_dir)

    if len(vectors) != len(chunks):
        print(f"  WARNING: vector count ({len(vectors)}) != metadata count "
              f"({len(chunks)}). Using min of both.")
        n = min(len(vectors), len(chunks))
        vectors = vectors[:n]
        chunks = chunks[:n]

    tokenized = _tokenize_chunks(chunks)
    stats = compute_corpus_stats(tokenized)
    sparse_vecs = [
        build_sparse_vector(toks, stats)
        for toks in tqdm(tokenized, desc="  Sparse vectors")
    ]

    dense_dim = vectors.shape[1]
    create_qdrant_collection(client, collection_name, dense_dim)
    upsert_to_qdrant(client, collection_name, chunks, vectors, sparse_vecs)

    info = client.get_collection(collection_name)
    print(f"  Done: {info.points_count:,} points in '{collection_name}'")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Migrate FAISS indexes to Qdrant collections",
    )
    parser.add_argument(
        "--primary-dir",
        default="embeddings/primary_index/",
        help="Primary FAISS index directory",
    )
    parser.add_argument(
        "--manual-dir",
        default="embeddings/manual_index/",
        help="Manual FAISS index directory",
    )
    parser.add_argument(
        "--qdrant-url",
        default=os.getenv("QDRANT_URL", "http://localhost:6333"),
    )
    parser.add_argument(
        "--qdrant-api-key",
        default=os.getenv("QDRANT_API_KEY"),
    )
    parser.add_argument(
        "--primary-collection",
        default=os.getenv("QDRANT_COLLECTION_PRIMARY", "supermicro_primary"),
    )
    parser.add_argument(
        "--manual-collection",
        default=os.getenv("QDRANT_COLLECTION_MANUAL", "supermicro_manual"),
    )
    args = parser.parse_args()

    client = get_qdrant_client(args.qdrant_url, args.qdrant_api_key)

    primary_dir = Path(args.primary_dir)
    manual_dir = Path(args.manual_dir)

    if primary_dir.exists() and (primary_dir / "faiss.index").exists():
        migrate_index(primary_dir, args.primary_collection, client)
    else:
        print(f"Skipping primary: {primary_dir} not found")

    if manual_dir.exists() and (manual_dir / "faiss.index").exists():
        migrate_index(manual_dir, args.manual_collection, client)
    else:
        print(f"Skipping manual: {manual_dir} not found")

    print(f"\n{'=' * 60}")
    print("Migration complete!")
    print(f"{'=' * 60}")
    print(f"\nCollections ready at {args.qdrant_url}")
    print(f"  Primary: {args.primary_collection}")
    print(f"  Manual:  {args.manual_collection}")
    print(f"\nStart the app: uvicorn src.server:app --host 0.0.0.0 --port 8000")


if __name__ == "__main__":
    main()
