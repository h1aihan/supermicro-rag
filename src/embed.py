#!/usr/bin/env python3
"""
Generate embeddings for text chunks and create FAISS vector index + BM25 index.
Supports hybrid search (semantic + keyword).
"""

import json
import pickle
import argparse
from pathlib import Path
from typing import List, Dict
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from rank_bm25 import BM25Okapi


def tokenize_for_bm25(text: str) -> List[str]:
    """
    Tokenizer for BM25 optimized for Supermicro product codes.
    
    For product codes like SYS-521GE-TNRT, generates both:
    - The full hyphenated token: sys-521ge-tnrt
    - Individual parts: sys, 521ge, tnrt
    
    This allows queries like "521GE" to match "SYS-521GE-TNRT".
    """
    import re
    text = text.lower()
    
    # Find all word tokens (with hyphens)
    hyphenated_tokens = re.findall(r'\b[\w]+-[\w-]+\b', text)
    
    # Find simple word tokens
    simple_tokens = re.findall(r'\b\w+\b', text)
    
    # Combine: keep hyphenated tokens AND their parts
    tokens = []
    for token in hyphenated_tokens:
        tokens.append(token)  # Keep full token: sys-521ge-tnrt
        parts = token.split('-')
        tokens.extend(parts)  # Add parts: sys, 521ge, tnrt
    
    # Add simple tokens
    tokens.extend(simple_tokens)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_tokens = []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            unique_tokens.append(t)
    
    return unique_tokens


def load_chunks(chunks_file: str) -> List[Dict]:
    """
    Load chunks from JSONL file.
    
    Args:
        chunks_file: Path to JSONL file containing chunks
        
    Returns:
        List of chunk dictionaries
    """
    chunks = []
    with open(chunks_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
    return chunks


def generate_embeddings(chunks: List[Dict], model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> np.ndarray:
    """
    Generate embeddings for all chunks.
    
    Args:
        chunks: List of chunk dictionaries
        model_name: Name of the sentence transformer model
        
    Returns:
        Numpy array of embeddings
    """
    print(f"Loading embedding model: {model_name}...")
    model = SentenceTransformer(model_name)
    
    texts = [chunk["text"] for chunk in chunks]
    
    print(f"Generating embeddings for {len(texts)} chunks...")
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)
    
    return embeddings


def create_faiss_index(embeddings: np.ndarray, chunks: List[Dict], output_dir: str):
    """
    Create FAISS index and save with metadata.
    
    Args:
        embeddings: Numpy array of embeddings
        chunks: List of chunk dictionaries
        output_dir: Directory to save FAISS index
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Normalize embeddings for cosine similarity
    faiss.normalize_L2(embeddings)
    
    # Create FAISS index (L2 normalized vectors, inner product = cosine similarity)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
    
    print(f"Adding {len(embeddings)} vectors to FAISS index...")
    index.add(embeddings.astype('float32'))
    
    # Save index
    index_file = output_path / "faiss.index"
    faiss.write_index(index, str(index_file))
    print(f"Saved FAISS index to {index_file}")
    
    # Save metadata (chunk info for each vector)
    metadata_file = output_path / "metadata.jsonl"
    with open(metadata_file, 'w', encoding='utf-8') as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + '\n')
    print(f"Saved metadata to {metadata_file}")
    
    return index


def create_bm25_index(chunks: List[Dict], output_dir: str):
    """
    Create BM25 index for keyword search.
    
    Args:
        chunks: List of chunk dictionaries
        output_dir: Directory to save BM25 index
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("Building BM25 index for keyword search...")
    
    # Tokenize all chunks (include source filename with boosted weight)
    tokenized_corpus = []
    FILENAME_BOOST = 5  # Repeat filename tokens N times to boost their weight
    
    for chunk in tqdm(chunks, desc="Tokenizing for BM25"):
        # Get source filename (e.g., "sys-521ge-tnrt.pdf")
        source_file = chunk.get("source_file", "")
        
        # Tokenize filename separately and repeat for boosting
        filename_tokens = tokenize_for_bm25(source_file)
        boosted_filename_tokens = filename_tokens * FILENAME_BOOST
        
        # Tokenize the chunk text
        text_tokens = tokenize_for_bm25(chunk["text"])
        
        # Combine: boosted filename tokens + text tokens
        tokens = boosted_filename_tokens + text_tokens
        tokenized_corpus.append(tokens)
    
    # Create BM25 index
    bm25 = BM25Okapi(tokenized_corpus)
    
    # Save BM25 index
    bm25_file = output_path / "bm25.pkl"
    with open(bm25_file, 'wb') as f:
        pickle.dump({
            'bm25': bm25,
            'tokenized_corpus': tokenized_corpus
        }, f)
    print(f"Saved BM25 index to {bm25_file}")
    
    return bm25


def embed_and_index(chunks_file: str, output_dir: str, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """
    Generate embeddings and create FAISS + BM25 indexes for hybrid search.
    
    Args:
        chunks_file: Path to JSONL file containing chunks
        output_dir: Directory to save indexes
        model_name: Name of the sentence transformer model
    """
    print(f"Loading chunks from '{chunks_file}'...")
    chunks = load_chunks(chunks_file)
    
    if not chunks:
        print("No chunks found to embed.")
        return
    
    print(f"Found {len(chunks)} chunks")
    
    # Generate embeddings and create FAISS index (semantic search)
    embeddings = generate_embeddings(chunks, model_name)
    faiss_index = create_faiss_index(embeddings, chunks, output_dir)
    
    # Create BM25 index (keyword search)
    bm25_index = create_bm25_index(chunks, output_dir)
    
    print(f"\n{'='*60}")
    print(f"Hybrid indexing complete!")
    print(f"{'='*60}")
    print(f"  Total chunks: {len(chunks)}")
    print(f"  FAISS index: {output_dir}/faiss.index")
    print(f"  BM25 index: {output_dir}/bm25.pkl")
    print(f"  Metadata: {output_dir}/metadata.jsonl")
    print(f"\nHybrid search enabled: semantic (FAISS) + keyword (BM25)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate embeddings and create FAISS index"
    )
    parser.add_argument(
        "--input",
        default="data/chunks.jsonl",
        help="Input JSONL file with chunks (default: data/chunks.jsonl)"
    )
    parser.add_argument(
        "--output",
        default="embeddings/faiss_index/",
        help="Output directory for FAISS index (default: embeddings/faiss_index/)"
    )
    parser.add_argument(
        "--model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence transformer model name (default: sentence-transformers/all-MiniLM-L6-v2)"
    )
    
    args = parser.parse_args()
    embed_and_index(args.input, args.output, args.model)
