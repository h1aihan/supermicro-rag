#!/usr/bin/env python3
"""
Generate embeddings for text chunks and create FAISS vector index.
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


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


def embed_and_index(chunks_file: str, output_dir: str, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """
    Generate embeddings and create FAISS index.
    
    Args:
        chunks_file: Path to JSONL file containing chunks
        output_dir: Directory to save FAISS index
        model_name: Name of the sentence transformer model
    """
    print(f"Loading chunks from '{chunks_file}'...")
    chunks = load_chunks(chunks_file)
    
    if not chunks:
        print("No chunks found to embed.")
        return
    
    print(f"Found {len(chunks)} chunks")
    
    # Generate embeddings
    embeddings = generate_embeddings(chunks, model_name)
    
    # Create FAISS index
    index = create_faiss_index(embeddings, chunks, output_dir)
    
    print(f"\nEmbedding and indexing complete!")
    print(f"  Total vectors: {len(chunks)}")
    print(f"  Embedding dimension: {embeddings.shape[1]}")
    print(f"  Index saved to: {output_dir}")


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
