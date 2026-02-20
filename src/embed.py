#!/usr/bin/env python3
"""
Generate embeddings for text chunks and create FAISS vector index + BM25 index.
Supports hybrid search (semantic + keyword).
"""

import json
import pickle
import re
import argparse
from pathlib import Path
from typing import List, Dict
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from rank_bm25 import BM25Okapi


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
    """Prepend source-file metadata so FAISS/BM25 encode product identity."""
    source = chunk.get('source_file', '')
    text = chunk.get('text', '')
    if not source:
        return text

    # Extract product codes from the raw filename (before normalising separators)
    product_codes = [pc.upper() for pc in _PRODUCT_CODE_RE.findall(source)]

    # Build a readable label from the filename
    name = _EXT_RE.sub('', source)
    name = _HASH_RE.sub('', name)
    name = _PREFIX_RE.sub('', name)
    label = re.sub(r'[_|]+', ' ', name).replace('-', ' ')
    label = re.sub(r'\s+', ' ', label).strip()

    # Detect platform generations and add cross-aliases
    platforms = _PLATFORM_RE.findall(label)
    platform_tags = []
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
    
    tokenized_corpus = []
    
    for chunk in tqdm(chunks, desc="Tokenizing for BM25"):
        text = chunk.get("text", "")
        source = chunk.get("source_file", "")
        
        text_tokens = tokenize_for_bm25(text)
        
        # Boost filename tokens so product identity terms from the source
        # file name carry extra weight (e.g., "sys-521ge-tnrt" from filename)
        if source:
            source_tokens = tokenize_for_bm25(source.replace("_", " ").replace("|", " "))
            text_tokens = source_tokens + text_tokens
        
        tokenized_corpus.append(text_tokens)
    
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
