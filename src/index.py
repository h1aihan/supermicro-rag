#!/usr/bin/env python3
"""
Load and manage FAISS vector index + BM25 keyword index for hybrid search.
"""

import json
import os
import re
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


def tokenize_for_bm25(text: str) -> List[str]:
    """
    Tokenizer for BM25 optimized for Supermicro product codes.
    
    For product codes like SYS-521GE-TNRT, generates both:
    - The full hyphenated token: sys-521ge-tnrt
    - Individual parts: sys, 521ge, tnrt
    
    This allows queries like "521GE" to match "SYS-521GE-TNRT".
    """
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


class HybridIndex:
    """
    Hybrid search index combining FAISS (semantic) and BM25 (keyword) search.
    Uses Reciprocal Rank Fusion (RRF) to combine results.
    """
    
    def __init__(self, index_dir: str, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the hybrid index.
        
        Args:
            index_dir: Directory containing FAISS index, BM25 index, and metadata
            model_name: Name of the sentence transformer model
        """
        self.index_dir = Path(index_dir)
        self.model_name = model_name
        self.model = None
        self.faiss_index = None
        self.bm25 = None
        self.metadata = []
        
        self._load_faiss_index()
        self._load_bm25_index()
        self._load_metadata()
        self._load_model()
    
    def _load_faiss_index(self):
        """Load FAISS index from disk."""
        index_file = self.index_dir / "faiss.index"
        if not index_file.exists():
            raise FileNotFoundError(f"FAISS index not found at {index_file}")

        use_mmap = (os.getenv("FAISS_MMAP", "1").strip() != "0")
        io_flags = faiss.IO_FLAG_MMAP if use_mmap else 0
        self.faiss_index = faiss.read_index(str(index_file), io_flags)
        print(f"Loaded FAISS index with {self.faiss_index.ntotal} vectors")
    
    def _load_bm25_index(self):
        """Load BM25 index from disk."""
        bm25_file = self.index_dir / "bm25.pkl"
        if not bm25_file.exists():
            print(f"Warning: BM25 index not found at {bm25_file}, keyword search disabled")
            self.bm25 = None
            return
        
        with open(bm25_file, 'rb') as f:
            data = pickle.load(f)
            self.bm25 = data['bm25']
        print(f"Loaded BM25 index for keyword search")
    
    def _load_metadata(self):
        """Load chunk metadata from JSONL file."""
        metadata_file = self.index_dir / "metadata.jsonl"
        if not metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found at {metadata_file}")
        
        with open(metadata_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.metadata.append(json.loads(line))
        
        print(f"Loaded metadata for {len(self.metadata)} chunks")
    
    def _load_model(self):
        """Load sentence transformer model."""
        print(f"Loading embedding model: {self.model_name}...")
        self.model = SentenceTransformer(self.model_name)
    
    def search_semantic(self, query: str, top_k: int = 20) -> List[Tuple[int, float]]:
        """
        Semantic search using FAISS.
        
        Returns:
            List of (index, score) tuples
        """
        query_embedding = self.model.encode([query])
        faiss.normalize_L2(query_embedding)
        scores, indices = self.faiss_index.search(query_embedding.astype('float32'), top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self.metadata):
                results.append((int(idx), float(score)))
        return results
    
    def search_keyword(self, query: str, top_k: int = 20) -> List[Tuple[int, float]]:
        """
        Keyword search using BM25.
        
        Returns:
            List of (index, score) tuples
        """
        if self.bm25 is None:
            return []
        
        tokens = tokenize_for_bm25(query)
        scores = self.bm25.get_scores(tokens)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include non-zero scores
                results.append((int(idx), float(scores[idx])))
        return results
    
    def _is_product_code_query(self, query: str) -> bool:
        """
        Detect if query looks like a product code/identifier.
        
        Product codes typically:
        - Are short (1-3 words)
        - Contain alphanumeric patterns with numbers
        - Match patterns like "521GE", "SYS-521GE", "X13DEI", etc.
        """
        words = query.strip().split()
        
        # Short queries (1-3 words) with alphanumeric codes
        if len(words) <= 3:
            for word in words:
                # Check if word looks like a product code (has both letters and numbers)
                has_letter = any(c.isalpha() for c in word)
                has_digit = any(c.isdigit() for c in word)
                if has_letter and has_digit:
                    return True
                # Also check for known prefixes
                if word.upper().startswith(('SYS-', 'AS-', 'SSG-', 'SBI-', 'AOC-', 'X1', 'H1')):
                    return True
        return False
    
    def search_hybrid(
        self, 
        query: str, 
        top_k: int = 10,
        semantic_weight: float = None,  # None = auto-detect
        keyword_weight: float = None,   # None = auto-detect
        rrf_k: int = 60
    ) -> List[Tuple[Dict, float]]:
        """
        Hybrid search combining semantic and keyword search using Reciprocal Rank Fusion.
        
        Uses adaptive weighting based on query type:
        - Product code queries (e.g., "521GE"): 80% BM25, 20% semantic
        - Natural language queries: 50% BM25, 50% semantic
        
        Args:
            query: Search query
            top_k: Number of results to return
            semantic_weight: Weight for semantic search (None = auto-detect)
            keyword_weight: Weight for keyword search (None = auto-detect)
            rrf_k: RRF constant (higher = more weight to lower-ranked items)
            
        Returns:
            List of (chunk_dict, score) tuples
        """
        # Auto-detect weights based on query type
        if semantic_weight is None or keyword_weight is None:
            if self._is_product_code_query(query):
                # Product code queries: heavily favor BM25
                semantic_weight = 0.2
                keyword_weight = 0.8
            else:
                # Natural language queries: balanced
                semantic_weight = 0.5
                keyword_weight = 0.5
        
        # Get results from both methods (fetch more for fusion)
        fetch_k = top_k * 3
        semantic_results = self.search_semantic(query, fetch_k)
        keyword_results = self.search_keyword(query, fetch_k)
        
        # Calculate RRF scores
        rrf_scores = {}
        
        # Add semantic scores
        for rank, (idx, _) in enumerate(semantic_results):
            rrf_score = semantic_weight * (1.0 / (rrf_k + rank + 1))
            rrf_scores[idx] = rrf_scores.get(idx, 0) + rrf_score
        
        # Add keyword scores
        for rank, (idx, _) in enumerate(keyword_results):
            rrf_score = keyword_weight * (1.0 / (rrf_k + rank + 1))
            rrf_scores[idx] = rrf_scores.get(idx, 0) + rrf_score
        
        # Sort by combined RRF score
        sorted_indices = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Return top_k results (no per-source limit - for product datasheets we want
        # multiple chunks from the same relevant document)
        results = []
        for idx, score in sorted_indices[:top_k]:
            if idx < len(self.metadata):
                results.append((self.metadata[idx], score))
        
        return results
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[Dict, float]]:
        """
        Default search method - uses hybrid search.
        
        Args:
            query: Query text
            top_k: Number of results to return
            
        Returns:
            List of tuples (chunk_dict, similarity_score)
        """
        return self.search_hybrid(query, top_k)


# Backward compatibility alias
VectorIndex = HybridIndex


if __name__ == "__main__":
    # Test hybrid search
    import argparse
    
    parser = argparse.ArgumentParser(description="Test hybrid search (FAISS + BM25)")
    parser.add_argument(
        "--index-dir",
        default="embeddings/faiss_index/",
        help="Directory containing indexes (default: embeddings/faiss_index/)"
    )
    parser.add_argument(
        "--query",
        help="Test query to search"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of results (default: 5)"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare semantic vs keyword vs hybrid results"
    )
    
    args = parser.parse_args()
    
    try:
        index = HybridIndex(args.index_dir)
        
        if args.query:
            print(f"\n{'='*60}")
            print(f"Query: '{args.query}'")
            print(f"{'='*60}")
            
            if args.compare:
                # Show tokenized query and detected type
                query_tokens = tokenize_for_bm25(args.query)
                is_product_code = index._is_product_code_query(args.query)
                print(f"\nQuery tokens: {query_tokens}")
                print(f"Detected as product code: {is_product_code}")
                if is_product_code:
                    print(f"Using weights: 20% semantic, 80% BM25")
                else:
                    print(f"Using weights: 50% semantic, 50% BM25")
                
                # Compare all three methods
                print("\n--- SEMANTIC SEARCH (FAISS) ---")
                semantic = index.search_semantic(args.query, args.top_k)
                for i, (idx, score) in enumerate(semantic, 1):
                    chunk = index.metadata[idx]
                    print(f"{i}. [{score:.4f}] {chunk['source_file']}")
                
                print("\n--- KEYWORD SEARCH (BM25) ---")
                keyword = index.search_keyword(args.query, args.top_k)
                if not keyword:
                    print("  (No BM25 results found!)")
                for i, (idx, score) in enumerate(keyword, 1):
                    chunk = index.metadata[idx]
                    print(f"{i}. [{score:.4f}] {chunk['source_file']}")
                
                print("\n--- HYBRID SEARCH (RRF) ---")
                hybrid = index.search_hybrid(args.query, args.top_k)
                for i, (chunk, score) in enumerate(hybrid, 1):
                    print(f"{i}. [{score:.6f}] {chunk['source_file']}")
            else:
                # Just show hybrid results
                results = index.search(args.query, args.top_k)
                print(f"\nTop {len(results)} hybrid results:")
                for i, (chunk, score) in enumerate(results, 1):
                    print(f"\n{i}. Score: {score:.6f}")
                    print(f"   Source: {chunk['source_file']}")
                    print(f"   Text preview: {chunk['text'][:200]}...")
        else:
            print("\nHybrid index loaded successfully!")
            print("  FAISS: semantic search")
            print("  BM25: keyword search")
            print("\nUse --query to test, --compare to see all methods side by side")
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
