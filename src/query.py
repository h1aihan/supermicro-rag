#!/usr/bin/env python3
"""
Query processing and retrieval system with hybrid search and cross-encoder reranking.
Combines semantic search (FAISS) with keyword search (BM25) and reranks with cross-encoder.
"""

import re
import os
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

# Support running as:
# - python -m src.query   (package mode)
# - python src/query.py   (script mode)
try:
    from src.index import HybridIndex
except ImportError:
    from index import HybridIndex


# =============================================================================
# Query Preprocessing - Expand product codes
# =============================================================================

# Common Supermicro product prefixes
PRODUCT_PREFIXES = ['SYS-', 'AS-', 'SBI-', 'AOC-', 'PWS-', 'BPN-', 'CSE-', 'SC', 'X']

def preprocess_query(query: str) -> str:
    """
    Lightweight preprocessing: expand bare product codes with SYS-/AS- prefixes.
    
    The LLM query planner already handles product code normalization and
    platform disambiguation (X-series = Intel, H-series = AMD), so this
    function only adds prefix variants for bare codes like "521GE".
    """
    words = query.split()
    expanded_terms = list(words)

    # --- Prefix expansion for bare product codes (e.g. "521GE" → SYS-521GE) ---
    _SKIP_EXPAND = {'1u','2u','4u','8u','10u','h12','h13','h14','x12','x13','x14','b200','ddr4','ddr5'}
    for word in words:
        if word.lower() in _SKIP_EXPAND:
            continue
        has_digit = any(c.isdigit() for c in word)
        if not has_digit:
            continue
        if re.match(r'^[A-Z0-9][-A-Z0-9]*$', word, re.IGNORECASE):
            has_prefix = any(word.upper().startswith(p) for p in PRODUCT_PREFIXES)
            if not has_prefix:
                expanded_terms.append(f"SYS-{word}")
                expanded_terms.append(f"AS-{word}")

    return ' '.join(expanded_terms)


# =============================================================================
# Cross-encoder Reranking
# =============================================================================

_reranker = None

def get_reranker():
    """Lazy-load the cross-encoder model for reranking."""
    global _reranker
    if _reranker is None:
        # Only load if reranking is enabled
        if os.getenv("ENABLE_RERANKING", "1") != "0":
            try:
                from sentence_transformers import CrossEncoder
                print("Loading cross-encoder for reranking...")
                _reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
                print("Cross-encoder loaded successfully")
            except Exception as e:
                print(f"Warning: Could not load cross-encoder: {e}")
                _reranker = False  # Mark as failed, don't retry
    return _reranker if _reranker else None


def _is_product_code_query(query: str) -> bool:
    """Detect if query contains a product code/identifier (e.g., SYS-521GE-TNRT)."""
    import re
    # Check if query contains any Supermicro product code pattern
    product_pattern = r'\b(?:SYS|AS|SSG|SBI|AOC|MBD|X1[0-9]|H1[0-9])-[\w-]+\b'
    if re.search(product_pattern, query, re.IGNORECASE):
        return True
    
    # Also check for short alphanumeric codes (e.g., "521GE", "X13DEI")
    words = query.strip().split()
    if len(words) <= 3:
        for word in words:
            has_letter = any(c.isalpha() for c in word)
            has_digit = any(c.isdigit() for c in word)
            if has_letter and has_digit:
                return True
    return False


def rerank_chunks(query: str, chunks: List[Dict], top_k: int = 10) -> List[Dict]:
    """
    Rerank chunks using a cross-encoder model.
    
    For product code queries, blends original ranking with cross-encoder scores
    to preserve BM25 filename matches which are highly reliable.
    
    Args:
        query: The user query
        chunks: List of chunk dictionaries with 'text' field
        top_k: Number of results to return after reranking
        
    Returns:
        Reranked list of chunks
    """
    reranker = get_reranker()
    if reranker is None or len(chunks) == 0:
        return chunks[:top_k]
    
    is_product_query = _is_product_code_query(query)
    
    # Create query-document pairs for the cross-encoder
    pairs = [(query, chunk['text']) for chunk in chunks]
    
    # Get cross-encoder scores
    ce_scores = reranker.predict(pairs)
    
    # Normalize cross-encoder scores to 0-1 range
    ce_min, ce_max = min(ce_scores), max(ce_scores)
    if ce_max > ce_min:
        ce_scores_norm = [(s - ce_min) / (ce_max - ce_min) for s in ce_scores]
    else:
        ce_scores_norm = [0.5] * len(ce_scores)
    
    # Combine scores
    scored_chunks = []
    for i, (chunk, ce_score, ce_norm) in enumerate(zip(chunks, ce_scores, ce_scores_norm)):
        # Original rank score (higher for earlier positions)
        original_rank_score = 1.0 - (i / len(chunks))
        
        if is_product_query:
            # For product codes: 60% original ranking (preserves BM25 filename matches), 40% cross-encoder
            combined_score = 0.6 * original_rank_score + 0.4 * ce_norm
        else:
            # For natural language: 30% original ranking, 70% cross-encoder
            combined_score = 0.3 * original_rank_score + 0.7 * ce_norm
        
        scored_chunks.append((chunk, ce_score, combined_score))
    
    # Sort by combined score
    scored_chunks.sort(key=lambda x: x[2], reverse=True)
    
    # Return top-k with updated scores
    reranked = []
    for chunk, ce_score, combined in scored_chunks[:top_k]:
        chunk_copy = chunk.copy()
        chunk_copy['rerank_score'] = float(ce_score)
        reranked.append(chunk_copy)
    
    return reranked


class RAGQueryProcessor:
    """Process queries and retrieve relevant context using hybrid search + reranking."""
    
    def __init__(
        self, 
        index_dir: str, 
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        enable_reranking: bool = True
    ):
        """
        Initialize the query processor with hybrid search (FAISS + BM25) and optional reranking.
        
        Args:
            index_dir: Directory containing FAISS index, BM25 index, and metadata
            model_name: Name of the sentence transformer model
            enable_reranking: Whether to use cross-encoder reranking
        """
        self.index = HybridIndex(index_dir, model_name)
        self.enable_reranking = enable_reranking and (os.getenv("ENABLE_RERANKING", "1") != "0")
    
    def retrieve(
        self, 
        query: str, 
        top_k: int = 10,
        max_per_source: Optional[int] = None,
    ) -> List[Dict]:
        """
        Retrieve relevant chunks using hybrid search.
        
        Pipeline:
        1. Preprocess query (expand product codes like "521GE" → "SYS-521GE")
        2. Hybrid search (FAISS semantic + BM25 keyword)
        
        Args:
            query: User query
            top_k: Number of chunks to return
            max_per_source: If set, cap chunks per source file for diversity
            
        Returns:
            List of chunk dictionaries with similarity scores
        """
        # Step 1: Preprocess query to expand product codes
        expanded_query = preprocess_query(query)
        
        # Step 2: Hybrid search
        results = self.index.search_hybrid(expanded_query, top_k, max_per_source=max_per_source)
        
        # Convert to list of dicts
        chunks = []
        for chunk, score in results:
            chunks.append({
                "text": chunk["text"],
                "source_file": chunk["source_file"],
                "chunk_id": chunk["chunk_id"],
                "similarity_score": score
            })
        
        # Debug output
        sources = [c['source_file'] for c in chunks[:5]]
        print(f"[DEBUG] Query: '{query[:50]}...'")
        print(f"[DEBUG] Top 5 sources: {sources}")
        
        return chunks
    
    def format_context(self, chunks: List[Dict]) -> str:
        """
        Format retrieved chunks into context string for LLM.
        
        Args:
            chunks: List of retrieved chunk dictionaries
            
        Returns:
            Formatted context string
        """
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            header = f"[Source {i}: {chunk['source_file']}]"
            context_parts.append(f"{header}\n{chunk['text']}\n")
        
        return "\n".join(context_parts)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test hybrid search with reranking")
    parser.add_argument(
        "--index-dir",
        default="embeddings/faiss_index/",
        help="Directory containing indexes (default: embeddings/faiss_index/)"
    )
    parser.add_argument(
        "--query",
        required=True,
        help="Query to process"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of results (default: 5)"
    )
    parser.add_argument(
        "--max-per-source",
        type=int,
        default=3,
        help="Max chunks per source document (default: 3)"
    )
    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="Disable cross-encoder reranking"
    )
    
    args = parser.parse_args()
    
    # Disable reranking if requested
    if args.no_rerank:
        os.environ["ENABLE_RERANKING"] = "0"
    
    try:
        print(f"\n{'='*80}")
        print("Initializing RAG Query Processor")
        print(f"{'='*80}")
        
        processor = RAGQueryProcessor(args.index_dir, enable_reranking=not args.no_rerank)
        
        # Show query expansion
        expanded = preprocess_query(args.query)
        if expanded != args.query:
            print(f"\nQuery expansion: '{args.query}' → '{expanded}'")
        
        print(f"\n{'='*80}")
        print(f"SEARCH: '{args.query}'")
        if processor.enable_reranking:
            print("Mode: Hybrid (FAISS + BM25) + Cross-encoder Reranking")
        else:
            print("Mode: Hybrid (FAISS + BM25)")
        print(f"{'='*80}")
        
        chunks = processor.retrieve(args.query, args.top_k, args.max_per_source)
        
        print(f"\nRetrieved {len(chunks)} chunks:\n")
        
        for i, chunk in enumerate(chunks, 1):
            print(f"{'='*80}")
            score_info = f"Score: {chunk['similarity_score']:.6f}"
            if 'rerank_score' in chunk:
                score_info += f" | Rerank: {chunk['rerank_score']:.4f}"
            print(f"Chunk {i} ({score_info})")
            print(f"Source: {chunk['source_file']}")
            print(f"{'='*80}")
            print(chunk['text'][:500])
            print("\n")
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
