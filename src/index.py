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
from collections import defaultdict
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
        self._build_filename_index()
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

    def _build_filename_index(self):
        """Build an inverted index mapping filename tokens to chunk indices."""
        self._filename_index: Dict[str, set] = defaultdict(set)
        for idx, meta in enumerate(self.metadata):
            source = meta.get('source_file', '').lower()
            tokens = set(re.findall(r'\b\w{2,}\b', source.replace('_', ' ').replace('-', ' ')))
            for token in tokens:
                self._filename_index[token].add(idx)
        print(f"Built filename index: {len(self._filename_index)} unique tokens across {len(self.metadata)} chunks")

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

    def search_by_filename(self, query: str, top_k: int = 20) -> List[Tuple[int, float]]:
        """
        Retrieve chunks whose source filename tokens match the query terms.

        Returns:
            List of (index, match_count) tuples, sorted by descending match count.
        """
        _STOPWORDS = {
            'the', 'is', 'at', 'which', 'on', 'for', 'and', 'or', 'to', 'in',
            'of', 'with', 'what', 'how', 'can', 'do', 'does', 'are', 'was',
            'be', 'it', 'its', 'an', 'as', 'by', 'from', 'that', 'this',
            'my', 'me', 'we', 'you', 'your', 'their', 'our', 'into', 'about',
            'please', 'compare', 'suggest', 'recommend', 'show', 'tell',
            'give', 'list', 'between', 'vs', 'versus', 'than', 'should',
            'would', 'could', 'will', 'need', 'want', 'like', 'have', 'has',
            'pdf', 'datasheet', 'spec', 'specs', 'specification', 'specifications',
            'supermicro', 'server', 'servers', 'system', 'systems', 'series',
            'rackmount', 'product', 'products', 'page', 'web', 'txt',
        }
        raw_tokens = re.findall(r'\b\w{2,}\b', query.lower().replace('-', ' ').replace('_', ' '))
        terms = list(dict.fromkeys(
            t for t in raw_tokens if len(t) >= 2 and t not in _STOPWORDS
        ))
        if not terms:
            return []

        chunk_scores: Dict[int, int] = defaultdict(int)
        for term in terms:
            for idx in self._filename_index.get(term, set()):
                chunk_scores[idx] += 1

        ranked = sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True)
        return [(idx, float(score)) for idx, score in ranked[:top_k]]

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
    
    def _is_keyword_heavy_query(self, query: str) -> bool:
        """
        Detect queries that should heavily favor BM25 keyword search.
        
        Used for queries where exact term matching matters more than
        semantic similarity (e.g., specific product names, short keyword queries).
        """
        words = query.strip().split()
        
        # Very short queries (1-2 words) are usually keyword lookups
        if len(words) <= 2:
            return True
        
        return False
    
    def _expand_query_for_bm25(self, query: str) -> str:
        """
        Expand query with stemmed variants for better BM25 matching.
        
        Uses general-purpose suffix rules so ANY query benefits 
        (e.g., "servers"→"server", "skus"→"sku", "golden"→"gold").
        No hardcoded term lists needed.
        """
        words = query.lower().split()
        expansions = set()
        
        for word in words:
            # Strip common English suffixes to create stem variants
            # Plural → singular
            if word.endswith('ies') and len(word) > 4:
                expansions.add(word[:-3] + 'y')  # categories → category
            elif word.endswith('ses') and len(word) > 4:
                expansions.add(word[:-2])  # processes → process
            elif word.endswith('es') and len(word) > 3:
                expansions.add(word[:-2])  # switches → switch
                expansions.add(word[:-1])  # also try just -s removed
            elif word.endswith('s') and not word.endswith('ss') and len(word) > 3:
                expansions.add(word[:-1])  # skus → sku, servers → server
            
            # -en suffix → base (golden → gold)  
            if word.endswith('en') and len(word) > 4:
                expansions.add(word[:-2])  # golden → gold
            
            # -ing suffix → base
            if word.endswith('ing') and len(word) > 5:
                expansions.add(word[:-3])  # computing → comput
                expansions.add(word[:-3] + 'e')  # configuring → configure
            
            # -ed suffix → base  
            if word.endswith('ed') and len(word) > 4:
                expansions.add(word[:-2])  # configured → configur
                expansions.add(word[:-1])  # also try just -d removed
        
        # Remove any expansions that are already in the query
        new_terms = expansions - set(words)
        # Remove very short stems (likely noise)
        new_terms = {t for t in new_terms if len(t) > 2}
        
        if new_terms:
            return query + ' ' + ' '.join(new_terms)
        return query
    
    def search_hybrid(
        self, 
        query: str, 
        top_k: int = 10,
        semantic_weight: float = None,  # None = auto-detect
        keyword_weight: float = None,   # None = auto-detect
        rrf_k: int = 60,
        max_per_source: Optional[int] = None,
    ) -> List[Tuple[Dict, float]]:
        """
        Hybrid search combining semantic and keyword search using Reciprocal Rank Fusion.
        
        Uses adaptive weighting based on query type:
        - Product code queries (e.g., "521GE"): 80% BM25, 20% semantic
        - Keyword-heavy queries (e.g., "gold series"): 75% BM25, 25% semantic
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
            elif self._is_keyword_heavy_query(query):
                # Category/keyword queries: favor BM25 to match exact terms
                semantic_weight = 0.25
                keyword_weight = 0.75
            else:
                # Natural language queries: balanced
                semantic_weight = 0.5
                keyword_weight = 0.5
        
        # Get results from both methods (fetch more for better fusion coverage)
        fetch_k = max(top_k * 5, 30)
        semantic_results = self.search_semantic(query, fetch_k)
        
        # Expand query for BM25 (add synonyms/stemmed variants)
        bm25_query = self._expand_query_for_bm25(query)
        keyword_results = self.search_keyword(bm25_query, fetch_k)
        
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

        # Third channel: filename matching (helps product family queries)
        filename_results = self.search_by_filename(query, fetch_k)
        if filename_results:
            best_match_score = filename_results[0][1]
            filename_weight = 1.0 if best_match_score >= 2 else 0.3
            for rank, (idx, score) in enumerate(filename_results):
                rrf_score = filename_weight * (1.0 / (rrf_k + rank + 1))
                rrf_scores[idx] = rrf_scores.get(idx, 0) + rrf_score
            if best_match_score >= 2:
                top_fn = self.metadata[filename_results[0][0]].get('source_file', '?') if filename_results[0][0] < len(self.metadata) else '?'
                print(f"[DEBUG] Filename channel: best={best_match_score} terms matched, weight={filename_weight}, top='{top_fn}'")

        # Apply source-based boosting and boilerplate penalty
        for idx in rrf_scores:
            if idx < len(self.metadata):
                meta = self.metadata[idx]
                source_file = meta.get('source_file', '')
                text = meta.get('text', '')
                
                # Penalize chunks that are mostly boilerplate footer text
                # (the "As a global leader...broad range of SKUs" paragraph
                #  appears in hundreds of Supermicro PDFs and pollutes keyword search)
                text_flat = re.sub(r'\s+', ' ', text.lower())
                if 'global leader in high performance' in text_flat and \
                   'broad range of skus' in text_flat:
                    rrf_scores[idx] *= 0.3  # Heavy penalty for boilerplate chunks
                    continue
                
                # Source-based boosting
                if source_file.endswith('.pdf'):
                    rrf_scores[idx] *= 1.05  # Subtle PDF preference as tiebreaker
                elif source_file.startswith('web_product_'):
                    rrf_scores[idx] *= 1.02  # Minimal boost for structured product data
                # web_page_* content: no boost (1.0x)
        
        # Sort by combined RRF score (with source boost applied)
        sorted_indices = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        # ----- Source-diversity path (broad queries) -----
        if max_per_source is not None:
            source_counts: Dict[str, int] = defaultdict(int)
            diversified = []
            for idx, score in sorted_indices:
                if idx < len(self.metadata):
                    src = self.metadata[idx].get('source_file', '')
                    norm_src = re.sub(r'__[0-9a-f]{8,}\.', '.', src)
                    if source_counts[norm_src] < max_per_source:
                        source_counts[norm_src] += 1
                        diversified.append((self.metadata[idx], score))
                        if len(diversified) >= top_k:
                            break
            return diversified

        # ----- Concentrated path (detail / follow-up queries) -----
        # Context expansion: if the top result comes from a multi-chunk document,
        # pull in sibling chunks so the LLM gets complete context (e.g., all 
        # product specs from a Global SKU page, not just the intro chunk).
        initial_results = []
        for idx, score in sorted_indices[:top_k]:
            if idx < len(self.metadata):
                initial_results.append((idx, self.metadata[idx], score))
        
        if initial_results:
            top_source = initial_results[0][1].get('source_file', '')
            top_total = initial_results[0][1].get('total_chunks', 1)
            top_score = initial_results[0][2]
            
            # Only expand if top source has many chunks (multi-page document)
            # and it scored significantly higher than #2
            should_expand = (
                top_total > 5 and 
                (len(initial_results) < 2 or top_score > initial_results[1][2] * 1.03)
            )
            
            if should_expand:
                # Find all chunks from this source and add missing ones
                existing_idxs = {r[0] for r in initial_results}
                sibling_chunks = []
                for i, m in enumerate(self.metadata):
                    if m.get('source_file') == top_source and i not in existing_idxs:
                        sibling_chunks.append((i, m, top_score * 0.95))
                
                # Sort siblings by chunk_index to maintain document order
                sibling_chunks.sort(key=lambda x: x[1].get('chunk_index', 0))
                
                # Insert siblings (up to half of top_k to leave room for other sources)
                max_siblings = top_k // 2
                siblings_to_add = sibling_chunks[:max_siblings]
                
                # Build final result: top result + siblings + remaining results
                final = [initial_results[0]]
                final.extend(siblings_to_add)
                remaining = [r for r in initial_results[1:] if r[0] not in {s[0] for s in siblings_to_add}]
                final.extend(remaining)
                initial_results = final[:top_k]
        
        return [(meta, score) for _, meta, score in initial_results]
    
    def search(self, query: str, top_k: int = 10) -> List[Tuple[Dict, float]]:
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
                is_keyword_heavy = index._is_keyword_heavy_query(args.query)
                print(f"\nQuery tokens: {query_tokens}")
                print(f"Detected as product code: {is_product_code}")
                print(f"Detected as keyword-heavy: {is_keyword_heavy}")
                if is_product_code:
                    print(f"Using weights: 20% semantic, 80% BM25")
                elif is_keyword_heavy:
                    print(f"Using weights: 25% semantic, 75% BM25")
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
