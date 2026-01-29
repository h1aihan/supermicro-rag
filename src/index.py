#!/usr/bin/env python3
"""
Load and manage the FAISS vector index.
This is a convenience module that wraps the FAISS index loading.
"""

import json
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


class VectorIndex:
    """Wrapper class for FAISS index and metadata."""
    
    def __init__(self, index_dir: str, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the vector index.
        
        Args:
            index_dir: Directory containing FAISS index and metadata
            model_name: Name of the sentence transformer model
        """
        self.index_dir = Path(index_dir)
        self.model_name = model_name
        self.model = None
        self.index = None
        self.metadata = []
        
        self._load_index()
        self._load_metadata()
        self._load_model()
    
    def _load_index(self):
        """Load FAISS index from disk."""
        index_file = self.index_dir / "faiss.index"
        if not index_file.exists():
            raise FileNotFoundError(f"FAISS index not found at {index_file}")
        
        self.index = faiss.read_index(str(index_file))
        print(f"Loaded FAISS index with {self.index.ntotal} vectors")
    
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
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[Dict, float]]:
        """
        Search for similar chunks.
        
        Args:
            query: Query text
            top_k: Number of results to return
            
        Returns:
            List of tuples (chunk_dict, similarity_score)
        """
        # Generate query embedding
        query_embedding = self.model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        # Get results with metadata
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.metadata):
                chunk = self.metadata[idx]
                results.append((chunk, float(score)))
        
        return results


if __name__ == "__main__":
    # Test loading the index
    import argparse
    
    parser = argparse.ArgumentParser(description="Test loading the FAISS index")
    parser.add_argument(
        "--index-dir",
        default="embeddings/faiss_index/",
        help="Directory containing FAISS index (default: embeddings/faiss_index/)"
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
    
    args = parser.parse_args()
    
    try:
        index = VectorIndex(args.index_dir)
        
        if args.query:
            print(f"\nSearching for: '{args.query}'")
            results = index.search(args.query, args.top_k)
            
            print(f"\nTop {len(results)} results:")
            for i, (chunk, score) in enumerate(results, 1):
                print(f"\n{i}. Score: {score:.4f}")
                print(f"   Source: {chunk['source_file']}")
                print(f"   Text preview: {chunk['text'][:200]}...")
        else:
            print("\nIndex loaded successfully!")
            print("Use --query to test a search query.")
    
    except Exception as e:
        print(f"Error: {e}")
