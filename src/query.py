#!/usr/bin/env python3
"""
Query processing and retrieval system.
"""

from typing import List, Dict, Tuple

# Support running as:
# - python -m src.query   (package mode)
# - python src/query.py   (script mode)
try:
    from src.index import VectorIndex
except ImportError:
    from index import VectorIndex


class RAGQueryProcessor:
    """Process queries and retrieve relevant context."""
    
    def __init__(self, index_dir: str, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the query processor.
        
        Args:
            index_dir: Directory containing FAISS index
            model_name: Name of the sentence transformer model
        """
        self.index = VectorIndex(index_dir, model_name)
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Retrieve relevant chunks for a query.
        
        Args:
            query: User query
            top_k: Number of chunks to retrieve
            
        Returns:
            List of chunk dictionaries with similarity scores
        """
        results = self.index.search(query, top_k)
        
        # Format results
        retrieved_chunks = []
        for chunk, score in results:
            retrieved_chunks.append({
                "text": chunk["text"],
                "source_file": chunk["source_file"],
                "chunk_id": chunk["chunk_id"],
                "similarity_score": score
            })
        
        return retrieved_chunks
    
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
            context_parts.append(
                f"[Source {i}: {chunk['source_file']}]\n{chunk['text']}\n"
            )
        
        return "\n".join(context_parts)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test query processing")
    parser.add_argument(
        "--index-dir",
        default="embeddings/faiss_index/",
        help="Directory containing FAISS index (default: embeddings/faiss_index/)"
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
    
    args = parser.parse_args()
    
    try:
        processor = RAGQueryProcessor(args.index_dir)
        chunks = processor.retrieve(args.query, args.top_k)
        
        print(f"\nRetrieved {len(chunks)} chunks for query: '{args.query}'\n")
        for i, chunk in enumerate(chunks, 1):
            print(f"{'='*80}")
            print(f"Chunk {i} (Score: {chunk['similarity_score']:.4f})")
            print(f"Source: {chunk['source_file']}")
            print(f"{'='*80}")
            print(chunk['text'][:500])
            print("\n")
    
    except Exception as e:
        print(f"Error: {e}")
