#!/usr/bin/env python3
"""
Setup script to run the complete RAG pipeline.
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*80}")
    print(f"{description}")
    print(f"{'='*80}")
    print(f"Running: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, check=False)
    
    if result.returncode != 0:
        print(f"\nError: {description} failed with exit code {result.returncode}")
        sys.exit(1)
    
    print(f"✓ {description} completed successfully")


def main():
    """Run the complete RAG setup pipeline."""
    print("=" * 80)
    print("Supermicro RAG Setup")
    print("=" * 80)
    print("\nThis script will:")
    print("1. Extract text from PDFs")
    print("2. Chunk the extracted text")
    print("3. Generate embeddings and create FAISS index")
    print("\nThis may take a while depending on the number of PDFs...")
    
    # Check if PDFs directory exists
    pdfs_dir = Path("pdfs")
    if not pdfs_dir.exists() or not list(pdfs_dir.glob("*.pdf")):
        print("\nError: No PDFs found in 'pdfs/' directory")
        print("Please copy PDFs to the pdfs/ directory first.")
        sys.exit(1)
    
    # Step 1: Extract text from PDFs (with parallel processing)
    import multiprocessing
    num_workers = multiprocessing.cpu_count()
    run_command(
        [sys.executable, "src/extract.py", "--input", "pdfs/", "--output", "data/raw_text/", "--workers", str(num_workers)],
        f"Step 1: Extracting text from PDFs (using {num_workers} parallel workers)"
    )
    
    # Step 2: Chunk the text
    run_command(
        [sys.executable, "src/chunk.py", "--input", "data/raw_text/", "--output", "data/chunks.jsonl"],
        "Step 2: Chunking text"
    )
    
    # Step 3: Generate embeddings and create index
    run_command(
        [sys.executable, "src/embed.py", "--input", "data/chunks.jsonl", "--output", "embeddings/faiss_index/"],
        "Step 3: Generating embeddings and creating FAISS index"
    )
    
    print("\n" + "=" * 80)
    print("Setup Complete!")
    print("=" * 80)
    print("\nYou can now use the chatbot:")
    print("  python src/chatbot.py --interactive")
    print("  python src/chatbot.py --query 'Your question here'")
    print("\nMake sure to set up your .env file with API keys if using OpenAI.")


if __name__ == "__main__":
    main()
