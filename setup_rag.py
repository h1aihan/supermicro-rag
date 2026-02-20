#!/usr/bin/env python3
"""
Setup script to run the complete RAG pipeline.
Supports both PDF documents and web page content.
"""

import subprocess
import sys
import argparse
from pathlib import Path


def run_command(cmd, description, allow_fail=False):
    """Run a command and handle errors."""
    print(f"\n{'='*80}")
    print(f"{description}")
    print(f"{'='*80}")
    print(f"Running: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, check=False)
    
    if result.returncode != 0:
        if allow_fail:
            print(f"Warning: {description} failed (continuing anyway)")
            return False
        print(f"\nError: {description} failed with exit code {result.returncode}")
        sys.exit(1)
    
    print(f"✓ {description} completed successfully")
    return True


def main():
    """Run the complete RAG setup pipeline."""
    parser = argparse.ArgumentParser(description="Setup Supermicro RAG pipeline")
    parser.add_argument(
        "--filter",
        choices=['datasheet', 'all'],
        default='datasheet',
        help="Filter PDFs: 'datasheet' (default, recommended) or 'all'"
    )
    parser.add_argument(
        "--source",
        choices=['pdf', 'pages', 'both'],
        default='both',
        help="Data source: 'pdf' (PDFs only), 'pages' (web pages only), 'both' (default)"
    )
    args = parser.parse_args()
    
    print("=" * 80)
    print("Supermicro RAG Setup")
    print("=" * 80)
    print(f"\nData source: {args.source}")
    print(f"PDF filter: {args.filter}")
    
    print("\nThis script will:")
    if args.source in ['pdf', 'both']:
        print("1. Extract text from PDFs")
    if args.source in ['pages', 'both']:
        print("2. Process web page content")
    print("3. Chunk all extracted text")
    print("4. Generate embeddings and create FAISS index")
    print("\nThis may take a while...")
    
    import multiprocessing
    num_workers = multiprocessing.cpu_count()
    
    input_dirs = []
    
    # Step 1: Process PDFs if requested
    if args.source in ['pdf', 'both']:
        pdfs_dir = Path("pdfs")
        if pdfs_dir.exists() and list(pdfs_dir.glob("*.pdf")):
            filter_desc = "datasheets only" if args.filter == 'datasheet' else "all PDFs"
            run_command(
                [sys.executable, "src/extract.py", 
                 "--input", "pdfs/", 
                 "--output", "data/raw_text/", 
                 "--workers", str(num_workers),
                 "--filter", args.filter],
                f"Step 1a: Extracting text from PDFs ({filter_desc}, {num_workers} workers)"
            )
            input_dirs.append("data/raw_text/")
        else:
            print("\nNote: No PDFs found in 'pdfs/' directory, skipping PDF extraction")
    
    # Step 2: Process web pages if requested
    if args.source in ['pages', 'both']:
        pages_dir = Path("data/pages")
        if pages_dir.exists() and (
            (pages_dir / "products.jsonl").exists() or 
            (pages_dir / "rag_content.jsonl").exists()
        ):
            run_command(
                [sys.executable, "src/process_pages.py", 
                 "--input", "data/pages/", 
                 "--output", "data/raw_pages/"],
                "Step 1b: Processing web page content"
            )
            input_dirs.append("data/raw_pages/")
        else:
            print("\nNote: No web page data found in 'data/pages/', skipping")
    
    if not input_dirs:
        print("\nError: No data sources available!")
        print("Please either:")
        print("  - Copy PDFs to the pdfs/ directory")
        print("  - Copy web content JSONL files to data/pages/")
        sys.exit(1)
    
    # Step 3: Chunk the text from all sources
    chunk_cmd = [sys.executable, "src/chunk.py", "--input"] + input_dirs + ["--output", "data/chunks.jsonl"]
    run_command(chunk_cmd, "Step 2: Chunking text from all sources")
    
    # Step 4: Generate embeddings and create index
    run_command(
        [sys.executable, "src/embed.py", "--input", "data/chunks.jsonl", "--output", "embeddings/faiss_index/"],
        "Step 3: Generating embeddings and creating FAISS index"
    )
    
    print("\n" + "=" * 80)
    print("Setup Complete!")
    print("=" * 80)
    print(f"\nData sources processed: {', '.join(input_dirs)}")
    print("\nYou can now use the chatbot:")
    print("  python src/chatbot.py --interactive")
    print("  python src/chatbot.py --query 'Your question here'")
    print("\nOr start the web server:")
    print("  python src/server.py")


if __name__ == "__main__":
    main()
