#!/usr/bin/env python3
"""
Setup script to run the complete RAG pipeline.
Supports both PDF documents and web page content.
"""

import json
import re
import subprocess
import sys
import argparse
from pathlib import Path

_RE_MANUAL = re.compile(r'^MNL-', re.IGNORECASE)
_RE_GUIDE = re.compile(r'[Uu]ser[_\s]?[Gg]uide|^QRG-|^BMC_IPMI|^IPMI', re.IGNORECASE)
_RE_CHASSIS_MANUAL = re.compile(r'^SC\d{3}', re.IGNORECASE)


def split_chunks(input_file: str, primary_output: str, manual_output: str):
    """Split chunks.jsonl into primary (datasheets, web, accessories) and manual indices."""
    primary_count = manual_count = 0
    with open(input_file, 'r', encoding='utf-8') as fin, \
         open(primary_output, 'w', encoding='utf-8') as fp, \
         open(manual_output, 'w', encoding='utf-8') as fm:
        for line in fin:
            if not line.strip():
                continue
            obj = json.loads(line)
            source = obj.get('source_file', '')
            if _RE_MANUAL.search(source) or _RE_GUIDE.search(source) or _RE_CHASSIS_MANUAL.search(source):
                fm.write(line)
                manual_count += 1
            else:
                fp.write(line)
                primary_count += 1
    print(f"  Split complete: {primary_count:,} primary, {manual_count:,} manual chunks")
    return primary_count, manual_count


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
        choices=['datasheet', 'product', 'all'],
        default='product',
        help="Filter PDFs: 'datasheet' (specs only), 'product' (default, datasheets + chassis + manuals), 'all' (everything)"
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

    # Step 4: Split chunks into primary (datasheets, web, accessories) and manual indices
    print(f"\n{'='*80}")
    print("Step 3: Splitting chunks into primary and manual indices")
    print(f"{'='*80}")
    split_chunks("data/chunks.jsonl", "data/chunks_primary.jsonl", "data/chunks_manuals.jsonl")
    print("✓ Chunk splitting completed successfully")

    # Step 5a: Embed primary index
    run_command(
        [sys.executable, "src/embed.py", "--input", "data/chunks_primary.jsonl", "--output", "embeddings/primary_index/"],
        "Step 4a: Building primary index (datasheets, web pages, accessories, FAQ)"
    )

    # Step 5b: Embed manual index
    run_command(
        [sys.executable, "src/embed.py", "--input", "data/chunks_manuals.jsonl", "--output", "embeddings/manual_index/"],
        "Step 4b: Building manual index (user guides, installation manuals)"
    )

    # Step 6: Build entity-relationship graph from primary index
    run_command(
        [sys.executable, "src/entity_graph.py",
         "--metadata", "embeddings/primary_index/metadata.jsonl",
         "--output", "embeddings/primary_index/entity_graph.json"],
        "Step 5: Building entity-relationship graph",
        allow_fail=True,
    )

    print("\n" + "=" * 80)
    print("Setup Complete!")
    print("=" * 80)
    print(f"\nData sources processed: {', '.join(input_dirs)}")
    print("Indices: embeddings/primary_index/, embeddings/manual_index/")
    print("\nYou can now use the chatbot:")
    print("  python src/chatbot.py --interactive")
    print("  python src/chatbot.py --query 'Your question here'")
    print("\nOr start the web server:")
    print("  python src/server.py")


if __name__ == "__main__":
    main()
