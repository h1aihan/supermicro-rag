#!/usr/bin/env python3
"""
Chunk extracted text from PDFs into smaller pieces for embedding.
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict

from tqdm import tqdm

# Support both older/langchain-integrated and newer splitters package
try:  # langchain <= 0.2 style
    from langchain.text_splitter import RecursiveCharacterTextSplitter  # type: ignore
except ImportError:  # langchain >= 0.3 splitters
    from langchain_text_splitters import RecursiveCharacterTextSplitter  # type: ignore


def load_extracted_text(input_dirs: List[str]) -> List[Dict]:
    """
    Load all extracted text JSON files from one or more directories.
    
    Args:
        input_dirs: List of directories containing extracted text JSON files
        
    Returns:
        List of extracted text dictionaries
    """
    documents = []
    
    for input_dir in input_dirs:
        input_path = Path(input_dir)
        if not input_path.exists():
            print(f"Warning: Directory '{input_dir}' does not exist, skipping")
            continue
            
        json_files = list(input_path.glob("*.json"))
        print(f"  Found {len(json_files)} JSON files in '{input_dir}'")
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if not data.get("error"):
                        documents.append(data)
            except Exception as e:
                print(f"Error loading {json_file.name}: {e}")
    
    return documents


def chunk_document(doc: Dict, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Dict]:
    """
    Chunk a single document into smaller pieces.
    
    Args:
        doc: Document dictionary with pages
        chunk_size: Size of each chunk in characters
        chunk_overlap: Overlap between chunks in characters
        
    Returns:
        List of chunk dictionaries with metadata
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = []
    
    # Combine all pages into full text
    full_text = ""
    for page in doc["pages"]:
        page_text = page.get("text", "")
        if page_text:
            full_text += f"\n\n--- Page {page['page_number']} ---\n\n{page_text}"
    
    if not full_text.strip():
        return chunks
    
    # Split into chunks
    text_chunks = text_splitter.split_text(full_text)
    
    # Create chunk metadata
    for idx, chunk_text in enumerate(text_chunks):
        if chunk_text.strip():  # Skip empty chunks
            chunk = {
                "chunk_id": f"{doc['filename']}_chunk_{idx}",
                "source_file": doc["filename"],
                "chunk_index": idx,
                "text": chunk_text,
                "total_chunks": len(text_chunks)
            }
            chunks.append(chunk)
    
    return chunks


def chunk_all_documents(input_dirs: List[str], output_file: str, chunk_size: int = 1000, chunk_overlap: int = 200):
    """
    Chunk all documents and save to JSONL file.
    
    Args:
        input_dirs: List of directories containing extracted text JSON files
        output_file: Output JSONL file path
        chunk_size: Size of each chunk in characters
        chunk_overlap: Overlap between chunks in characters
    """
    print(f"Loading extracted text from {len(input_dirs)} directories...")
    documents = load_extracted_text(input_dirs)
    
    if not documents:
        print("No documents found to chunk.")
        return
    
    print(f"Found {len(documents)} documents")
    print(f"Chunking with size={chunk_size}, overlap={chunk_overlap}...")
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    total_chunks = 0
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for doc in tqdm(documents, desc="Chunking documents"):
            chunks = chunk_document(doc, chunk_size, chunk_overlap)
            
            for chunk in chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + '\n')
                total_chunks += 1
    
    print(f"\nChunking complete!")
    print(f"  Total chunks created: {total_chunks}")
    print(f"  Saved to: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Chunk extracted text into smaller pieces"
    )
    parser.add_argument(
        "--input",
        nargs='+',
        default=["data/raw_text/"],
        help="Input directories with extracted text JSON files (default: data/raw_text/)"
    )
    parser.add_argument(
        "--output",
        default="data/chunks.jsonl",
        help="Output JSONL file for chunks (default: data/chunks.jsonl)"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Chunk size in characters (default: 1000)"
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
        help="Chunk overlap in characters (default: 200)"
    )
    
    args = parser.parse_args()
    chunk_all_documents(args.input, args.output, args.chunk_size, args.chunk_overlap)
