#!/usr/bin/env python3
"""
Extract text from PDF files in the pdfs/ directory.
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import pypdf
from tqdm import tqdm


def extract_text_from_pdf(pdf_path: Path) -> Dict:
    """
    Extract text from a single PDF file.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Dictionary with extracted text and metadata
    """
    result = {
        "filename": pdf_path.name,
        "pages": [],
        "total_pages": 0,
        "error": None
    }
    
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = pypdf.PdfReader(file)
            result["total_pages"] = len(pdf_reader.pages)
            
            for page_num, page in enumerate(pdf_reader.pages, start=1):
                try:
                    text = page.extract_text()
                    result["pages"].append({
                        "page_number": page_num,
                        "text": text
                    })
                except Exception as e:
                    result["pages"].append({
                        "page_number": page_num,
                        "text": "",
                        "error": str(e)
                    })
                    
    except pypdf.errors.PdfReadError as e:
        result["error"] = f"PDF read error: {str(e)}"
    except Exception as e:
        result["error"] = f"Unexpected error: {str(e)}"
    
    return result


def process_single_pdf(args: Tuple[Path, Path, bool]) -> Tuple[str, bool, bool, str]:
    """
    Process a single PDF file (for parallel processing).
    
    Args:
        args: Tuple of (pdf_path, output_path, resume)
        
    Returns:
        Tuple of (filename, processed, skipped, error_message)
    """
    pdf_file, output_path, resume = args
    output_file = output_path / f"{pdf_file.stem}.json"
    
    # Skip if already processed
    if resume and output_file.exists():
        return (pdf_file.name, False, True, None)
    
    # Extract text
    result = extract_text_from_pdf(pdf_file)
    
    # Save result
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        if result["error"]:
            return (pdf_file.name, False, False, result["error"])
        else:
            return (pdf_file.name, True, False, None)
    except Exception as e:
        return (pdf_file.name, False, False, f"Error saving: {e}")


def extract_all_pdfs(input_dir: str, output_dir: str, resume: bool = True, num_workers: int = None):
    """
    Extract text from all PDFs in the input directory using parallel processing.
    
    Args:
        input_dir: Directory containing PDF files
        output_dir: Directory to save extracted text JSON files
        resume: If True, skip already processed PDFs
        num_workers: Number of parallel workers (default: CPU count)
    """
    import multiprocessing
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all PDF files
    pdf_files = list(input_path.glob("*.pdf"))
    
    if not pdf_files:
        print(f"No PDF files found in '{input_dir}'")
        return
    
    # Determine number of workers
    if num_workers is None:
        num_workers = multiprocessing.cpu_count()
    
    print(f"Found {len(pdf_files)} PDF files")
    print(f"Using {num_workers} parallel workers")
    print(f"Extracting text to '{output_dir}'...")
    
    # Prepare arguments for parallel processing
    tasks = [(pdf_file, output_path, resume) for pdf_file in pdf_files]
    
    processed = 0
    skipped = 0
    errors = 0
    error_messages = []
    
    # Process PDFs in parallel
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        future_to_pdf = {
            executor.submit(process_single_pdf, task): task[0] 
            for task in tasks
        }
        
        # Process completed tasks with progress bar
        with tqdm(total=len(pdf_files), desc="Extracting PDFs") as pbar:
            for future in as_completed(future_to_pdf):
                filename, was_processed, was_skipped, error_msg = future.result()
                
                if was_processed:
                    processed += 1
                elif was_skipped:
                    skipped += 1
                else:
                    errors += 1
                    if error_msg:
                        error_messages.append(f"{filename}: {error_msg}")
                
                pbar.update(1)
    
    print(f"\nExtraction complete!")
    print(f"  Processed: {processed} files")
    print(f"  Skipped (already exist): {skipped} files")
    print(f"  Errors: {errors} files")
    
    # Print error messages if any
    if error_messages:
        print(f"\nError details (showing first 10):")
        for msg in error_messages[:10]:
            print(f"  - {msg}")
        if len(error_messages) > 10:
            print(f"  ... and {len(error_messages) - 10} more errors")


if __name__ == "__main__":
    # Required for multiprocessing on Windows/WSL
    import multiprocessing
    multiprocessing.freeze_support()
    
    parser = argparse.ArgumentParser(
        description="Extract text from PDF files"
    )
    parser.add_argument(
        "--input",
        default="pdfs/",
        help="Input directory containing PDF files (default: pdfs/)"
    )
    parser.add_argument(
        "--output",
        default="data/raw_text/",
        help="Output directory for extracted text (default: data/raw_text/)"
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Re-process all PDFs even if already extracted"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: CPU count)"
    )
    
    args = parser.parse_args()
    extract_all_pdfs(args.input, args.output, resume=not args.no_resume, num_workers=args.workers)
