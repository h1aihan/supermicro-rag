#!/usr/bin/env python3
"""
Extract text from PDF files in the pdfs/ directory.
Supports filtering to only process specific document types (e.g., datasheets).
"""

import os
import re
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import pypdf
from tqdm import tqdm


# =============================================================================
# Document Type Filtering
# =============================================================================

# Patterns for identifying product datasheets (case-insensitive)
# These are the files that contain product specifications
DATASHEET_PATTERNS = [
    # Explicitly named datasheets
    r'^datasheet',
    r'_datasheet',
    r'-datasheet',
    
    # Product model datasheets (named by model number)
    r'^sys-',          # Server systems (e.g., sys-521ge-tnrt.pdf)
    r'^ssg-',          # Storage systems (e.g., ssg-5049p-e1cr45h.pdf)
    r'^as-',           # AMD systems (e.g., as-4125gs-tnrt.pdf)
    r'^ars-',          # ARM systems (e.g., ars-111l-fr.pdf)
    r'^sbi-',          # Blade modules (e.g., sbi-7428r-t3.pdf)
    r'^aoc-',          # Add-on cards (e.g., aoc-s25g6-m2s.pdf)
    
    # Motherboard datasheets
    r'^h1[0-9]',       # H11, H12, H13 series motherboards
    r'^x1[0-9]',       # X10, X11, X12, X13 series motherboards
    r'^a[0-9]+',       # A+ series motherboards
    
    # Other product docs
    r'^spec[_-]',
    r'_spec[_-]',
]

# Patterns to EXCLUDE (even if they match above)
EXCLUDE_PATTERNS = [
    r'_manual',
    r'_guide',
    r'_qrg',           # Quick reference guides
    r'_compliance',
]


def is_datasheet(filename: str) -> bool:
    """
    Check if a filename matches datasheet patterns (and doesn't match exclusions).
    
    Args:
        filename: The PDF filename to check
        
    Returns:
        True if the file appears to be a product datasheet
    """
    # First check exclusions
    for pattern in EXCLUDE_PATTERNS:
        if re.search(pattern, filename, re.IGNORECASE):
            return False
    
    # Then check if it matches datasheet patterns
    for pattern in DATASHEET_PATTERNS:
        if re.search(pattern, filename, re.IGNORECASE):
            return True
    return False


def filter_pdfs(pdf_files: List[Path], filter_type: Optional[str] = None) -> List[Path]:
    """
    Filter PDF files based on document type.
    
    Args:
        pdf_files: List of PDF file paths
        filter_type: Type of filter to apply ('datasheet', 'all', or None)
        
    Returns:
        Filtered list of PDF files
    """
    if filter_type is None or filter_type == 'all':
        return pdf_files
    
    if filter_type == 'datasheet':
        filtered = [f for f in pdf_files if is_datasheet(f.name)]
        print(f"Filtered to {len(filtered)} datasheets (from {len(pdf_files)} total PDFs)")
        return filtered
    
    # Unknown filter type - return all
    print(f"Warning: Unknown filter type '{filter_type}', processing all PDFs")
    return pdf_files


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


def extract_all_pdfs(
    input_dir: str, 
    output_dir: str, 
    resume: bool = True, 
    num_workers: int = None,
    filter_type: Optional[str] = None
):
    """
    Extract text from all PDFs in the input directory using parallel processing.
    
    Args:
        input_dir: Directory containing PDF files
        output_dir: Directory to save extracted text JSON files
        resume: If True, skip already processed PDFs
        num_workers: Number of parallel workers (default: CPU count)
        filter_type: Filter PDFs by type ('datasheet', 'all', or None for all)
    """
    import multiprocessing
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all PDF files
    all_pdf_files = list(input_path.glob("*.pdf"))
    
    if not all_pdf_files:
        print(f"No PDF files found in '{input_dir}'")
        return
    
    # Apply filter
    pdf_files = filter_pdfs(all_pdf_files, filter_type)
    
    if not pdf_files:
        print(f"No PDF files match the filter '{filter_type}'")
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
    parser.add_argument(
        "--filter",
        choices=['datasheet', 'all'],
        default='all',
        help="Filter PDFs by type: 'datasheet' for datasheets only, 'all' for everything (default: all)"
    )
    
    args = parser.parse_args()
    extract_all_pdfs(
        args.input, 
        args.output, 
        resume=not args.no_resume, 
        num_workers=args.workers,
        filter_type=args.filter
    )
