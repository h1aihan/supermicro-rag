#!/usr/bin/env python3
"""
Process web page content (JSONL files) into the same format as PDF extraction.
This allows the chunking pipeline to handle both PDFs and web pages uniformly.

Applies aggressive filtering:
- Expanded boilerplate removal (~10 known patterns)
- Content deduplication by URL path hash
- 200-char minimum content threshold
- Avoids field overlap (content vs full_text vs features)
"""

import json
import hashlib
import argparse
import re
from pathlib import Path
from urllib.parse import urlparse
from typing import Dict, List, Optional, Generator, Set
from tqdm import tqdm


def load_jsonl(filepath: Path) -> Generator[Dict, None, None]:
    """Load records from a JSONL file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


# --- Boilerplate patterns to strip ---
BOILERPLATE_STRINGS = [
    # JavaScript / browser warnings
    'JavaScript seems to be disabled in your browser.',
    'For the best experience on our site, be sure to turn on Javascript in your browser.',
    # Geographic restriction
    'It seems that you are geographically located outside of the United States. Please note that items in the Supermicro eStore can only be delivered within the United States.',
    'Certain products may not be available in your region.',
    # CTA / marketing headings
    'Why Wait?',
    'Get Our Top-Rated',
    'Welcome to the Supermicro eStore!',
    "Don't Miss Out!",
    'Need Help?',
    'in 24 Hours',
]

# Regex patterns for larger marketing blocks
BOILERPLATE_REGEXES = [
    # "The industry's broadest portfolio..." marketing paragraph (appears in 22K+ chunks)
    re.compile(
        r"The industry's broadest portfolio of .*?(?:servers|systems)[\s\S]*?(?:networking|performance)[^\n]*",
        re.IGNORECASE
    ),
    # Empty section markers like "Specifications:\n\n\nFeatures:"
    re.compile(r'Specifications:\s*\n\s*\n\s*Features:', re.IGNORECASE),
    # Repeated empty specification blocks
    re.compile(r'Specifications:\s*$', re.MULTILINE),
    # Repeated empty feature blocks  
    re.compile(r'Features:\s*$', re.MULTILINE),
]


def clean_web_text(text: str) -> str:
    """
    Aggressively clean boilerplate text from web content.
    
    Removes:
    - Known boilerplate strings (JS warnings, geo restrictions, CTAs)
    - Marketing block paragraphs via regex
    - Excessive whitespace
    """
    # Remove known boilerplate strings
    for bp in BOILERPLATE_STRINGS:
        text = text.replace(bp, '')
    
    # Remove marketing block patterns
    for pattern in BOILERPLATE_REGEXES:
        text = pattern.sub('', text)
    
    # Collapse multiple blank lines into single blank line
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Remove leading/trailing whitespace on each line
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(lines)
    
    return text.strip()


def normalize_url_path(url: str) -> str:
    """Extract and normalize URL path for deduplication."""
    parsed = urlparse(url)
    # Strip trailing slashes and query params
    path = parsed.path.rstrip('/')
    return path.lower()


def content_hash(text: str) -> str:
    """Generate a hash for content deduplication."""
    # Normalize whitespace before hashing
    normalized = re.sub(r'\s+', ' ', text.strip().lower())
    return hashlib.md5(normalized.encode()).hexdigest()


def product_to_document(product: Dict) -> Dict:
    """
    Convert a product record to the same format as PDF extraction output.
    
    Args:
        product: Product record from products.jsonl
        
    Returns:
        Document dict compatible with chunk.py
    """
    parts = []
    
    # Product name and model
    if product.get('name'):
        parts.append(f"Product: {product['name']}")
    if product.get('model'):
        parts.append(f"Model: {product['model']}")
    
    # Category - check for Gold Series
    if 'gold' in product.get('url', '').lower() or 'gold' in product.get('name', '').lower():
        parts.append("Category: Gold Series")
    elif product.get('category'):
        parts.append(f"Category: {product['category']}")
    
    # Key features/applications
    if product.get('key_features'):
        parts.append(f"Applications: {product['key_features']}")
    
    # Specifications
    specs = []
    for key in ['cpu', 'memory', 'storage', 'network', 'gpu', 'chassis']:
        if product.get(key):
            specs.append(f"  {key.upper()}: {product[key]}")
    if specs:
        parts.append("Specifications:")
        parts.extend(specs)
    
    # Price intentionally omitted — crawled prices go stale quickly.
    # Users are directed to the eStore for current pricing.
    
    # Source URL
    if product.get('url'):
        parts.append(f"Source URL: {product['url']}")
    
    text = '\n'.join(parts)
    
    # Create document in same format as extract.py output
    model_name = product.get('model', 'unknown').replace('/', '_')[:50]
    return {
        "filename": f"web_product_{model_name}.txt",
        "pages": [{"page_number": 1, "text": text}],
        "total_pages": 1,
        "error": None,
        "source_type": "web_product",
        "url": product.get('url', ''),
        "category": product.get('category', '')
    }


def rag_content_to_document(content: Dict) -> Optional[Dict]:
    """
    Convert RAG content record to the same format as PDF extraction output.
    
    Uses only the richest non-overlapping fields to avoid duplication:
    - Prefers 'content' over 'full_text' (full_text typically duplicates content + features)
    - Only appends 'features' and 'specifications' if they add new information
    
    Args:
        content: Content record from rag_content.jsonl
        
    Returns:
        Document dict compatible with chunk.py, or None if content is too short
    """
    parts = []
    
    # Title
    if content.get('title'):
        parts.append(f"Page: {content['title']}")
    
    # Category
    if content.get('category'):
        parts.append(f"Category: {content['category']}")
    
    # Check headings for Gold Series
    headings = content.get('headings', [])
    if any('gold' in h.lower() for h in headings):
        parts.append("Tags: Gold Series")
    
    # --- Use 'content' field as primary (avoids full_text duplication) ---
    primary_text = content.get('content', '') or ''
    primary_text = clean_web_text(primary_text)
    
    if primary_text:
        parts.append(primary_text)
    
    # Only add features if they contain info NOT already in primary text
    features = content.get('features', '') or ''
    if features:
        features_clean = clean_web_text(features)
        # Check if features text is substantially different from primary
        if features_clean and features_clean not in primary_text:
            # Only add if it brings at least 50 chars of new content
            overlap = sum(1 for line in features_clean.split('\n') if line.strip() in primary_text)
            total_lines = len([l for l in features_clean.split('\n') if l.strip()])
            if total_lines > 0 and overlap / total_lines < 0.7:
                parts.append(f"Features:\n{features_clean}")
    
    # Only add specifications if they contain actual data
    specs = content.get('specifications', '') or ''
    if specs:
        specs_clean = clean_web_text(specs)
        if specs_clean and len(specs_clean) > 20 and specs_clean not in primary_text:
            parts.append(f"Specifications:\n{specs_clean}")
    
    # Source URL
    if content.get('url'):
        parts.append(f"Source URL: {content['url']}")
    
    text = '\n'.join(parts)
    
    # FAQ entries are intentionally short (Q&A pairs); use a lower threshold
    category = content.get('category', '')
    min_chars = 80 if category.startswith("FAQ") else 200
    if len(text) < min_chars:
        return None
    
    # Create document in same format as extract.py output
    title = content.get('title', 'unknown').replace('/', '_').replace(' ', '_')[:50]
    return {
        "filename": f"web_page_{title}.txt",
        "pages": [{"page_number": 1, "text": text}],
        "total_pages": 1,
        "error": None,
        "source_type": "web_rag",
        "url": content.get('url', ''),
        "category": content.get('category', '')
    }


def accessory_to_document(acc: Dict) -> Optional[Dict]:
    """Convert an estore_accessories.jsonl record to chunk-compatible format."""
    parts = []
    name = acc.get("name", "")
    pn = acc.get("part_number", "")
    if name:
        parts.append(name)
    if pn:
        parts.append(f"Part Number: {pn}")
    if acc.get("category"):
        parts.append(f"Category: {acc['category']}")
    if acc.get("subcategory"):
        parts.append(f"Subcategory: {acc['subcategory']}")
    if acc.get("price"):
        parts.append(f"Price: ${acc['price']}")

    if acc.get("compatible_with"):
        parts.append(f"Compatible with: {acc['compatible_with']}")
    if acc.get("compatible_chassis"):
        parts.append(f"Compatible chassis: {acc['compatible_chassis']}")
    if acc.get("validated_systems"):
        parts.append(f"Validated systems: {acc['validated_systems']}")
    if acc.get("validated_chassis"):
        parts.append(f"Validated chassis: {acc['validated_chassis']}")
    if acc.get("designed_for"):
        parts.append(f"Designed for: {acc['designed_for']}")

    if acc.get("dimensions"):
        parts.append(f"Dimensions: {acc['dimensions']}")
    if acc.get("warranty"):
        parts.append(f"Warranty: {acc['warranty']}")

    specs = acc.get("specifications", {})
    if isinstance(specs, dict) and specs:
        parts.append("Specifications:")
        for k, v in specs.items():
            parts.append(f"  {k}: {v}")

    desc = acc.get("description", "")
    if desc and len(desc) > 20:
        parts.append(f"\n{desc}")

    features = acc.get("bullet_features", [])
    if isinstance(features, list) and features:
        clean = [f for f in features if len(f) > 5 and not f.startswith("My Account")]
        if clean:
            parts.append("Features:\n" + "\n".join(f"- {f}" for f in clean[:15]))

    if acc.get("url"):
        parts.append(f"Source URL: {acc['url']}")

    text = "\n".join(parts)
    if len(text) < 80:
        return None

    safe_pn = re.sub(r'[/\\]', '_', pn)[:60] if pn else "unknown"
    return {
        "filename": f"accessory_{safe_pn}.txt",
        "pages": [{"page_number": 1, "text": text}],
        "total_pages": 1,
        "error": None,
        "source_type": "accessory",
        "url": acc.get("url", ""),
    }


def process_pages(input_dir: str, output_dir: str):
    """
    Process web page JSONL files and output JSON files compatible with chunk.py.
    
    Applies:
    - Aggressive boilerplate removal
    - URL-path-based deduplication (skip pages with same URL path)
    - Content-hash deduplication (skip pages with near-identical text)
    - 200-char minimum content threshold
    
    Args:
        input_dir: Directory containing products.jsonl and rag_content.jsonl
        output_dir: Directory to save JSON files (same format as extract.py output)
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Clear old output to avoid stale files from previous runs
    if output_path.exists():
        for old_file in output_path.glob("*.json"):
            old_file.unlink()
    output_path.mkdir(parents=True, exist_ok=True)
    
    processed = 0
    skipped_short = 0
    skipped_dup_url = 0
    skipped_dup_content = 0
    
    # Deduplication sets
    seen_url_paths: Set[str] = set()
    seen_content_hashes: Set[str] = set()
    
    # --- Process products.jsonl (clean, structured - keep all) ---
    products_file = input_path / "products.jsonl"
    if products_file.exists():
        print(f"Processing products from {products_file}...")
        for product in tqdm(list(load_jsonl(products_file)), desc="Products"):
            doc = product_to_document(product)
            output_file = output_path / f"{doc['filename'].replace('.txt', '.json')}"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(doc, f, indent=2, ensure_ascii=False)
            processed += 1
    
    # --- Process rag_content.jsonl (needs aggressive filtering) ---
    rag_file = input_path / "rag_content.jsonl"
    if rag_file.exists():
        print(f"Processing RAG content from {rag_file}...")
        idx = 0
        for content in tqdm(load_jsonl(rag_file), desc="RAG Content"):
            # Dedup by URL path (skip for FAQ entries — they share a base URL)
            is_faq = content.get('category', '').startswith("FAQ")
            url = content.get('url', '')
            if url and not is_faq:
                url_path = normalize_url_path(url)
                if url_path in seen_url_paths:
                    skipped_dup_url += 1
                    continue
                seen_url_paths.add(url_path)
            
            # Convert to document (applies cleaning + 200-char threshold)
            doc = rag_content_to_document(content)
            if doc is None:
                skipped_short += 1
                continue
            
            # Dedup by content hash
            text_hash = content_hash(doc['pages'][0]['text'])
            if text_hash in seen_content_hashes:
                skipped_dup_content += 1
                continue
            seen_content_hashes.add(text_hash)
            
            # Write output
            base_name = doc['filename'].replace('.txt', '')
            output_file = output_path / f"{base_name}_{idx}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(doc, f, indent=2, ensure_ascii=False)
            processed += 1
            idx += 1
    
    # --- Process estore_accessories.jsonl ---
    acc_file = input_path / ".." / "accessories" / "estore_accessories.jsonl"
    if not acc_file.exists():
        acc_file = input_path.parent / "accessories" / "estore_accessories.jsonl"
    if acc_file.exists():
        print(f"\nProcessing accessories from {acc_file}...")
        acc_count = 0
        for acc in tqdm(list(load_jsonl(acc_file)), desc="Accessories"):
            doc = accessory_to_document(acc)
            if doc is None:
                skipped_short += 1
                continue
            text_hash = content_hash(doc['pages'][0]['text'])
            if text_hash in seen_content_hashes:
                skipped_dup_content += 1
                continue
            seen_content_hashes.add(text_hash)
            output_file = output_path / f"{doc['filename'].replace('.txt', '.json')}"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(doc, f, indent=2, ensure_ascii=False)
            processed += 1
            acc_count += 1
        print(f"  Accessories processed: {acc_count}")

    print(f"\nProcessing complete!")
    print(f"  Processed: {processed} documents")
    print(f"  Skipped (too short after cleaning): {skipped_short}")
    print(f"  Skipped (duplicate URL path): {skipped_dup_url}")
    print(f"  Skipped (duplicate content): {skipped_dup_content}")
    print(f"  Total skipped: {skipped_short + skipped_dup_url + skipped_dup_content}")
    print(f"  Output directory: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process web page content into format compatible with chunk.py"
    )
    parser.add_argument(
        "--input",
        default="data/pages/",
        help="Input directory containing products.jsonl and rag_content.jsonl (default: data/pages/)"
    )
    parser.add_argument(
        "--output",
        default="data/raw_pages/",
        help="Output directory for processed JSON files (default: data/raw_pages/)"
    )
    
    args = parser.parse_args()
    process_pages(args.input, args.output)
