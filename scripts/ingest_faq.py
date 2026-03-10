#!/usr/bin/env python3
"""
Convert estore_faq.jsonl into rag_content.jsonl format and append.

Each FAQ Q&A pair becomes a single RAG document. The question is used as the
title and headings (natural search keywords), and the answer becomes the
content. Category is prefixed with "FAQ - " so the query planner can
distinguish FAQ results from product documentation.

Source: output_faq/estore_faq.jsonl (from scrape_estore_faq.py)
Target: data/pages/rag_content.jsonl (appended, deduped by title)
"""

import json
import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional


def faq_to_rag_content(record: Dict) -> Optional[Dict]:
    """Convert an estore_faq.jsonl record to rag_content.jsonl format."""
    question = record.get("question", "").strip()
    answer = record.get("answer_text", "").strip()
    category = record.get("category_name", "General").strip()

    if not question or not answer:
        return None

    content_text = f"Q: {question}\n\nA: {answer}"

    full_text = (
        f"Supermicro eStore FAQ — {category}\n\n"
        f"Question: {question}\n\n"
        f"Answer: {answer}"
    )

    faq_id = record.get("faq_id", "0")
    base_url = record.get("source_url", "https://store.supermicro.com/us_en/faq/")
    unique_url = f"{base_url}#{faq_id}"

    return {
        "url": unique_url,
        "title": f"FAQ: {question}",
        "category": f"FAQ - {category}",
        "headings": [question],
        "content": content_text,
        "specifications": "",
        "features": "",
        "meta_description": question,
        "full_text": full_text,
        "scraped_at": datetime.now(timezone.utc).isoformat(),
    }


def main():
    parser = argparse.ArgumentParser(description="Ingest eStore FAQ into rag_content.jsonl")
    parser.add_argument(
        "--input", default="data/faq/estore_faq.jsonl",
        help="Input estore_faq.jsonl path",
    )
    parser.add_argument(
        "--output", default="data/pages/rag_content.jsonl",
        help="Output rag_content.jsonl to append to",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print stats without writing",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Remove old FAQ entries from rag_content.jsonl before re-ingesting",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"Error: {input_path} not found")
        print(f"Copy estore_faq.jsonl from the crawler output:")
        print(f"  cp <supermicro>/output_faq/estore_faq.jsonl {input_path}")
        return

    if args.force and output_path.exists():
        kept_lines = []
        removed = 0
        with open(output_path) as f:
            for line in f:
                if line.strip():
                    try:
                        rec = json.loads(line)
                        if rec.get("title", "").startswith("FAQ:"):
                            removed += 1
                            continue
                    except json.JSONDecodeError:
                        pass
                kept_lines.append(line)
        with open(output_path, "w") as f:
            f.writelines(kept_lines)
        print(f"Removed {removed} old FAQ entries from {output_path}")

    existing_titles = set()
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                if line.strip():
                    try:
                        rec = json.loads(line)
                        if rec.get("title"):
                            existing_titles.add(rec["title"])
                    except json.JSONDecodeError:
                        pass
    print(f"Existing rag_content entries: {len(existing_titles)}")

    total = 0
    converted = 0
    skipped_exists = 0
    skipped_empty = 0
    records_to_write = []

    with open(input_path) as f:
        for line in f:
            if not line.strip():
                continue
            total += 1
            record = json.loads(line)

            rag_record = faq_to_rag_content(record)
            if rag_record is None:
                skipped_empty += 1
                continue

            if rag_record["title"] in existing_titles:
                skipped_exists += 1
                continue

            records_to_write.append(rag_record)
            converted += 1

    print(f"\nResults:")
    print(f"  Total input records: {total}")
    print(f"  Converted: {converted}")
    print(f"  Skipped (already exists): {skipped_exists}")
    print(f"  Skipped (empty Q or A): {skipped_empty}")

    if args.dry_run:
        print(f"\n[DRY RUN] Would append {converted} records to {output_path}")
        if records_to_write:
            print(f"\nSample output (first record):")
            print(json.dumps(records_to_write[0], indent=2, ensure_ascii=False)[:2000])
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "a") as f:
        for rec in records_to_write:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\nAppended {converted} FAQ records to {output_path}")
    print(f"\nNext steps:")
    print(f"  1. Re-run the RAG pipeline to rebuild the index:")
    print(f"     python setup_rag.py --source pages")
    print(f"  2. Or rebuild everything:")
    print(f"     python setup_rag.py --source both")


if __name__ == "__main__":
    main()
