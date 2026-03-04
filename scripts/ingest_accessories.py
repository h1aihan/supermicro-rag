#!/usr/bin/env python3
"""
Convert estore_accessories.jsonl into rag_content.jsonl format and append.

Cleans the noisy bullet_features (full nav menu boilerplate) and builds
structured text from the meaningful fields: part_number, name, description,
compatible_with, compatible_chassis, validated_systems, validated_chassis,
specifications, designed_for, mechanism, dimensions.
"""

import json
import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional


NAV_BOILERPLATE_MARKERS = [
    "My AccountSign In",
    "SystemsGold Series",
    "SolutionsCloud Computing",
    "CablesNetworking",
    "Compare Products",
    "About Us",
    "Contact Us",
    "Condition of Use",
    "Privacy Policy",
    "Return Policy",
    "Warranty Policy",
    "Tax Exemption",
    "My Configurations",
]


def is_nav_boilerplate(item: str) -> bool:
    """Check if a bullet_features item is navigation boilerplate."""
    if len(item) < 3:
        return True
    for marker in NAV_BOILERPLATE_MARKERS:
        if marker in item:
            return True
    # Single menu items that are just category names
    single_nav = {
        "New", "Storage", "Edge", "CPU", "Drive Trays", "I/O Shields",
        "Power Supplies", "Rail Kits", "Computer Accessories", "Deals",
        "Sign In", "Register", "My Account", "Single Processor",
        "Dual Processor", "2U", "4U", "5U", "8U", "10U",
        "2 Nodes", "4 Nodes", "8 Nodes", "10 Nodes", "Tower", "Rack",
        "WIO", "Mainstream", "Hyper", "Ultra", "CloudDC",
        "Cloud Computing", "Virtualization", "Artificial Intelligence",
        "Data Center", "Security", "Networking", "High Performance Computing",
        "Data Storage", "Edge Computing",
        "MiniSAS Breakout Cables", "MiniSAS Cables", "MiniSAS HD Breakout Cables",
        "MiniSAS HD Cables", "MiniSAS to MiniSAS HD Cables", "OCuLink Cables",
        "SATA Cables", "Slimline SAS", "MCIO Cables",
        "10G Cables", "25G Cables", "40G Cables", "100G Cables",
        "Power Cables", "Power Cords",
        "1U Heatsink", "2U Heatsink", "3U Heatsink", "4U Heatsink",
        "Drive Accessories", "DVD Accessories", "Fan Accessories",
        "Cable Arm", "GPU Kit", "Front Bezel", "Power Adapter",
        "Client Access Licenses", "Gold Series", "GPU Systems",
        "SuperWorkstations", "X14 Systems", "A+ Systems",
        "System", "Chassis",
    }
    if item.strip() in single_nav:
        return True
    return False


def clean_bullet_features(features: list) -> list:
    """Filter bullet_features to only keep actual product features."""
    return [f for f in features if not is_nav_boilerplate(f)]


def accessory_to_rag_content(record: Dict) -> Optional[Dict]:
    """Convert an estore_accessories.jsonl record to rag_content.jsonl format."""
    part_number = record.get("part_number", "")
    name = record.get("name", "")
    url = record.get("url", "")

    if not part_number or not name:
        return None

    # Build clean content text
    parts = []
    parts.append(f"Product: {name}")
    parts.append(f"Part Number: {part_number}")

    if record.get("category"):
        parts.append(f"Category: {record['category']}")
    if record.get("subcategory") and record["subcategory"] != "Other":
        parts.append(f"Subcategory: {record['subcategory']}")

    if record.get("price"):
        parts.append(f"Price: ${record['price']}")

    if record.get("designed_for"):
        parts.append(f"Designed For: {record['designed_for']}")
    if record.get("mechanism"):
        parts.append(f"Mechanism: {record['mechanism']}")
    if record.get("dimensions"):
        parts.append(f"Dimensions: {record['dimensions']}")

    # Compatibility — the most important fields
    if record.get("compatible_with"):
        parts.append(f"Compatible With: {record['compatible_with']}")
    if record.get("compatible_chassis"):
        parts.append(f"Compatible Chassis: {record['compatible_chassis']}")
    if record.get("validated_systems"):
        parts.append(f"Validated Systems: {record['validated_systems']}")
    if record.get("validated_chassis"):
        parts.append(f"Validated Chassis: {record['validated_chassis']}")

    if record.get("description"):
        parts.append(f"\n{record['description']}")

    content_text = "\n".join(parts)

    # Build features from clean bullet_features
    features_text = ""
    if record.get("bullet_features"):
        clean_features = clean_bullet_features(record["bullet_features"])
        if clean_features:
            features_text = "\n".join(f"• {f}" for f in clean_features)

    # Build specifications text
    specs_text = ""
    specs = record.get("specifications", {})
    if isinstance(specs, dict) and specs:
        spec_lines = []
        for k, v in specs.items():
            if v and str(v).strip():
                spec_lines.append(f"{k}: {v}")
        if spec_lines:
            specs_text = "\n".join(spec_lines)

    # Build full_text
    full_parts = [f"eStore Accessory: {name}"]
    full_parts.append(content_text)
    if specs_text:
        full_parts.append(f"\nSpecifications:\n{specs_text}")
    if features_text:
        full_parts.append(f"\nFeatures:\n{features_text}")
    if url:
        full_parts.append(f"\nSource URL: {url}")

    full_text = "\n".join(full_parts)

    # Skip very short entries
    if len(full_text) < 100:
        return None

    return {
        "url": url,
        "title": f"eStore {part_number} - {name}",
        "category": record.get("category", "Accessories"),
        "headings": [name, part_number],
        "content": content_text,
        "specifications": specs_text,
        "features": features_text,
        "meta_description": "",
        "full_text": full_text,
        "scraped_at": datetime.now(timezone.utc).isoformat(),
    }


def main():
    parser = argparse.ArgumentParser(description="Ingest eStore accessories into rag_content.jsonl")
    parser.add_argument(
        "--input", default="data/accessories/estore_accessories.jsonl",
        help="Input estore_accessories.jsonl path",
    )
    parser.add_argument(
        "--output", default="data/pages/rag_content.jsonl",
        help="Output rag_content.jsonl to append to",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print stats without writing",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"Error: {input_path} not found")
        return

    # Load existing URLs from rag_content to avoid duplicates
    existing_urls = set()
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                if line.strip():
                    try:
                        rec = json.loads(line)
                        if rec.get("url"):
                            existing_urls.add(rec["url"])
                    except json.JSONDecodeError:
                        pass
    print(f"Existing rag_content entries: {len(existing_urls)}")

    # Process accessories
    total = 0
    converted = 0
    skipped_exists = 0
    skipped_short = 0

    records_to_write = []

    with open(input_path) as f:
        for line in f:
            if not line.strip():
                continue
            total += 1
            record = json.loads(line)

            if record.get("url") in existing_urls:
                skipped_exists += 1
                continue

            rag_record = accessory_to_rag_content(record)
            if rag_record is None:
                skipped_short += 1
                continue

            records_to_write.append(rag_record)
            converted += 1

    print(f"\nResults:")
    print(f"  Total input records: {total}")
    print(f"  Converted: {converted}")
    print(f"  Skipped (already exists): {skipped_exists}")
    print(f"  Skipped (too short): {skipped_short}")

    if args.dry_run:
        print(f"\n[DRY RUN] Would append {converted} records to {output_path}")
        if records_to_write:
            print(f"\nSample output (first record):")
            print(json.dumps(records_to_write[0], indent=2, ensure_ascii=False)[:2000])
        return

    # Append to rag_content.jsonl
    with open(output_path, "a") as f:
        for rec in records_to_write:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\nAppended {converted} records to {output_path}")


if __name__ == "__main__":
    main()
