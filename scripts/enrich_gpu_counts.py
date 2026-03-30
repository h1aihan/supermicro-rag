#!/usr/bin/env python3
"""
Enrich products.jsonl with GPU max counts from indexed web pages/datasheets.

The eStore scraper captures GPU model lists ("Supported GPUs NVIDIA PCIe: ...")
but sometimes misses the "Max GPU Count: Up to X" field. This script
cross-references with the indexed metadata to fill in the gap.

Run: python scripts/enrich_gpu_counts.py
"""
import json, re, sys
from pathlib import Path

PRODUCTS = Path("data/pages/products.jsonl")
METADATA = Path("embeddings/primary_index/metadata.jsonl")

GPU_COUNT_PATTERNS = [
    re.compile(r"Max\s+GPU\s+Count:\s*Up\s+to\s+(\d+)", re.IGNORECASE),
    re.compile(r"Up\s+to\s+(\d+)\s+(?:double-width|single-width|GPU)", re.IGNORECASE),
]


def extract_gpu_count(text: str) -> int:
    for pat in GPU_COUNT_PATTERNS:
        m = pat.search(text)
        if m:
            return int(m.group(1))
    return 0


def load_gpu_counts_from_index() -> dict:
    """Build {sku_stem: max_gpu_count} from indexed web pages and datasheets."""
    counts = {}
    if not METADATA.exists():
        print(f"[WARN] {METADATA} not found")
        return counts

    with open(METADATA) as f:
        for line in f:
            if not line.strip():
                continue
            meta = json.loads(line)
            src = meta.get("source_file", "")
            text = meta.get("text", "")

            count = extract_gpu_count(text)
            if not count:
                continue

            sku_stem = ""
            if src.startswith("web_page_SuperServer_"):
                sku_stem = src.replace("web_page_SuperServer_", "").replace(".txt", "").strip()
            elif src.endswith(".pdf"):
                sku_stem = src.split("__")[0].replace(".pdf", "").strip()
            elif src.startswith("web_product_"):
                sku_stem = src.replace("web_product_", "").replace(".txt", "").strip()

            if sku_stem:
                key = sku_stem.upper().replace(" ", "").replace("_", "-")
                if key not in counts or count > counts[key]:
                    counts[key] = count

    return counts


def sku_to_key(sku: str) -> str:
    return sku.upper().replace(" ", "").replace("_", "-")


def main():
    gpu_index = load_gpu_counts_from_index()
    print(f"[INFO] Found GPU counts for {len(gpu_index)} products from indexed data")

    products = []
    enriched = 0
    with open(PRODUCTS) as f:
        for line in f:
            if not line.strip():
                continue
            p = json.loads(line)
            gpu = p.get("gpu", "")

            already_has_count = bool(
                re.search(r"Max\s+Number\s+of\s+GPU\s+Support\s+\d+", gpu, re.IGNORECASE)
                or re.search(r"Up\s+to\s+(\d+)\s+(?:Double-Wide|double-width|GPU)", gpu, re.IGNORECASE)
            )

            if not already_has_count and gpu:
                name = p.get("name", "")
                sku_match = re.search(r'\(([A-Z]{2,4}\s?-[\w-]+)\)', name)
                sku = sku_match.group(1).strip().replace(" ", "") if sku_match else ""
                model = p.get("model", "").upper().replace(" ", "")

                found_count = 0
                for key_candidate in [sku_to_key(sku), sku_to_key(model)]:
                    for idx_key, cnt in gpu_index.items():
                        if key_candidate and key_candidate in idx_key or idx_key in key_candidate:
                            found_count = max(found_count, cnt)

                if found_count:
                    p["gpu"] = f"Max Number of GPU Support {found_count}; {gpu}"
                    enriched += 1
                    print(f"  ENRICHED: {sku or name[:50]} → Max {found_count} GPUs")

            products.append(p)

    if enriched:
        with open(PRODUCTS, "w") as f:
            for p in products:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
        print(f"\n[DONE] Enriched {enriched} products with GPU max counts")
    else:
        print("\n[DONE] No products needed enrichment")


if __name__ == "__main__":
    main()
