#!/usr/bin/env python3
"""
Structured product catalog for answering listing/enumeration queries.

Instead of searching through 97K embedding chunks, this module loads the 
structured product data and supports fast keyword/category filtering.
This handles queries like:
  - "list all 1U servers"
  - "what GPU systems do you have"
  - "show me gold series products"
  - "what storage solutions are available"
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional

try:
    from src.form_factors import extract_form_factor
except ImportError:
    from form_factors import extract_form_factor


class ProductCatalog:
    """In-memory product catalog built from products.jsonl."""
    
    def __init__(self, products_file: str = None):
        if products_file is None:
            products_file = os.getenv("PRODUCTS_FILE", "data/pages/products.jsonl")
        self.products: List[Dict] = []
        self._load(products_file)
    
    def _load(self, path: str):
        """Load products and enrich with derived fields."""
        p = Path(path)
        if not p.exists():
            print(f"[ProductCatalog] Warning: {path} not found, catalog empty")
            return
        
        with open(p) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                product = json.loads(line)
                self._enrich(product)
                self.products.append(product)
        
        print(f"[ProductCatalog] Loaded {len(self.products)} products")
    
    def _enrich(self, product: Dict):
        """Add derived fields for better searchability."""
        name = product.get("name", "")
        model = product.get("model", "")
        chassis = product.get("chassis", "")
        
        product["form_factor"] = extract_form_factor(name, chassis, model)
        
        # Extract product SKU (e.g., SYS-521C-NR from the name)
        sku_match = re.search(
            r'\(([A-Z]{2,4}\s?-[\w-]+)\)', name
        )
        product["sku"] = sku_match.group(1).strip() if sku_match else ""
        
        # Detect series/program from name or model URL
        tags = set()
        name_lower = name.lower()
        model_lower = model.lower()
        
        if "gold series" in name_lower or "gold-series" in model_lower or "gold-sku" in model_lower:
            tags.add("Gold Series")
        if "clouddc" in name_lower or "clouddc" in model_lower:
            tags.add("CloudDC")
        if "hyper" in name_lower:
            tags.add("Hyper")
        if "edge" in name_lower or "embedded" in name_lower or "iot" in name_lower:
            tags.add("Edge")
        if "storage" in name_lower or "superstorage" in name_lower:
            tags.add("Storage")
        if "gpu" in name_lower:
            tags.add("GPU")
        if "blade" in name_lower:
            tags.add("Blade")
        if "workstation" in name_lower:
            tags.add("Workstation")
        if "mainstream" in name_lower:
            tags.add("Mainstream")
        if "wio" in name_lower:
            tags.add("WIO")
        if "fattwin" in name_lower:
            tags.add("FatTwin")
            tags.add("Twin")
        if "bigtwin" in name_lower:
            tags.add("BigTwin")
            tags.add("Twin")
        if "grandtwin" in name_lower:
            tags.add("GrandTwin")
            tags.add("Twin")
        if "flextwin" in name_lower:
            tags.add("FlexTwin")
            tags.add("Twin")
        if "twin" in name_lower:
            tags.add("Twin")
        if "microcloud" in name_lower or "micro cloud" in name_lower:
            tags.add("MicroCloud")
        
        # GPU support — parse max count and model list separately
        gpu_field = product.get("gpu", "")
        if gpu_field and "supported" in gpu_field.lower():
            tags.add("GPU-capable")

        gpu_max_count = 0
        gpu_models = []
        if gpu_field:
            m_count = re.search(r'Max\s+Number\s+of\s+GPU\s+Support\s+(\d+)', gpu_field, re.IGNORECASE)
            if not m_count:
                m_count = re.search(r'Up\s+to\s+(\d+)\s+(?:Double-Wide|double-width|single-width|GPU)', gpu_field, re.IGNORECASE)
            if m_count:
                gpu_max_count = int(m_count.group(1))
            model_hits = re.findall(r'NVIDIA\s+PCIe:\s*([^,\n]+)', gpu_field)
            gpu_models = [m.strip() for m in model_hits if m.strip()]
        if gpu_max_count:
            tags.add("GPU-capable")
        product["gpu_max_count"] = gpu_max_count
        product["gpu_models"] = gpu_models

        product["tags"] = tags

        # --- Parsed spec fields for structured filtering ---
        cpu_field = product.get("cpu", "")
        storage_field = product.get("storage", "")

        cpu_count = 0
        m = re.search(r'(?:Max\s+Number\s+of\s+CPU|Processor(?:s)?)\s*(\d)', cpu_field, re.IGNORECASE)
        if m:
            cpu_count = int(m.group(1))
        elif "dual" in cpu_field.lower() or "two" in cpu_field.lower():
            cpu_count = 2
        elif "single" in cpu_field.lower() or "one" in cpu_field.lower():
            cpu_count = 1
        product["cpu_count"] = cpu_count

        all_text_lower = f"{name} {cpu_field} {product.get('category', '')}".lower()
        sku_upper = product.get("sku", "").upper().replace(" ", "")
        cpu_family = ""
        if "epyc" in all_text_lower:
            cpu_family = "EPYC"
        elif "xeon" in all_text_lower:
            cpu_family = "Xeon"
        elif sku_upper.startswith("AS-"):
            cpu_family = "EPYC"
        elif sku_upper.startswith("SYS-"):
            cpu_family = "Xeon"
        product["cpu_family"] = cpu_family

        # Detect platform generation from model URL or product name (H12-H14, X12-X14)
        model_url = product.get("model", "").lower()
        platform_gen = ""
        for gen in ["h14", "h13", "h12", "x14", "x13", "x12"]:
            if gen in model_url or gen in name_lower:
                platform_gen = gen.upper()
                break
        if not platform_gen and sku_upper.startswith("AS-"):
            platform_gen = "AMD"
        elif not platform_gen and sku_upper.startswith("SYS-"):
            platform_gen = "Intel"
        product["platform_generation"] = platform_gen

        # Detect specific CPU series from cpu field text and platform generation
        cpu_series = []
        if "9005" in cpu_field or "turin" in cpu_field.lower():
            cpu_series.append("EPYC 9005")
        if "9004" in cpu_field or "genoa" in cpu_field.lower() or "bergamo" in cpu_field.lower():
            cpu_series.append("EPYC 9004")
        if "7003" in cpu_field or "milan" in cpu_field.lower():
            cpu_series.append("EPYC 7003")
        if "xeon 6" in cpu_field.lower() or "granite" in cpu_field.lower() or "sierra" in cpu_field.lower():
            cpu_series.append("Xeon 6")
        if "5th gen" in cpu_field.lower() or "emerald" in cpu_field.lower():
            cpu_series.append("Xeon 5th Gen")
        if "4th gen" in cpu_field.lower() or "sapphire" in cpu_field.lower():
            cpu_series.append("Xeon 4th Gen")

        # Infer additional CPU series from platform generation when data is incomplete
        if platform_gen == "H14" and "EPYC 9005" not in cpu_series:
            cpu_series.append("EPYC 9005")
        if platform_gen == "H14" and "EPYC 9004" not in cpu_series and cpu_family == "EPYC":
            cpu_series.append("EPYC 9004")
        if platform_gen == "H13" and not cpu_series and cpu_family == "EPYC":
            cpu_series.extend(["EPYC 9004", "EPYC 9005"])

        product["cpu_series"] = cpu_series

        drive_bay_count = 0
        drive_size = ""
        m = re.search(r'(\d+)\s+(?:\d[\d./\"\']*\s*)?(?:Hot-Swap|hot-swap|Internal)', storage_field)
        if m:
            drive_bay_count = int(m.group(1))
        if '3.5' in storage_field:
            drive_size = "3.5"
        elif '2.5' in storage_field:
            drive_size = "2.5"
        product["drive_bay_count"] = drive_bay_count
        product["drive_size"] = drive_size

        # Build searchable text blob (all fields concatenated)
        search_parts = [
            name, model, product.get("category", ""),
            product.get("key_features", ""),
            cpu_field, product.get("gpu", ""),
            product.get("memory", ""), storage_field,
            product.get("network", ""), chassis,
            " ".join(tags),
            cpu_family,
            " ".join(cpu_series),
            platform_gen,
        ]
        product["_search_text"] = " ".join(search_parts).lower()
    
    def search(self, query: str, max_results: int = 50) -> List[Dict]:
        """
        Search products by matching query terms against all product fields.
        
        Uses a simple scoring system: each query term that appears in the 
        product's searchable text adds 1 point. Products are ranked by score.
        
        Args:
            query: Natural language query
            max_results: Maximum products to return
            
        Returns:
            List of matching products sorted by relevance
        """
        # Tokenize query into meaningful terms (skip very common words)
        stop_words = {
            # English function words
            'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been',
            'do', 'does', 'did', 'will', 'would', 'could', 'should',
            'can', 'you', 'me', 'my', 'your', 'i', 'we', 'they',
            'what', 'which', 'how', 'give', 'show', 'tell', 'get',
            'have', 'has', 'of', 'in', 'on', 'at', 'to', 'for',
            'with', 'and', 'or', 'not', 'all', 'any', 'some',
            'about', 'more', 'does', 'support',
            'please', 'hi', 'hello', 'thanks',
            # Domain-generic vocabulary (appear in every product, don't differentiate)
            'supermicro', 'list', 'products', 'product', 'available',
            'server', 'servers', 'system', 'systems', 'solution', 'solutions',
            'sku', 'skus', 'model', 'models', 'series',
        }
        
        raw_terms = re.findall(r'[\w-]+', query.lower())
        terms = [t for t in raw_terms if t not in stop_words and len(t) > 1]
        
        if not terms:
            return []
        
        # Generate stem variants for each term
        def stem_variants(term):
            """Generate common morphological variants of a term."""
            variants = {term}
            # Plural ↔ singular
            if term.endswith('ies') and len(term) > 4:
                variants.add(term[:-3] + 'y')
            elif term.endswith('es') and len(term) > 3:
                variants.add(term[:-2])
                variants.add(term[:-1])
            elif term.endswith('s') and not term.endswith('ss') and len(term) > 3:
                variants.add(term[:-1])
            else:
                variants.add(term + 's')
            # -en suffix (golden → gold)
            if term.endswith('en') and len(term) > 4:
                variants.add(term[:-2])
            # -ing suffix
            if term.endswith('ing') and len(term) > 5:
                variants.add(term[:-3])
                variants.add(term[:-3] + 'e')
            return variants
        
        term_variants = {term: stem_variants(term) for term in terms}
        
        # Score each product
        scored = []
        for product in self.products:
            text = product["_search_text"]
            tags = product.get("tags", set())
            score = 0
            matched_terms = 0
            
            for term in terms:
                term_score = 0
                tag_bonus = 0
                
                # Check text matches (exact → stem variant)
                if term in text:
                    term_score = max(term_score, 1.0)
                else:
                    for variant in term_variants[term]:
                        if variant != term and variant in text:
                            term_score = max(term_score, 0.8)
                            break
                
                # Check tag matches — tags are FIRST-CLASS matches.
                # If "gold" matches tag "Gold Series", it counts as a full match.
                # Also provides a ranking bonus so tag-matched products sort higher.
                for tag in tags:
                    tag_lower = tag.lower()
                    for variant in term_variants[term]:
                        if variant in tag_lower:
                            term_score = max(term_score, 1.0)  # Tag = full match
                            tag_bonus = 0.3  # Ranking bonus for tag relevance
                            break
                
                if term_score > 0:
                    matched_terms += 1
                score += term_score + tag_bonus
            
            if score > 0:
                # Relevance = fraction of query terms that matched something
                relevance = matched_terms / len(terms)
                scored.append((product, score, relevance))
        
        # Require majority of query terms to match (filters out single-term-only matches
        # in multi-term queries, e.g., "global skus" matching only on "sku")
        # For 2-term queries: need both terms (>0.5 = both must match)
        # For 3-term queries: need at least 2 (>0.5 = 2/3)
        scored = [(p, s, r) for p, s, r in scored if r > 0.5]
        
        # Sort by score descending, then by name
        scored.sort(key=lambda x: (-x[1], x[0].get("name", "")))
        
        return [p for p, _ , _ in scored[:max_results]]
    
    def filter_structured(
        self,
        form_factor: Optional[str] = None,
        tags: Optional[List[str]] = None,
        keywords: Optional[List[str]] = None,
        max_results: int = 50,
    ) -> List[Dict]:
        """
        Filter products using exact structural criteria from the query planner.

        Only uses unambiguous exact-match filters (form_factor, tags).
        Keywords are accepted for API compatibility but NOT used for filtering
        -- the LLM handles vocabulary matching natively when it receives the
        filtered product list alongside RAG context.

        Args:
            form_factor: Exact form factor to match ("1U", "2U", etc.)
            tags: Product must have ALL of these tags
            keywords: Accepted but unused (kept for call-site compatibility)
            max_results: Maximum products to return

        Returns:
            Filtered product list, sorted by data richness
        """
        results = self.products

        if form_factor:
            results = [p for p in results if p.get("form_factor") == form_factor]

        if tags:
            tag_set = set(tags)
            results = [p for p in results if tag_set.issubset(p.get("tags", set()))]

        def richness(p):
            score = 0
            for field in ["cpu", "gpu", "memory", "storage", "network", "price_range"]:
                if p.get(field):
                    score += 1
            return -score
        results.sort(key=lambda p: (richness(p), p.get("name", "")))

        return results[:max_results]

    def format_for_llm(self, products: List[Dict], max_products: int = 30) -> str:
        """
        Format product list as structured text for LLM context.
        
        Args:
            products: List of product dicts
            max_products: Maximum products to include
            
        Returns:
            Formatted text suitable for LLM context
        """
        if not products:
            return "No matching products found in the catalog."
        
        lines = [f"Found {len(products)} matching products:\n"]
        
        for i, p in enumerate(products[:max_products], 1):
            sku = p.get("sku", "")
            name = p.get("name", "")
            tags = ", ".join(sorted(p.get("tags", set()))) or p.get("category", "")
            # price intentionally excluded from LLM context
            
            line = f"{i}. **{sku or name}**"
            if sku:
                line += f" - {name}"
            if tags:
                line += f" [{tags}]"
            
            details = []
            if p.get("cpu"):
                cpu_detail = p['cpu']
                if p.get("cpu_series"):
                    cpu_detail += f" (Supports: {', '.join(p['cpu_series'])})"
                details.append(f"CPU: {cpu_detail}")
            if p.get("gpu"):
                gpu_count = p.get("gpu_max_count", 0)
                if gpu_count:
                    details.append(f"GPU: Max {gpu_count} GPUs")
                elif p.get("gpu_models"):
                    details.append("GPU: Yes (see datasheet for max GPU count)")
                else:
                    details.append(f"GPU: {p['gpu']}")
            if p.get("memory"):
                details.append(f"Memory: {p['memory']}")
            if p.get("storage"):
                details.append(f"Storage: {p['storage']}")
            if p.get("chassis"):
                details.append(f"Chassis: {p['chassis']}")
            # Price data omitted — crawled prices go stale quickly.
            # Users are directed to the eStore for current pricing.
            
            if details:
                line += "\n   " + " | ".join(details)
            
            lines.append(line)
        
        if len(products) > max_products:
            lines.append(f"\n... and {len(products) - max_products} more products")
        
        return "\n".join(lines)


def is_listing_query(query: str) -> bool:
    """
    Detect if a query is asking to list/enumerate products.
    
    Uses general patterns, NOT hardcoded product terms.
    This catches questions like:
      - "list all 1U servers"
      - "what GPU systems do you have"  
      - "show me storage solutions"
      - "what are golden skus"
      - "give me a list of global skus"
    """
    q = query.lower()
    
    # Pattern 1: Explicit listing request
    # "list ...", "give me a list of ..."
    if re.search(r'\b(list|enumerate|catalog)\b', q):
        return True
    
    # Pattern 2: "what are [product term]s" (plural "are" signals enumeration)
    # "what are golden skus", "what are the AMD EPYC servers"
    # NOT "what is X" (singular = info request, not listing)
    if re.search(r'\bwhat\s+are\b.*\b(skus?|servers?|systems?|products?|solutions?|models?|series)\b', q):
        return True
    
    # Pattern 3: "what [terms] do you have / are available / are there"
    if re.search(r'\bwhat\b.*\b(do you have|are available|do you offer|are there)\b', q):
        return True
    
    # Reusable product-term pattern for patterns below
    _pt = r'(skus?|servers?|systems?|products?|solutions?|models?|series|configurations?)'
    
    # Pattern 4: "show me" / "give me" + product-related terms
    if re.search(r'\b(show|give)\s+me\b.*\b' + _pt, q):
        return True
    
    # Pattern 5: "how many [products]" 
    if re.search(r'\bhow many\b.*\b' + _pt, q):
        return True
    
    # Pattern 6: "all [product type]s"
    if re.search(r'\ball\s+\w*\s*' + _pt, q):
        return True
    
    # Pattern 7: Short noun-phrase queries (1-4 words) containing a product term
    # e.g., "golden skus", "GPU servers", "1U systems", "edge servers", "gold series"
    # These are implicit listing requests — the user typed a category, not a question.
    words = q.strip().split()
    if len(words) <= 4 and re.search(r'\b' + _pt, q):
        return True
    
    return False


if __name__ == "__main__":
    import sys
    
    catalog = ProductCatalog()
    
    # Interactive test
    queries = sys.argv[1:] or [
        "list all 1U servers",
        "what GPU systems do you have",
        "golden skus",
        "gold series products",
        "global skus", 
        "storage solutions",
        "edge servers",
        "what are the AMD EPYC servers",
        "2U CloudDC servers",
        "list workstations",
    ]
    
    for query in queries:
        is_listing = is_listing_query(query)
        results = catalog.search(query)
        print(f"\n{'='*60}")
        print(f"Query: '{query}'")
        print(f"Is listing query: {is_listing}")
        print(f"Results: {len(results)} products")
        for p in results[:5]:
            print(f"  - {p.get('sku', '')} | {p.get('name', '')} | tags={p.get('tags', set())}")
        if len(results) > 5:
            print(f"  ... and {len(results)-5} more")
