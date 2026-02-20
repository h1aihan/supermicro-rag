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
        
        # Extract form factor from chassis or name
        form_factor = "Other"
        for ff in ["1U", "2U", "4U", "8U"]:
            if ff in chassis or ff in name:
                form_factor = ff
                break
        if "mid-tower" in name.lower() or "mid-tower" in chassis.lower():
            form_factor = "Mid-Tower"
        product["form_factor"] = form_factor
        
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
        
        # GPU support
        gpu_field = product.get("gpu", "")
        if gpu_field and "supported" in gpu_field.lower():
            tags.add("GPU-capable")
        
        product["tags"] = tags
        
        # Build searchable text blob (all fields concatenated)
        search_parts = [
            name, model, product.get("category", ""),
            product.get("key_features", ""),
            product.get("cpu", ""), product.get("gpu", ""),
            product.get("memory", ""), product.get("storage", ""),
            product.get("network", ""), chassis,
            " ".join(tags)
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
        Filter products using structured criteria from the query planner.
        
        This is much more precise than keyword search — it does exact field
        matching on form_factor and tags, with optional keyword refinement.
        
        Args:
            form_factor: Exact form factor to match ("1U", "2U", etc.)
            tags: Product must have ALL of these tags
            keywords: Additional free-text terms to match in _search_text
            max_results: Maximum products to return
            
        Returns:
            Filtered product list, sorted by relevance
        """
        results = self.products  # Start with all products
        
        # Filter by form factor (exact match)
        if form_factor:
            results = [p for p in results if p.get("form_factor") == form_factor]
        
        # Filter by tags (product must have ALL requested tags)
        if tags:
            tag_set = set(tags)
            results = [p for p in results if tag_set.issubset(p.get("tags", set()))]
        
        # Filter by keywords (all keywords must appear in search text)
        # Multi-word keywords like "nvidia h100" are split into individual terms
        # and ALL terms must appear (but not necessarily as an exact phrase).
        if keywords:
            def matches_all_keywords(product):
                text = product.get("_search_text", "")
                for kw in keywords:
                    # Split multi-word keywords into individual terms
                    terms = kw.lower().split()
                    for term in terms:
                        if term not in text:
                            return False
                return True
            results = [p for p in results if matches_all_keywords(p)]
        
        # Sort: tagged products with more detail sort first
        # Score by how many populated spec fields they have
        def richness(p):
            score = 0
            for field in ["cpu", "gpu", "memory", "storage", "network", "price_range"]:
                if p.get(field):
                    score += 1
            return -score  # Negative for descending sort
        
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
            price = p.get("price_range", "")
            
            line = f"{i}. **{sku or name}**"
            if sku:
                line += f" - {name}"
            if tags:
                line += f" [{tags}]"
            
            details = []
            if p.get("cpu"):
                details.append(f"CPU: {p['cpu']}")
            if p.get("gpu"):
                details.append(f"GPU: {p['gpu']}")
            if p.get("memory"):
                details.append(f"Memory: {p['memory']}")
            if p.get("storage"):
                details.append(f"Storage: {p['storage']}")
            if p.get("chassis"):
                details.append(f"Chassis: {p['chassis']}")
            if price:
                details.append(f"Price: {price}")
            
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
