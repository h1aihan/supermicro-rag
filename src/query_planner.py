#!/usr/bin/env python3
"""
LLM-based query planner for the Supermicro RAG system.

Replaces fragile regex routing with a fast LLM call that extracts:
  - User intent (list, detail, compare, general)
  - Structured product filters (form factor, tags, keywords)
  - An optimized search query for RAG retrieval

Uses a cheap/fast model (e.g., claude-3-5-haiku) to keep latency < 1s.
Falls back to a simple heuristic if the LLM call fails.
"""

import json
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv

_repo_root_env = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(dotenv_path=_repo_root_env, override=False)


# =============================================================================
# Query Plan data structure
# =============================================================================

@dataclass
class QueryPlan:
    """Structured plan produced by the query planner."""

    # Intent classification
    intent: str = "general"  # "list" | "detail" | "compare" | "general"

    # Normalized product codes extracted by the LLM
    product_codes: List[str] = field(default_factory=list)  # ["SYS-521GE-TNRT", "SYS-421GE-TNRT"]

    # Structured filters for the product catalog
    form_factor: Optional[str] = None        # "1U", "2U", "4U", "8U", "Mid-Tower"
    tags: List[str] = field(default_factory=list)  # ["Gold Series", "GPU", ...]
    keywords: List[str] = field(default_factory=list)  # free-text terms

    # Optimized queries for RAG vector/keyword search.
    # Single item = one search. Multiple items = split retrieval (one per topic).
    search_queries: List[str] = field(default_factory=list)

    # Kept for backward compat / simple access
    @property
    def search_query(self) -> str:
        return self.search_queries[0] if self.search_queries else ""

    # Whether the catalog and/or RAG should be consulted
    use_catalog: bool = False
    use_rag: bool = True

    def __repr__(self):
        parts = [f"intent={self.intent}"]
        if self.product_codes:
            parts.append(f"codes={self.product_codes}")
        if len(self.search_queries) > 1:
            parts.append(f"queries({len(self.search_queries)})={self.search_queries}")
        if self.form_factor:
            parts.append(f"ff={self.form_factor}")
        if self.tags:
            parts.append(f"tags={self.tags}")
        if self.keywords:
            parts.append(f"kw={self.keywords}")
        parts.append(f"catalog={'Y' if self.use_catalog else 'N'}")
        parts.append(f"rag={'Y' if self.use_rag else 'N'}")
        return f"QueryPlan({', '.join(parts)})"


# =============================================================================
# Planner prompt
# =============================================================================

PLANNER_SYSTEM = """You are a query router for a Supermicro server product database. Given a user's question, output a JSON plan.

## PRODUCT TAXONOMY (use these exact strings)

Form factors: "1U", "2U", "4U", "8U", "Mid-Tower"

Tags (product categories/series):
- "Gold Series"    — Pre-configured, quick-ship Gold Series SKUs (suffix -G1, -G2)
- "CloudDC"        — Cloud-optimized data center servers
- "Hyper"          — High-performance Hyper servers
- "Edge"           — Edge/embedded/IoT servers
- "Storage"        — Storage-focused SuperStorage servers
- "GPU"            — GPU-optimized servers (for AI/ML/HPC)
- "GPU-capable"    — Servers that support GPUs but aren't GPU-primary
- "Blade"          — Blade servers
- "Workstation"    — Workstation-class systems
- "Mainstream"     — Mainstream/general-purpose servers
- "WIO"            — WIO (Work-optimized I/O) servers
- "Twin"           — Multi-node Twin servers (general Twin family)
- "BigTwin"        — BigTwin multi-node servers (2U 2-node or 4-node, X12/X13/X14 generations)
- "FatTwin"        — FatTwin multi-node servers
- "GrandTwin"      — GrandTwin multi-node servers
- "FlexTwin"       — FlexTwin liquid-cooled HPC multi-node servers
- "MicroCloud"     — Multi-node MicroCloud servers (high-density 3U, 5/8/10/12 nodes)

## PLATFORM GENERATIONS
Supermicro uses different naming prefixes for Intel vs AMD platforms:
- X-series (X12, X13, X14) = Intel platforms. Server SKUs use "SYS-" prefix.
- H-series (H12, H13, H14) = AMD platforms. Server SKUs use "AS-" prefix (A+ Servers).
- Product families like Hyper, CloudDC, Mainstream, etc. may span BOTH Intel and AMD platforms.
  Example: "H13 Hyper" = AMD EPYC-based Hyper (AS- prefix), "X14 Hyper" = Intel Xeon 6-based Hyper (SYS- prefix).

Intel CPU generation to Supermicro platform mapping (important — do not confuse these):
- 3rd Gen Intel Xeon Scalable (Ice Lake) = X12
- 4th Gen Intel Xeon Scalable (Sapphire Rapids) = X13
- 5th Gen Intel Xeon Scalable (Emerald Rapids) = X13
- Intel Xeon 6 / 6th Gen (Granite Rapids P-cores, Sierra Forest E-cores) = X14

AMD CPU generation to Supermicro platform mapping:
- AMD EPYC 7003 (Milan) = H12
- AMD EPYC 9004 (Genoa/Bergamo) = H13
- AMD EPYC 9005 (Turin) = H14

## GPU / ACCELERATOR QUERIES
- "HGX" is NVIDIA's GPU baseboard platform name, NOT a Supermicro product code. NEVER put "HGX-B200", "HGX-H100", etc. in product_codes. Use them as search keywords only.
- GPU names (H100, H200, B200, etc.) often do NOT appear in Supermicro datasheet filenames. The datasheets use Supermicro model numbers instead (e.g. sys-421ge-tnrt.pdf, not "H100.pdf").
- CRITICAL: When generating search_queries for GPU systems, EVERY query MUST include the corresponding Supermicro model number(s) from the mapping below. Generic queries like "H100 air cooled specifications" will NOT find relevant documents.
- GPU-to-Supermicro-server mapping for search queries:
  * H100 → ALWAYS include "SYS-421GE" or "SYS-521GE" in EVERY search query. Do NOT use "SuperCluster" for H100 (SuperCluster = B200 only).
  * H200 → ALWAYS include "SYS-821GE" in EVERY search query.
  * B200 air-cooled → include "SuperCluster B200 10U Air Cooled" in search queries
  * B200 liquid-cooled → include "SuperCluster B200 4U Liquid Cooled" or "SYS-421GE-NBRT-LCC" in search queries
- For GPU system hardware specs (PSU, power, cooling, form factor), include "specifications datasheet" alongside the Supermicro model number.

## IMPORTANT DISTINCTIONS
- "Gold Series" / "golden SKUs" = specific pre-configured product line (use_catalog=true, tags=["Gold Series"])
- "Global SKU Program" = a logistics/fulfillment program; Supermicro's website has an official list (Systems, Motherboards, Chassis). For "list of global skus" use_rag=true and set search_query to target that list page, e.g. "Supermicro Global SKU Program list Systems Motherboards Chassis SKU table" so retrieval finds the page with the enumerated SKUs.
- "IPMI", "BMC", "DCSCM", etc. = technical concepts, not products (use_catalog=false)

## OUTPUT FORMAT (JSON only, no markdown)
{
  "intent": "list|detail|compare|general|follow_up",
  "product_codes": [],
  "form_factor": null or "1U"|"2U"|"4U"|"8U"|"Mid-Tower",
  "tags": [],
  "keywords": [],
  "search_queries": ["query1", "query2"],
  "use_catalog": true/false,
  "use_rag": true/false
}

## search_queries FIELD
This is a LIST of search queries. Use MULTIPLE queries when the user asks about DISTINCT topics so each gets its own retrieval:
- "Compare SYS-521GE and SYS-421GE" → ["SYS-521GE-TNRT specifications datasheet", "SYS-421GE-TNRT specifications datasheet"]
- "List both MicroCloud and BigTwin" → ["Supermicro MicroCloud multi-node 3U servers AS-3015MR", "Supermicro BigTwin multi-node 2U servers SYS-221BT X14 X13 X12"]
- "Give me specs on 521GE, 821GE, and 3015MR" → ["SYS-521GE-TNRT specifications", "SYS-821GE-TNHR specifications", "AS-3015MR-H10TNR specifications"]
- "Difference between 1U and 2U GPU servers" → ["Supermicro 1U GPU servers specifications", "Supermicro 2U GPU servers specifications"]
Use a SINGLE query when the user asks about ONE topic:
- "What are the specs of SYS-521GE?" → ["SYS-521GE-TNRT specifications GPU server 5U datasheet"]
- "List all 1U servers" → ["Supermicro 1U servers"]
- "How does IPMI work?" → ["Supermicro IPMI BMC management"]

## PRODUCT CODE EXTRACTION (product_codes field)
Extract and normalize product identifiers that the user EXPLICITLY mentions. Do NOT invent or guess model numbers.
- Full codes: keep as-is (e.g. "SYS-521GE-TNRT" → "SYS-521GE-TNRT")
- Partial codes the user typed: add the most likely prefix (e.g. "521GE" → "SYS-521GE", "3015MR" → "AS-3015MR")
- Misspelled/spaced: fix them (e.g. "521 GE" → "SYS-521GE", "sys521ge tnrt" → "SYS-521GE-TNRT", "micro cloud" → keep as keyword not code)
- Motherboards: "X13DEI" → "X13DEI", "x13 dei" → "X13DEI"
- From conversation context: if intent=follow_up, include the product code from the conversation
- Family names like "MicroCloud", "BigTwin", "Twin" are NOT product codes — put them in tags instead
- Generation names like "H13", "H14", "X13", "X14" alone are NOT product codes — put them in keywords
- NEVER fabricate or guess a full model number. If the user says "H13 Hyper" or "X14 CloudDC", those are family+generation references, not product codes. Leave product_codes empty and use descriptive search queries instead.

## PRODUCT FAMILIES vs SPECIFIC MODELS
When the user asks about a product FAMILY or GENERATION (e.g. "H13 Hyper", "X14 CloudDC", "X13 GrandTwin", "FatTwin with AMD"), use natural-language search queries with the family name, generation, and platform — do NOT try to construct model numbers you are unsure about. The search engine works well with descriptive queries.

GOOD (family/generation query — no codes fabricated):
- "Compare H14 Hyper 1U and H13 Hyper 2U" → product_codes=[], search_queries=["Supermicro H14 Hyper 1U AMD EPYC specifications datasheet", "Supermicro H13 Hyper 2U AMD EPYC specifications datasheet"]
- "X13 vs X14 GrandTwin" → product_codes=[], search_queries=["Supermicro X13 GrandTwin specifications datasheet", "Supermicro X14 GrandTwin specifications datasheet"]
- "List FatTwin servers with AMD" → product_codes=[], search_queries=["Supermicro FatTwin AMD EPYC multi-node servers"]

BAD (fabricating model numbers the user never mentioned):
- "Compare H14 Hyper and H13 Hyper" → product_codes=["SYS-1124U-TNRT", "SYS-2124U-TNRT"]  ← WRONG: these codes are made up
- "X13 GrandTwin" → product_codes=["SYS-221GT-TNR"]  ← WRONG: guessed model number

When the user provides an ACTUAL model number (even partial), include it:
- "Tell me about 521GE" → product_codes=["SYS-521GE"], search_queries=["SYS-521GE specifications datasheet"]
- "Compare AS-2025HS-TNR and SYS-222H-TN" → product_codes=["AS-2025HS-TNR", "SYS-222H-TN"], search_queries=["AS-2025HS-TNR specifications datasheet", "SYS-222H-TN specifications datasheet"]

## INTENT RULES
- "list": user wants to enumerate/browse products, OR asks "what are" about a product category/series. Examples:
  - "list all 1U servers" → list
  - "golden skus" → list (Gold Series products)
  - "what are golden skus?" → list (this asks about a product line, not a concept)
  - "GPU systems" → list
  - "what GPU servers do you have?" → list
  - "What are the X14 UP WIO Systems?" → list
- "detail": user asks about a specific product or model (e.g., "specs of SYS-521GE-TNRT", "tell me about the 8U GPU server")
- "compare": user wants to compare products (e.g., "compare 1U vs 2U servers", "difference between Gold and standard")
- "general": technical/conceptual questions not about specific products (e.g., "how does IPMI work", "what is DCSCM", "what is the Global SKU program")
- "follow_up": ONLY when conversation context is provided AND the current question clearly refers to the same product/topic just discussed (e.g. "what about its storage?", "how many GPUs does it support?", "what's the price?"). Do NOT use follow_up for a NEW product question (e.g. user said "521ge" then "x13" — "x13" is the X13 motherboard family, not a follow-up about 521GE).

KEY DISTINCTION: "what are [product line]?" → list (products).  "what is [concept]?" → general (explanation).
- "What are golden skus?" → list (Gold Series is a product line with specific SKUs to enumerate)
- "What is the Global SKU program?" → general (a logistics program, not a product line)
- "What are CloudDC servers?" → list (CloudDC is a product family with models to enumerate)
- "What is IPMI?" → general (IPMI is a technical concept)
- "H14" or "X14" or "H13 servers" → list (these are product platform generations with multiple product families to enumerate, use_catalog=true). Generate MULTIPLE search queries covering different product families within the platform so retrieval hits actual datasheets rather than one broad query that only matches product briefs.

## GUIDELINES
- For "list" intent: use_catalog=true, use_rag=true (catalog for product data, RAG for supplementary context)
- For "detail" intent: use_catalog=true (if asking about a specific model), use_rag=true
- For "compare" intent: use_catalog=true, use_rag=true
- For "general" intent: use_catalog=false, use_rag=true
- For "follow_up" intent: use_catalog=true, use_rag=true. Combine the product code from the conversation with the aspect asked in search_queries (e.g. ["SYS-521GE-TNRT storage options"]). keywords may include the product code so catalog can filter to that product.
- IMPORTANT: whenever a product tag is relevant (Gold Series, GPU, CloudDC, Hyper, etc.), set use_catalog=true so the catalog can provide structured product data, even if the intent is "general".
- keywords: extract product-specific terms NOT covered by tags/form_factor (e.g., "NVIDIA", "AMD EPYC", "NVMe", "h13", "x14")
- search_queries: rewrite into effective search queries:
  * When the user provided a real product code, lead with it (e.g. "SYS-111C-NR GPU support specifications").
  * When the user asked about a product family or generation without a specific model number, use descriptive natural-language queries (e.g. "Supermicro H13 Hyper 2U specifications datasheet").
  * For "detail" with a single product, use ONE search query with the product code + topic.
  * For "compare" with 2+ specific products, use SEPARATE search queries per product.
  * For "compare" across generations/families, use SEPARATE descriptive queries per generation.
- For product families spanning multiple generations (e.g. MicroCloud, Twin, BigTwin): try to cover all relevant generations in the search queries, not just the latest.
- If unsure whether something is a product or concept, set use_catalog=true and use_rag=true"""


# =============================================================================
# Token usage tracking (planner calls)
# =============================================================================

_planner_usage = defaultdict(int)


def get_planner_usage() -> dict:
    """Return accumulated planner token usage since last reset."""
    return dict(_planner_usage)


def reset_planner_usage():
    _planner_usage.clear()


# =============================================================================
# LLM call (fast model)
# =============================================================================

def _build_planner_user_message(query: str, conversation_context: Optional[str] = None) -> str:
    """Build the user message for the planner. When conversation is provided, prepend it so the LLM can detect follow-ups."""
    if not conversation_context or not conversation_context.strip():
        return query
    return f"""Conversation:
{conversation_context.strip()}

Current question: {query}

Output your JSON plan. If the current question is a follow-up to the conversation (user referring to the same product/topic), use intent=follow_up and set search_query to the product from the conversation plus the aspect asked (e.g. SYS-521GE-TNRT storage, SYS-521GE-TNRT GPU support). If the current question is a NEW topic (e.g. a different product or board like "x13"), use detail or general, not follow_up."""


def _call_planner_llm(query: str, conversation_context: Optional[str] = None) -> Optional[str]:
    """
    Call a fast LLM to plan the query. Returns raw JSON string or None on failure.
    
    When conversation_context is provided, the LLM can output intent=follow_up and a
    search_query that combines the conversation's product with the current question.
    """
    user_content = _build_planner_user_message(query, conversation_context)
    provider = os.getenv("LLM_PROVIDER", "openai")

    if provider == "anthropic":
        try:
            from anthropic import Anthropic
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                return None

            client = Anthropic(api_key=api_key)
            planner_model = os.getenv("PLANNER_MODEL", "claude-haiku-4-5")

            response = client.messages.create(
                model=planner_model,
                max_tokens=300,
                temperature=0.0,
                system=[{
                    "type": "text",
                    "text": PLANNER_SYSTEM,
                    "cache_control": {"type": "ephemeral"},
                }],
                messages=[{"role": "user", "content": user_content}],
            )
            if hasattr(response, "usage") and response.usage:
                _planner_usage["input_tokens"] += response.usage.input_tokens
                _planner_usage["output_tokens"] += response.usage.output_tokens
                _planner_usage["cache_read"] += getattr(response.usage, "cache_read_input_tokens", 0)
                _planner_usage["cache_creation"] += getattr(response.usage, "cache_creation_input_tokens", 0)
                _planner_usage["calls"] += 1
            return response.content[0].text
        except Exception as e:
            print(f"[QueryPlanner] LLM error: {e}")
            return None

    elif provider == "openai":
        try:
            from openai import OpenAI
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                return None

            client = OpenAI(api_key=api_key)
            planner_model = os.getenv("PLANNER_MODEL", "gpt-4o-mini")

            response = client.chat.completions.create(
                model=planner_model,
                messages=[
                    {"role": "system", "content": PLANNER_SYSTEM},
                    {"role": "user", "content": user_content},
                ],
                max_tokens=300,
                temperature=0.0,
            )
            if hasattr(response, "usage") and response.usage:
                _planner_usage["input_tokens"] += response.usage.prompt_tokens
                _planner_usage["output_tokens"] += response.usage.completion_tokens
                _planner_usage["calls"] += 1
            return response.choices[0].message.content
        except Exception as e:
            print(f"[QueryPlanner] LLM error: {e}")
            return None

    return None


# =============================================================================
# Parse LLM response into QueryPlan
# =============================================================================

VALID_INTENTS = {"list", "detail", "compare", "general", "follow_up"}
VALID_FORM_FACTORS = {"1U", "2U", "4U", "8U", "Mid-Tower"}
VALID_TAGS = {
    "Gold Series", "CloudDC", "Hyper", "Edge", "Storage", "GPU",
    "GPU-capable", "Blade", "Workstation", "Mainstream", "WIO",
    "Twin", "BigTwin", "FatTwin", "GrandTwin", "FlexTwin", "MicroCloud",
}


def _extract_first_json_object(text: str) -> str:
    """Extract the first complete {...} object so trailing text (e.g. '**Expla') is ignored."""
    start = text.find("{")
    if start < 0:
        return text
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return text[start:]


def _parse_plan(raw: str, original_query: str) -> QueryPlan:
    """Parse the LLM JSON response into a QueryPlan, with validation."""
    try:
        # Strip markdown code fences if present
        text = raw.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
        text = text.strip()
        # Ignore trailing text after the JSON object (e.g. "**Expla" or truncated prose)
        text = _extract_first_json_object(text)
        data = json.loads(text)
    except (json.JSONDecodeError, IndexError):
        print(f"[QueryPlanner] Failed to parse JSON: {raw[:200]}")
        return _fallback_plan(original_query)
    
    plan = QueryPlan()
    
    # Intent
    intent = data.get("intent", "general")
    plan.intent = intent if intent in VALID_INTENTS else "general"
    
    # Product codes (LLM-normalized)
    raw_codes = data.get("product_codes", [])
    if isinstance(raw_codes, list):
        plan.product_codes = [str(c).strip() for c in raw_codes if c and str(c).strip()]
    
    # Form factor
    ff = data.get("form_factor")
    if ff and ff in VALID_FORM_FACTORS:
        plan.form_factor = ff
    
    # Tags — validate against known set
    raw_tags = data.get("tags", [])
    if isinstance(raw_tags, list):
        plan.tags = [t for t in raw_tags if t in VALID_TAGS]
    
    # Keywords
    raw_kw = data.get("keywords", [])
    if isinstance(raw_kw, list):
        plan.keywords = [str(k).lower() for k in raw_kw if k]
    
    # Search queries (list) — accept both "search_queries" (new) and "search_query" (old/single)
    raw_queries = data.get("search_queries")
    if isinstance(raw_queries, list) and raw_queries:
        plan.search_queries = [str(q).strip() for q in raw_queries if q and str(q).strip()]
    if not plan.search_queries:
        sq = data.get("search_query", original_query) or original_query
        plan.search_queries = [sq]
    
    # Catalog / RAG flags
    plan.use_catalog = bool(data.get("use_catalog", False))
    plan.use_rag = bool(data.get("use_rag", True))
    
    # Safety: ensure at least one retrieval path
    if not plan.use_catalog and not plan.use_rag:
        plan.use_rag = True
    
    return plan


# =============================================================================
# Heuristic fallback (no LLM)
# =============================================================================

def _fallback_plan(query: str) -> QueryPlan:
    """
    Simple rule-based fallback when the planner LLM is unavailable.
    
    This is essentially the old is_listing_query + keyword logic,
    kept as a safety net.
    """
    import re
    q = query.lower()
    plan = QueryPlan(search_queries=[query], use_rag=True)
    
    # Detect listing intent
    _pt = r'(skus?|servers?|systems?|products?|solutions?|models?|series|configurations?)'
    is_list = bool(
        re.search(r'\b(list|enumerate|catalog)\b', q)
        or re.search(r'\bwhat\s+are\b.*\b' + _pt, q)
        or re.search(r'\bwhat\b.*\b(do you have|are available|do you offer|are there)\b', q)
        or re.search(r'\b(show|give)\s+me\b.*\b' + _pt, q)
        or (len(q.split()) <= 4 and re.search(r'\b' + _pt, q))
    )
    
    if is_list:
        plan.intent = "list"
        plan.use_catalog = True
    
    # Detect product-related terms for catalog
    if re.search(r'\b(SYS|AS|SSG|SBI|AOC)-[\w-]+', query, re.IGNORECASE):
        plan.intent = "detail"
        plan.use_catalog = True
    
    # Extract simple tag hints
    tag_map = {
        "gold": "Gold Series", "golden": "Gold Series",
        "clouddc": "CloudDC", "cloud dc": "CloudDC",
        "hyper": "Hyper", "edge": "Edge", "storage": "Storage",
        "gpu": "GPU", "blade": "Blade", "workstation": "Workstation",
        "wio": "WIO", "twin": "Twin", "mainstream": "Mainstream",
        "grandtwin": "GrandTwin", "grand twin": "GrandTwin",
        "flextwin": "FlexTwin", "flex twin": "FlexTwin",
        "fattwin": "FatTwin", "fat twin": "FatTwin",
        "bigtwin": "BigTwin", "big twin": "BigTwin",
        "microcloud": "MicroCloud", "micro cloud": "MicroCloud",
    }
    for keyword, tag in tag_map.items():
        if keyword in q:
            plan.tags.append(tag)
            plan.use_catalog = True
    
    # Extract form factor
    for ff in ["1U", "2U", "4U", "8U"]:
        if ff.lower() in q or ff in query:
            plan.form_factor = ff
            plan.use_catalog = True
            break
    
    print(f"[QueryPlanner] Fallback plan: {plan}")
    return plan


# =============================================================================
# Public API
# =============================================================================

def plan_query(query: str, conversation_context: Optional[str] = None) -> QueryPlan:
    """
    Plan how to answer a user query using a fast LLM call.
    
    When conversation_context is provided (e.g. previous user/assistant turns), the
    planner can output intent=follow_up and a search_query combining the conversation's
    product with the current question (e.g. "SYS-521GE-TNRT GPU support").
    
    Returns a QueryPlan with intent, structured filters, and retrieval strategy.
    Falls back to heuristics if the LLM call fails.
    """
    t0 = time.time()

    raw = _call_planner_llm(query, conversation_context)

    if raw:
        plan = _parse_plan(raw, query)
        elapsed_ms = int((time.time() - t0) * 1000)
        print(f"[QueryPlanner] LLM plan ({elapsed_ms}ms): {plan}")
    else:
        plan = _fallback_plan(query)

    return plan


# =============================================================================
# CLI test
# =============================================================================

if __name__ == "__main__":
    import sys
    
    queries = sys.argv[1:] or [
        "golden skus",
        "global skus",
        "give me a list of global skus",
        "list all 1U servers",
        "GPU systems",
        "what GPU servers support NVIDIA H100",
        "CloudDC products",
        "edge servers",
        "what is the power consumption of the 8U GPU server",
        "how do I configure IPMI",
        "what is DCSCM",
        "compare 1U and 2U servers",
        "tell me about SYS-521GE-TNRT",
        "which servers support more than 2TB of memory",
        "what is the Global SKU program",
    ]
    
    for query in queries:
        print(f"\n{'='*60}")
        print(f"  Query: '{query}'")
        plan = plan_query(query)
        print(f"  Plan:  {plan}")
