#!/usr/bin/env python3
"""
Main chatbot interface for Supermicro RAG system.
"""

import json
import os
import re
import argparse
from typing import Dict, List, Optional, Set
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv

# Support running as:
# - python -m src.chatbot   (package mode)
# - python src/chatbot.py   (script mode)
try:
    from src.query import RAGQueryProcessor
    from src.product_catalog import ProductCatalog
    from src.query_planner import plan_query, QueryPlan
except ImportError:
    from query import RAGQueryProcessor
    from product_catalog import ProductCatalog
    from query_planner import plan_query, QueryPlan


# Load environment variables
_repo_root_env = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(dotenv_path=_repo_root_env, override=False)


# =============================================================================
# SYSTEM PROMPT - Domain knowledge for Supermicro products
# =============================================================================
SYSTEM_MESSAGE = """You are a technical assistant specializing in Supermicro server and storage products.

## PRODUCT NAMING
- Servers: SYS-{series} (Intel) or AS-{series} (AMD), e.g. SYS-521GE-TNRT, AS-4125GS-TNRT
- Motherboards: X{gen}{chipset}-{features}, e.g. X13DEI-T
- Other: CSE- (chassis), PWS- (PSU), SBI- (blades), AOC- (add-on cards)
- Partial model queries (e.g. "521GE") → find full model numbers containing that string

## RESPONSE FORMAT
- Target 300-500 words. Use markdown tables for specs — they pack more info per word.
- For product questions include: form factor, CPU, GPU (if any), memory, storage, networking, use cases.
- When listing a product family, include ALL generations found in context, not just the latest.
- Synthesize information across multiple sources into one coherent answer.
- If no product matches ALL requested criteria, present the closest match(es) and clearly note which criteria they meet and which they don't. Never just say "no match found" when close alternatives exist.

## STRICT RULES
1. Never fabricate specific hardware numbers (DIMM slot counts, drive bays, GPU counts, PSU wattage, clock speeds, etc.). If a spec isn't in the provided documents, omit it rather than guessing.
2. Never invent specific part numbers (AOC cards, NIC models, cable SKUs) unless they appear in the provided documents.
3. You MAY supplement with general domain knowledge to provide context, explain concepts, or describe typical use cases — just don't fabricate specific specs or part numbers.
4. Do not speculate about WHY information is missing (e.g. "the datasheet wasn't fully extracted"). Just present what you have.
5. Do not list things you "need" or "would need" — focus on what you CAN answer.
6. Minimize "I don't have" statements. If you have partial info, lead with what you know. Only mention a gap if the user specifically asked for that detail.
7. Do not over-hedge ("I can't confirm without...", "treat as TBD"). Be direct.
8. Do not reference unrelated products from conversation history.
9. **Pricing**: Prices in the context are approximate and may not reflect current eStore pricing. When mentioning a price, always note it is approximate (e.g. "starting at approximately $X,XXX") and direct the user to the Supermicro eStore for current pricing.

## TONE
- Never say "based on the retrieved context", "according to my database", "the retrieved documents show", or similar phrases that expose the system internals. Just state the information directly and confidently.
- When you HAVE the info: state it as fact. E.g. "The SYS-421GE-TNRT supports up to 10 GPUs."
- When you DON'T have the info: say it naturally. E.g. "I don't have detailed specs for the SYS-221GE-NR — visit supermicro.com for the full datasheet."
- Do NOT list sources or filenames in your response. The UI displays sources separately.

## DOMAIN KNOWLEDGE

### NVIDIA GPU → Supermicro System Mapping
GPU names (H100, H200, B200) are NVIDIA's names, not Supermicro's. Datasheets use Supermicro model numbers. When context contains these systems and the user asks about the GPU, present them as the answer:
- H100 → SYS-421GE-TNRT (4U PCIe), SYS-421GE-TNHR (4U HGX), SYS-521GE-TNRT (5U)
- H200 → SYS-821GE-TNHR (8U)
- B200 air-cooled → SuperCluster 10U Air Cooled
- B200 liquid-cooled → SuperCluster 4U Liquid Cooled, SYS-421GE-NBRT-LCC

### Gold Series vs Global SKU Program
- **Gold Series** ("Quick Ship") = pre-configured products with -G1/-G2 suffix. NOT the same as Global SKUs.
- **Global SKU Program** = logistics program. Link: https://www.supermicro.com/en/products/SMC_Global_skus — only mention when user specifically asks about "global SKUs".

## FAQ / eSTORE QUESTIONS
When answering eStore FAQ questions (ordering, shipping, returns, payments, account, warranty, tax, software licensing):
- Be concise and direct. The answer is typically contained in the FAQ context provided.
- Use a helpful customer-service tone rather than technical spec presentation.
- Do not use markdown tables — plain text or short bullet lists are preferred.
- If the FAQ provides a specific process (step-by-step), present it clearly.
- If the user's question isn't covered by the FAQ context, suggest contacting Supermicro support via live chat or email."""


# ---------------------------------------------------------------------------
# Token usage tracking (accumulated across calls within a process)
# ---------------------------------------------------------------------------
_llm_usage = defaultdict(int)

# ---------------------------------------------------------------------------
# Cached LLM clients (avoid re-instantiating on every call)
# ---------------------------------------------------------------------------
_openai_client = None
_anthropic_client = None


def get_llm_usage() -> dict:
    """Return accumulated main-LLM token usage since last reset."""
    return dict(_llm_usage)


def reset_llm_usage():
    _llm_usage.clear()


def get_llm_response(prompt: str, model: str = "gpt-5.2", provider: str = "openai",
                     temperature: float = 0.5, top_p: float = 1.0) -> str:
    """
    Get response from LLM.
    
    Args:
        prompt: Full prompt including system message, context, and question
        model: Model name
        provider: LLM provider (openai, ollama)
        temperature: Sampling temperature
        top_p: Nucleus sampling threshold (1.0 = no filtering)
        
    Returns:
        LLM response text
    """
    if provider == "openai":
        try:
            global _openai_client
            from openai import OpenAI
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                return (
                    "Missing OPENAI_API_KEY. Create a `.env` file in the repo root and set:\n"
                    "  OPENAI_API_KEY=sk-...\n"
                    "Then re-run the chatbot (or set LLM_PROVIDER=ollama to avoid OpenAI)."
                )

            if _openai_client is None:
                _openai_client = OpenAI(api_key=api_key)
            client = _openai_client
            
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_MESSAGE},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                top_p=top_p,
            )
            if hasattr(response, "usage") and response.usage:
                _llm_usage["input_tokens"] += response.usage.prompt_tokens
                _llm_usage["output_tokens"] += response.usage.completion_tokens
                _llm_usage["calls"] += 1
            return response.choices[0].message.content
        
        except Exception as e:
            return f"Error calling OpenAI API: {e}"
    
    elif provider == "anthropic":
        try:
            global _anthropic_client
            from anthropic import Anthropic
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                return (
                    "Missing ANTHROPIC_API_KEY. Create a `.env` file in the repo root and set:\n"
                    "  ANTHROPIC_API_KEY=sk-ant-...\n"
                    "Then re-run the chatbot."
                )
            
            if _anthropic_client is None:
                _anthropic_client = Anthropic(api_key=api_key)
            client = _anthropic_client
            anthropic_model = model if "claude" in model.lower() else os.getenv("ANTHROPIC_MODEL", "claude-opus-4-5")
            
            kwargs = dict(
                model=anthropic_model,
                max_tokens=2048,
                system=[{
                    "type": "text",
                    "text": SYSTEM_MESSAGE,
                    "cache_control": {"type": "ephemeral"},
                }],
                messages=[{"role": "user", "content": prompt}],
            )
            if top_p < 1.0:
                kwargs["top_p"] = top_p
            else:
                kwargs["temperature"] = temperature
            response = client.messages.create(**kwargs)
            if hasattr(response, "usage") and response.usage:
                _llm_usage["input_tokens"] += response.usage.input_tokens
                _llm_usage["output_tokens"] += response.usage.output_tokens
                _llm_usage["cache_read"] += getattr(response.usage, "cache_read_input_tokens", 0)
                _llm_usage["cache_creation"] += getattr(response.usage, "cache_creation_input_tokens", 0)
                _llm_usage["calls"] += 1
            return response.content[0].text
        
        except Exception as e:
            return f"Error calling Anthropic API: {e}"
    
    elif provider == "ollama":
        try:
            import requests
            base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            ollama_model = os.getenv("OLLAMA_MODEL", "llama3")
            
            response = requests.post(
                f"{base_url}/api/chat",
                json={
                    "model": ollama_model,
                    "messages": [
                        {"role": "system", "content": SYSTEM_MESSAGE},
                        {"role": "user", "content": prompt}
                    ],
                    "stream": False,
                    "options": {"temperature": temperature, "top_p": top_p}
                }
            )
            response.raise_for_status()
            return response.json()["message"]["content"]
        
        except Exception as e:
            return f"Error calling Ollama API: {e}"
    
    else:
        return f"Unknown LLM provider: {provider}"


def get_llm_response_stream(prompt: str, model: str = "gpt-5.2", provider: str = "openai",
                            temperature: float = 0.5, top_p: float = 1.0):
    """Yield text chunks from the LLM as they arrive (streaming)."""
    if provider == "openai":
        global _openai_client
        from openai import OpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            yield "Missing OPENAI_API_KEY."
            return
        if _openai_client is None:
            _openai_client = OpenAI(api_key=api_key)
        stream = _openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            top_p=top_p,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                yield delta.content

    elif provider == "anthropic":
        global _anthropic_client
        from anthropic import Anthropic
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            yield "Missing ANTHROPIC_API_KEY."
            return
        if _anthropic_client is None:
            _anthropic_client = Anthropic(api_key=api_key)
        anthropic_model = model if "claude" in model.lower() else os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5")
        kwargs = dict(
            model=anthropic_model,
            max_tokens=2048,
            system=[{
                "type": "text",
                "text": SYSTEM_MESSAGE,
                "cache_control": {"type": "ephemeral"},
            }],
            messages=[{"role": "user", "content": prompt}],
        )
        if top_p < 1.0:
            kwargs["top_p"] = top_p
        else:
            kwargs["temperature"] = temperature
        with _anthropic_client.messages.stream(**kwargs) as stream:
            for text in stream.text_stream:
                yield text
            response = stream.get_final_message()
            if hasattr(response, "usage") and response.usage:
                _llm_usage["input_tokens"] += response.usage.input_tokens
                _llm_usage["output_tokens"] += response.usage.output_tokens
                _llm_usage["cache_read"] += getattr(response.usage, "cache_read_input_tokens", 0)
                _llm_usage["cache_creation"] += getattr(response.usage, "cache_creation_input_tokens", 0)
                _llm_usage["calls"] += 1

    elif provider == "ollama":
        yield from [get_llm_response(prompt, model, provider, temperature, top_p)]

    else:
        yield f"Unknown LLM provider: {provider}"


class SupermicroChatbot:
    """Main chatbot class."""
    
    def __init__(
        self,
        index_dir: str = "embeddings/faiss_index/",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        llm_model: str = "gpt-3.5-turbo",
        llm_provider: str = "openai",
        top_k: int = 10,
        temperature: float = 0.5,
        top_p: float = 1.0,
    ):
        """
        Initialize the chatbot.
        
        Args:
            index_dir: Directory containing FAISS index
            embedding_model: Sentence transformer model name
            llm_model: LLM model name
            llm_provider: LLM provider (openai, ollama)
            top_k: Number of chunks to retrieve
            temperature: LLM sampling temperature (0.0 = deterministic, 1.0 = creative)
            top_p: LLM nucleus sampling threshold (1.0 = no filtering)
        """
        self.query_processor = RAGQueryProcessor(index_dir, embedding_model)
        self.llm_model = llm_model
        self.llm_provider = llm_provider
        self.top_k = top_k
        self.temperature = temperature
        self.top_p = top_p

        # Structured product catalog for listing/enumeration queries
        self.catalog = ProductCatalog()

        # Entity-relationship graph for multi-hop retrieval
        self.entity_graph = self._load_entity_graph(index_dir)
    
    @staticmethod
    def _load_entity_graph(index_dir: str) -> Dict:
        graph_path = os.path.join(index_dir, "entity_graph.json")
        if os.path.exists(graph_path):
            with open(graph_path) as f:
                graph = json.load(f)
            print(f"[EntityGraph] Loaded graph: {len(graph)} entities")
            return graph
        print("[EntityGraph] No entity_graph.json found — graph expansion disabled")
        return {}

    _GRAPH_TRAVERSAL_RELATIONS = frozenset({
        "uses_chassis", "used_by_system",
        "uses_motherboard", "used_by_system",
        "has_part", "part_of",
        "has_standard_part", "standard_part_of",
        "compatible_with",
        "belongs_to_family", "has_member",
        "requires", "required_by",
        "included_with", "includes",
        "validated_for", "validated_by",
    })

    _ACCESSORY_TYPE_KEYWORDS = {
        "rail": "MCP-290",
        "rail kit": "MCP-290",
        "bezel": "MCP-210",
        "io shield": "MCP-260",
        "i/o shield": "MCP-260",
        "bracket": "MCP-230",
        "add-on card": "AOC-",
        "riser": "RSC-",
        "backplane": "BPN-",
        "power supply": "PWS-",
        "psu": "PWS-",
        "fan": "FAN-",
        "cable": "CBL-",
        "heatsink": "SNK-",
    }

    def _expand_via_graph(
        self,
        query_entities: List[str],
        existing_chunks: list,
        max_hops: int = 2,
        max_extra_chunks: int = 6,
        query_text: str = "",
    ) -> list:
        """Follow entity-graph edges to pull in related chunks not yet retrieved.

        Uses a family-aware BFS: system -> chassis -> chassis_family -> parts.
        When query_text mentions a specific accessory type (e.g. "rail kit"),
        entities matching that type are prioritized.

        Returns additional chunks (does NOT include the existing ones).
        """
        if not self.entity_graph or not query_entities:
            return []

        # Detect accessory type hint from query text
        type_prefix = None
        if query_text:
            q_lower = query_text.lower()
            for kw, prefix in self._ACCESSORY_TYPE_KEYWORDS.items():
                if kw in q_lower:
                    type_prefix = prefix
                    break

        # BFS up to max_hops (only following structural edges)
        visited: Set[str] = set()
        frontier: Set[str] = set()
        for entity in query_entities:
            key = entity.upper()
            if key in self.entity_graph:
                frontier.add(key)

        related_entities: List[str] = []

        for _hop in range(max_hops):
            next_frontier: Set[str] = set()
            for node in frontier:
                if node in visited:
                    continue
                visited.add(node)
                node_data = self.entity_graph.get(node, {})
                for edge in node_data.get("edges", []):
                    relation = edge.get("relation", "")
                    if relation not in self._GRAPH_TRAVERSAL_RELATIONS:
                        continue
                    target = edge["target"]
                    if target not in visited:
                        next_frontier.add(target)
                        related_entities.append(target)
            frontier = next_frontier

        if not related_entities:
            return []

        print(f"[EntityGraph] Graph expansion: {query_entities} → {len(related_entities)} related entities")

        extra_chunks = []
        seen_ids = {c.get("chunk_id") for c in existing_chunks}

        # Step 1: Fetch chassis BOM pages using the chassis family prefix.
        # A single BOM chunk lists ALL standard parts for a chassis family,
        # so it's far more efficient than fetching individual part pages.
        seen_families: Set[str] = set()
        for entity in query_entities:
            key = entity.upper()
            node_data = self.entity_graph.get(key, {})
            for edge in node_data.get("edges", []):
                if edge.get("relation") == "uses_chassis":
                    chassis = edge["target"]
                    chassis_node = self.entity_graph.get(chassis, {})
                    for ce in chassis_node.get("edges", []):
                        if ce.get("relation") == "belongs_to_family":
                            family = ce["target"]
                            if family not in seen_families:
                                seen_families.add(family)
                                prefix = family.replace("SC", "CSE-")
                                if type_prefix:
                                    query_str = f"{type_prefix} standard parts list"
                                else:
                                    query_str = "standard parts list optional parts"
                                retrieved = self.query_processor.retrieve(
                                    query_str, top_k=8, max_per_source=3,
                                    source_filter=prefix)
                                added = 0
                                for rc in retrieved:
                                    cid = rc.get("chunk_id")
                                    if cid not in seen_ids and added < 4:
                                        seen_ids.add(cid)
                                        rc["_graph_expanded"] = True
                                        rc["_bom_chunk"] = True
                                        extra_chunks.append(rc)
                                        added += 1
                                if added:
                                    print(f"[EntityGraph]   +{added} BOM chunks for family '{family}' (query: '{query_str}')")

        # Step 2: Backfill with individual accessory pages, prioritizing
        # by query-relevant type (e.g., MCP-290 for "rail kit" queries).
        def _accessory_sort_key(entity: str) -> int:
            e = entity.upper()
            if type_prefix and e.startswith(type_prefix.upper()):
                return -1
            if e.startswith("MCP-"):
                return 0
            if e.startswith(("AOC-", "RSC-", "BPN-")):
                return 1
            if e.startswith(("PWS-", "FAN-", "CBL-", "SNK-")):
                return 2
            if e.startswith("CSE-"):
                return 3
            return 4

        bom_found = any(c.get("_bom_chunk") for c in extra_chunks)
        if bom_found and type_prefix:
            accessories_first = sorted(
                [e for e in related_entities if e.upper().startswith(type_prefix.upper())],
                key=_accessory_sort_key)
        else:
            accessories_first = sorted(related_entities, key=_accessory_sort_key)

        for entity in accessories_first:
            if len(extra_chunks) >= max_extra_chunks:
                break

            node_data = self.entity_graph.get(entity, {})
            chunk_ids = node_data.get("chunk_ids", [])

            already_have = any(cid in seen_ids for cid in chunk_ids)
            if already_have:
                continue

            if node_data.get("type") in ("chassis_family", "chassis"):
                continue

            retrieved = self.query_processor.retrieve(entity, top_k=3, max_per_source=2)
            added = 0
            for rc in retrieved:
                cid = rc.get("chunk_id")
                if cid not in seen_ids and added < 2:
                    seen_ids.add(cid)
                    rc["_graph_expanded"] = True
                    extra_chunks.append(rc)
                    added += 1

            if added:
                print(f"[EntityGraph]   +{added} chunks for related entity '{entity}'")

        return extra_chunks[:max_extra_chunks]

    def answer(self, question: str, conversation_context: str = "") -> dict:
        """Answer a question using RAG (non-streaming)."""
        ctx = self._retrieve_context(question, conversation_context)

        if ctx["prompt"] is None:
            return {
                "answer": "No relevant information found in the documentation.",
                "sources": ctx["sources"],
                "chunks": ctx["chunks"],
                "plan": ctx["plan"],
                "search_queries": ctx["search_queries"],
                "rag_top_k": ctx["rag_top_k"],
                "max_per_source": ctx["max_per_source"],
            }

        answer = get_llm_response(
            ctx["prompt"], self.llm_model, self.llm_provider,
            self.temperature, self.top_p,
        )

        return {
            "answer": answer,
            "sources": ctx["sources"],
            "chunks": ctx["chunks"],
            "plan": ctx["plan"],
            "search_queries": ctx["search_queries"],
            "rag_top_k": ctx["rag_top_k"],
            "max_per_source": ctx["max_per_source"],
        }
    
    def _retrieve_context(self, question: str, conversation_context: str = "") -> dict:
        """Shared retrieval logic used by both answer() and answer_stream().

        Returns dict with keys: prompt, sources, chunks, plan, search_queries,
        rag_top_k, max_per_source.
        """
        # =====================================================================
        # FOLLOW-UP DETECTION
        # =====================================================================
        retrieval_query = question
        is_followup = False

        if conversation_context:
            q_clean = re.sub(r'[?!.,;:]+$', '', question.strip()).lower().strip()

            AFFIRMATIVE = frozenset({
                'yes', 'yeah', 'yep', 'sure', 'please', 'go ahead',
                'ok', 'okay', 'correct', 'absolutely', 'yea',
            })
            if q_clean in AFFIRMATIVE:
                last_assistant = ""
                for line in reversed(conversation_context.strip().split('\n')):
                    if line.startswith('Assistant:') or line.startswith('assistant:'):
                        last_assistant = line.split(':', 1)[1].strip()
                        break
                if last_assistant:
                    retrieval_query = last_assistant[:400]
                    is_followup = True
                    print(f"[DEBUG] Affirmative continuation → using last assistant msg ({len(retrieval_query)} chars)")

            elif (
                re.findall(r'\b(?:SYS|AS|SSG|SBI|AOC)-[\w-]+\b', question, re.IGNORECASE)
                or re.match(r'^x\d{2}[a-z0-9-]*$', q_clean)
                or re.match(r'^\d{3}[a-z]{2,}(?:-[a-z0-9]+)?$', q_clean)
            ):
                print(f"[DEBUG] Product code detected in '{q_clean}' → NEW query (context suppressed)")

            else:
                REFERENTIAL = [
                    r'\bit\b', r'\bits\b', r'\bthis\b',
                    r'\bthe same\b', r'\bthat one\b', r'\bthis one\b',
                    r'\bthat system\b', r'\bthat server\b', r'\bthat model\b',
                    r'\bthat product\b', r'\bthose servers\b', r'\bthose systems\b',
                    r'\bthe above\b', r'\bmentioned\b', r'\babove\b',
                ]
                CONTINUATION = [
                    r'\btell me more\b', r'\bmore details\b', r'\bwhat else\b',
                    r'\banything else\b', r'\bgo on\b', r'\belaborate\b',
                    r'\bexplain more\b', r'\bmore about\b', r'\bmore info\b',
                ]
                has_referential = any(re.search(p, q_clean) for p in REFERENTIAL)
                has_continuation = any(re.search(p, q_clean) for p in CONTINUATION)

                # Queries with multiple hardware specs are new product discovery,
                # not follow-ups, even if they contain words like "that" as
                # relative pronouns (e.g., "system that supports 12 drive bays")
                _SPEC_SIGNALS = [
                    r'\b[124]u\b',
                    r'\bdual\s+processor\b', r'\bsingle\s+processor\b',
                    r'\bepyc\b', r'\bxeon\b', r'\bamd\b', r'\bintel\b',
                    r'\b\d+\s*[\d.]*["\u201d]?\s*drive\b',
                    r'\b\d+\s*bay', r'\bnvme\b', r'\bsata\b', r'\bsas\b',
                    r'\bgpu\b', r'\b\d+\s*dimm\b',
                ]
                spec_count = sum(1 for p in _SPEC_SIGNALS if re.search(p, q_clean))
                is_product_discovery = spec_count >= 2

                if is_product_discovery:
                    print(f"[DEBUG] Product discovery query ({spec_count} spec signals) → NEW query (context suppressed)")
                elif has_referential or has_continuation:
                    is_followup = True
                    all_codes = re.findall(
                        r'\b(?:SYS|AS|SSG|SBI|AOC)-[\w-]+\b',
                        conversation_context, re.IGNORECASE,
                    )
                    product_code = None
                    seen = set()
                    for c in all_codes:
                        u = c.upper()
                        if u not in seen:
                            seen.add(u)
                            product_code = c
                    if product_code:
                        retrieval_query = f"{product_code} {question}"
                    print(f"[DEBUG] Follow-up confirmed (referential language) → query: {retrieval_query}")
                else:
                    print(f"[DEBUG] No follow-up signals → treating as NEW query (context suppressed)")

        effective_conversation = conversation_context if is_followup else None

        plan = plan_query(retrieval_query, conversation_context=effective_conversation)

        # --- Catalog retrieval ---
        catalog_results = []
        if plan.use_catalog and self.catalog.products:
            catalog_results = self.catalog.filter_structured(
                form_factor=plan.form_factor,
                tags=plan.tags if plan.tags else None,
                keywords=plan.keywords if plan.keywords else None,
            )
            if not catalog_results and (plan.tags or plan.keywords):
                catalog_results = self.catalog.search(retrieval_query)
                if catalog_results:
                    print(f"[DEBUG] Structured filter empty, keyword fallback: {len(catalog_results)} results")

        max_per_source = None
        source_filter = None

        if plan.intent == "list" and catalog_results:
            rag_top_k = max(self.top_k, 10)
            catalog_max = 30
            max_per_source = 2
            print(f"[DEBUG] Plan: list → catalog: {len(catalog_results)} products, RAG: top {rag_top_k}, max_per_source={max_per_source}")
        elif plan.intent == "list" and not catalog_results:
            rag_top_k = max(self.top_k, 15)
            catalog_max = 0
            max_per_source = 2
            print(f"[DEBUG] Plan: list (no catalog) → RAG only: top {rag_top_k}, max_per_source={max_per_source}")
        elif plan.intent == "follow_up":
            rag_top_k = self.top_k
            catalog_max = min(len(catalog_results), 5) if catalog_results else 0
            print(f"[DEBUG] Plan: follow_up → catalog: {len(catalog_results)} (showing {catalog_max}), RAG: top {rag_top_k}")
        elif plan.intent == "detail":
            rag_top_k = self.top_k
            catalog_max = min(len(catalog_results), 5) if catalog_results else 0
            print(f"[DEBUG] Plan: detail → catalog: {len(catalog_results)} (showing {catalog_max}), RAG: top {rag_top_k}")
        elif plan.intent == "compare":
            rag_top_k = max(int(self.top_k * 1.5), 15)
            catalog_max = min(len(catalog_results), 10) if catalog_results else 0
            max_per_source = 3
            print(f"[DEBUG] Plan: compare → catalog: {catalog_max}, RAG: top {rag_top_k}, max_per_source={max_per_source}")
        elif plan.intent == "faq":
            rag_top_k = 5
            catalog_max = 0
            catalog_results = []
            print(f"[DEBUG] Plan: faq → FAQ question bank: top {rag_top_k} (no catalog, no graph)")
        else:
            rag_top_k = self.top_k
            catalog_max = 0
            max_per_source = 3
            print(f"[DEBUG] Plan: {plan.intent} → RAG only: top {rag_top_k}, max_per_source={max_per_source}")

        search_queries = plan.search_queries if plan.search_queries else [retrieval_query]

        if len(plan.product_codes) >= 2 and len(search_queries) == 1:
            search_queries = [f"{code} specifications datasheet" for code in plan.product_codes]
            print(f"[DEBUG] Auto-split: planner gave 1 query but {len(plan.product_codes)} codes → {len(search_queries)} queries")

        if len(search_queries) == 1 and plan.product_codes:
            for code in plan.product_codes:
                if code.upper() not in search_queries[0].upper():
                    search_queries[0] = f"{search_queries[0]} {code}"
                    print(f"[DEBUG] Injected '{code}' into search query")

        rag_query = search_queries[0] if search_queries else retrieval_query

        # FAQ intent: combine question bank + source-filtered hybrid search
        if plan.intent == "faq":
            # Pass 1: question-to-question matching (precise title similarity)
            qbank_chunks = self.query_processor.retrieve_faq(retrieval_query, rag_top_k)

            # Pass 2: source-filtered hybrid search (BM25 keyword coverage)
            hybrid_chunks = self.query_processor.retrieve(
                rag_query, rag_top_k, source_filter="FAQ:")

            # Merge: question bank first, then backfill from hybrid
            chunks = list(qbank_chunks)
            seen_ids = {c.get("chunk_id") for c in chunks}
            hybrid_added = 0
            for hc in hybrid_chunks:
                if hc.get("chunk_id") not in seen_ids:
                    seen_ids.add(hc.get("chunk_id"))
                    chunks.append(hc)
                    hybrid_added += 1
            if hybrid_added:
                print(f"[DEBUG] FAQ merged: {len(qbank_chunks)} qbank + {hybrid_added} hybrid backfill")
            faq_combined_limit = rag_top_k + min(hybrid_added, rag_top_k)
            chunks = chunks[:faq_combined_limit]

            # Supplement with a couple of general-corpus chunks for context
            if chunks:
                faq_count = len(chunks)
                supplement_slots = 2
                seen_ids = {c.get("chunk_id") for c in chunks}
                supplement = self.query_processor.retrieve(retrieval_query, 5, max_per_source=1)
                added = 0
                for sc in supplement:
                    if sc.get("chunk_id") not in seen_ids and "FAQ:" not in sc.get("source_file", ""):
                        chunks.append(sc)
                        added += 1
                        if added >= supplement_slots:
                            break
                if added:
                    print(f"[DEBUG] FAQ supplement: {faq_count} FAQ + {added} general context chunks")

        elif len(search_queries) >= 2 and plan.use_rag:
            per_k = max(5, rag_top_k // len(search_queries))
            rag_top_k = max(rag_top_k, len(search_queries) * per_k)
            per_query_chunks = {}
            with ThreadPoolExecutor(max_workers=len(search_queries)) as pool:
                futures = {sq: pool.submit(self.query_processor.retrieve, sq, per_k,
                                           source_filter=source_filter) for sq in search_queries}
                for sq, fut in futures.items():
                    per_query_chunks[sq] = fut.result()
                    print(f"[DEBUG]   '{sq[:60]}' → {len(per_query_chunks[sq])} chunks")

            chunks = []
            seen_ids = set()
            max_rounds = max((len(v) for v in per_query_chunks.values()), default=0)
            for round_i in range(max_rounds):
                for sq in search_queries:
                    sq_chunks = per_query_chunks[sq]
                    if round_i < len(sq_chunks):
                        chunk = sq_chunks[round_i]
                        cid = chunk.get("chunk_id")
                        if cid not in seen_ids:
                            seen_ids.add(cid)
                            chunks.append(chunk)

            chunks = chunks[:rag_top_k]
            print(f"[DEBUG] Split retrieval: {len(search_queries)} queries, {len(chunks)} chunks (balanced)")
        else:
            chunks = self.query_processor.retrieve(rag_query, rag_top_k, max_per_source=max_per_source,
                                                   source_filter=source_filter) if plan.use_rag else []

        if chunks and plan.product_codes and plan.use_rag:
            chunk_sources = " ".join(c.get("source_file", "") for c in chunks).upper()
            missing_codes = []
            for code in plan.product_codes:
                code_stem = code.upper().replace("SYS-", "").replace("AS-", "").replace("SSG-", "")
                if code_stem and code_stem not in chunk_sources:
                    missing_codes.append(code)

            if missing_codes:
                print(f"[DEBUG] Product safety net: {missing_codes} not in retrieved sources, doing focused retrieval")
                rescue_slots = max(3, rag_top_k // 3)
                seen_ids = {c.get("chunk_id") for c in chunks}
                with ThreadPoolExecutor(max_workers=len(missing_codes)) as pool:
                    futures = {code: pool.submit(self.query_processor.retrieve, code, rescue_slots) for code in missing_codes}
                    for code, fut in futures.items():
                        rescue_chunks = fut.result()
                        added = 0
                        for rc in rescue_chunks:
                            cid = rc.get("chunk_id")
                            if cid not in seen_ids:
                                seen_ids.add(cid)
                                chunks.append(rc)
                                added += 1
                        if added:
                            print(f"[DEBUG]   Rescued {added} chunks for '{code}'")
                chunks = chunks[:rag_top_k + len(missing_codes) * rescue_slots]

        _mb_topic = (
            "motherboard" in question.lower()
            or (plan.intent == "detail" and any("X1" in c.upper() or "MBD-" in c.upper() for c in plan.product_codes))
        )
        if chunks and _mb_topic:
            non_mb = re.compile(
                r"Security_Bulletin|AOC-\w|PWS-\d|quick_spec.*PWS", re.IGNORECASE
            )
            filtered = [c for c in chunks if not non_mb.search(c.get("source_file", ""))]
            if len(filtered) >= 3:
                chunks = filtered[:rag_top_k]
                print(f"[DEBUG] Motherboard query: filtered to {len(chunks)} chunks (dropped non-motherboard sources)")

        # --- Graph expansion for multi-hop retrieval ---
        # 3 hops needed for: system → chassis → family → accessory
        graph_chunks = []
        if self.entity_graph and plan.accessory_query and plan.product_codes:
            graph_chunks = self._expand_via_graph(
                plan.product_codes, chunks,
                max_hops=3, max_extra_chunks=10,
                query_text=question,
            )
        elif self.entity_graph and plan.product_codes:
            _accessory_kw = re.compile(
                r"rail\s*kit|part\s*number|accessory|cable|psu|power\s*supply|compatible|add-on\s*card|aoc-|mcp-|fan\s*module",
                re.IGNORECASE,
            )
            if _accessory_kw.search(question):
                graph_chunks = self._expand_via_graph(
                    plan.product_codes, chunks,
                    max_hops=3, max_extra_chunks=10,
                    query_text=question,
                )

        # --- Build combined context ---
        context_parts = []
        if catalog_results and catalog_max > 0:
            catalog_context = self.catalog.format_for_llm(catalog_results, max_products=catalog_max)
            context_parts.append(f"PRODUCT CATALOG DATA:\n{catalog_context}")
        if chunks:
            rag_context = self.query_processor.format_context(chunks)
            label = "DOCUMENTATION CONTEXT:" if catalog_results and catalog_max > 0 else ""
            if label:
                context_parts.append(f"{label}\n{rag_context}")
            else:
                context_parts.append(rag_context)
        if graph_chunks:
            bom_chunks = [c for c in graph_chunks if c.get("_bom_chunk")]
            other_graph = [c for c in graph_chunks if not c.get("_bom_chunk")]
            if bom_chunks:
                bom_context = self.query_processor.format_context(bom_chunks)
                chassis_label = ""
                if plan.product_codes:
                    chassis_label = f" for {plan.product_codes[0]}'s chassis family"
                context_parts.append(
                    f"CHASSIS STANDARD & OPTIONAL PARTS LIST{chassis_label} "
                    f"(these are the specific parts/accessories/rail kits for this system):\n{bom_context}")
            if other_graph:
                other_context = self.query_processor.format_context(other_graph)
                context_parts.append(f"RELATED PRODUCT DATA (from linked documents):\n{other_context}")

        sources_from_catalog = catalog_results[:catalog_max] if catalog_max > 0 else []
        seen = set()
        sources = []
        if sources_from_catalog:
            sources.append(f"Product Catalog ({len(sources_from_catalog)} products)")
            seen.add("product_catalog")
        for chunk in chunks:
            src = chunk["source_file"]
            if src not in seen:
                seen.add(src)
                sources.append(src)
        for chunk in graph_chunks:
            src = chunk["source_file"]
            if src not in seen:
                seen.add(src)
                sources.append(src)

        prompt = None
        if context_parts:
            context = "\n\n".join(context_parts)
            prompt = self._build_user_prompt(question, context, chunks, effective_conversation, intent=plan.intent)

        return {
            "prompt": prompt,
            "sources": sources,
            "chunks": chunks,
            "plan": plan,
            "search_queries": search_queries,
            "rag_top_k": rag_top_k,
            "max_per_source": max_per_source,
        }

    def answer_stream(self, question: str, conversation_context: str = ""):
        """Yield (event, data) tuples for SSE streaming.

        Events: ("sources", json_list), ("token", text), ("done", "").
        """
        import json as _json
        ctx = self._retrieve_context(question, conversation_context)

        yield ("sources", _json.dumps(ctx["sources"]))

        if ctx["prompt"] is None:
            yield ("token", "No relevant information found in the documentation.")
            yield ("done", "")
            return

        for token in get_llm_response_stream(
            ctx["prompt"], self.llm_model, self.llm_provider,
            self.temperature, self.top_p,
        ):
            yield ("token", token)

        yield ("done", "")

    def _build_user_prompt(self, question: str, context: str, chunks: list,
                           conversation_context: str = "", intent: str = "general") -> str:
        """Build a structured user prompt for the LLM."""
        source_files = list(set(chunk["source_file"] for chunk in chunks))
        source_summary = ", ".join(source_files[:5])
        if len(source_files) > 5:
            source_summary += f" (+{len(source_files) - 5} more)"

        conversation_section = ""
        if conversation_context:
            conversation_section = f"""## CONVERSATION HISTORY
{conversation_context}
---

"""

        if intent == "faq":
            instructions = (
                "1. Answer the eStore question directly and concisely using the FAQ content provided.\n"
                "2. Use a helpful, customer-service tone. Keep the response short — no tables or lengthy spec lists.\n"
                "3. If the FAQ doesn't cover this specific question, suggest contacting Supermicro support via live chat or email."
            )
        else:
            instructions = (
                "1. Use the reference documents as your primary source. You may supplement with general domain knowledge for context, but never invent specific specs or part numbers.\n"
                "2. If this is a follow-up question, use conversation history for context."
            )

        prompt = f"""{conversation_section}## REFERENCE DOCUMENTS
Sources: {source_summary}

---
{context}
---

## USER QUESTION
{question}

## INSTRUCTIONS
{instructions}"""
        
        return prompt
    
    def interactive_mode(self):
        """Run interactive chat mode."""
        print("=" * 80)
        print("Supermicro RAG Chatbot")
        print("=" * 80)
        print("Ask questions about Supermicro products and documentation.")
        print("Type 'quit' or 'exit' to end the conversation.\n")
        
        while True:
            try:
                question = input("\nQuestion: ").strip()
                
                if question.lower() in ['quit', 'exit', 'q']:
                    print("\nGoodbye!")
                    break
                
                if not question:
                    continue
                
                print("\nSearching documentation...")
                result = self.answer(question)
                
                print("\n" + "=" * 80)
                print("Answer:")
                print("=" * 80)
                print(result["answer"])
                
                if result["sources"]:
                    print("\n" + "=" * 80)
                    print("Sources:")
                    print("=" * 80)
                    for source in result["sources"]:
                        print(f"  - {source}")
                
                print()
            
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"\nError: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Supermicro RAG Chatbot"
    )
    parser.add_argument(
        "--query",
        help="Single question to answer"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode"
    )
    parser.add_argument(
        "--index-dir",
        default="embeddings/faiss_index/",
        help="Directory containing FAISS index (default: embeddings/faiss_index/)"
    )
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Embedding model name (default: sentence-transformers/all-MiniLM-L6-v2)"
    )
    parser.add_argument(
        "--llm-model",
        default=None,
        help="LLM model name (default: from .env or gpt-5.2)"
    )
    parser.add_argument(
        "--llm-provider",
        default=None,
        help="LLM provider: openai or ollama (default: from .env or openai)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of chunks to retrieve (default: 10)"
    )
    
    args = parser.parse_args()
    
    # Get LLM settings from environment or args
    llm_model = args.llm_model or os.getenv("LLM_MODEL", "gpt-5.2")
    llm_provider = args.llm_provider or os.getenv("LLM_PROVIDER", "openai")
    temperature = float(os.getenv("LLM_TEMPERATURE", "0.5"))
    top_p = float(os.getenv("LLM_TOP_P", "1.0"))
    
    # Initialize chatbot
    chatbot = SupermicroChatbot(
        index_dir=args.index_dir,
        embedding_model=args.embedding_model,
        llm_model=llm_model,
        llm_provider=llm_provider,
        top_k=args.top_k,
        temperature=temperature,
        top_p=top_p,
    )
    
    if args.interactive:
        chatbot.interactive_mode()
    elif args.query:
        result = chatbot.answer(args.query)
        print("\n" + "=" * 80)
        print("Answer:")
        print("=" * 80)
        print(result["answer"])
        if result["sources"]:
            print("\n" + "=" * 80)
            print("Sources:")
            print("=" * 80)
            for source in result["sources"]:
                print(f"  - {source}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
