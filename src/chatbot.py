#!/usr/bin/env python3
"""
Main chatbot interface for Supermicro RAG system.
"""

import os
import re
import argparse
from typing import Optional
from pathlib import Path
from collections import defaultdict
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

## SUPERMICRO PRODUCT NAMING CONVENTIONS
- Server systems: SYS-{series}{form factor}{features}-{suffix} (e.g., SYS-521GE-TNRT, SYS-421GE-TNRT)
- AMD systems: AS-{series} (e.g., AS-4125GS-TNRT)
- Motherboards: X{generation}{chipset}-{features} (e.g., X13DEI-T, X12SPi-TF)
- Chassis: SC{series} or CSE-{series} (e.g., SC847, CSE-826)
- Power supplies: PWS-{wattage}{features} (e.g., PWS-1K28P-SQ)
- Blades: SBI-{series} (e.g., SBI-7428R-T3)
- Add-on cards: AOC-{type}-{features} (e.g., AOC-S3908L-H8IR)

## WHEN ANSWERING PRODUCT QUESTIONS
1. If asked about a partial model number (e.g., "521GE"), look for full model numbers containing that string
2. For product questions, provide key specifications when available:
   - Form factor (1U, 2U, 4U, etc.)
   - CPU support (Intel Xeon, AMD EPYC, etc.)
   - GPU support (if applicable)
   - Memory capacity and type
   - Storage options
   - Network connectivity
   - Target use cases
3. If multiple sources cover the same product, synthesize the information
4. When listing a product FAMILY (e.g., MicroCloud, Twin, BigTwin): include ALL generations and models found in the context — do NOT only show the latest generation. For example, MicroCloud includes both AMD H13 (AS-3015MR-*) and Intel Xeon E (SYS-530MT-*, SYS-5039MS-*) models. List every model found in your context, grouped by generation if helpful.

## RESPONSE GUIDELINES
- Aim for 200-350 words - detailed enough to be helpful, but not rambling
- Focus on what you CAN answer, not what you can't
- When using information from the provided context, cite the source briefly
- For comparisons, use tables

## CRITICAL: AVOID THESE BAD HABITS
- Do NOT list things you "need" or "would need" to answer better
- Do NOT say "the context doesn't include X" for multiple items - one brief mention is enough
- Do NOT write long explanations of what information is missing
- Do NOT over-hedge with phrases like "I can't confirm without...", "treat as TBD", etc.
- Do NOT reference unrelated products from conversation history

## WHEN DATA IS INCOMPLETE
If the exact product datasheet isn't available in the retrieved context:
1. State briefly: "I don't have the [product] datasheet in my database."
2. Provide what you DO know FROM THE RETRIEVED CONTEXT (related models, product family info).
3. Suggest the user check Supermicro's website for the full datasheet.
4. STOP - do not ramble about what's missing.

CRITICAL: Do NOT invent or guess specifications. If you don't have a spec in the retrieved context, do not fabricate it.
- Do NOT create "Expected Specifications" tables based on naming conventions or general knowledge.
- Do NOT guess CPU counts, DIMM slots, drive bays, GPU counts, or any other hardware specs.
- Do NOT decode model numbers into speculative specs (e.g. "7 = 7th generation, 2 = 2U" is speculation).
- It is OK to mention CONFIRMED specs from related models that ARE in the context, clearly labeled as such.
- It is OK to say a specific spec (like PSU wattage) is not in the retrieved data and suggest where to find it.

## NVIDIA GPU TO SUPERMICRO SYSTEM MAPPING
NVIDIA GPU names (H100, H200, B200) are NOT Supermicro product names. Supermicro's datasheets use their own model numbers. When the user asks about an NVIDIA GPU and the retrieved context contains the corresponding Supermicro systems, present those systems as the answer — do NOT say "I don't have an H100 datasheet" just because the filename doesn't say "H100".
- H100 PCIe/SXM → SYS-421GE-TNRT (4U PCIe GPU), SYS-421GE-TNHR (4U HGX), SYS-521GE-TNRT (5U)
- H200 SXM/HGX → SYS-821GE-TNHR (8U)
- B200 air-cooled → SuperCluster 10U Air Cooled
- B200 liquid-cooled → SuperCluster 4U Liquid Cooled, SYS-421GE-NBRT-LCC
If you see SYS-421GE in your context and the user asked about H100, that IS the H100 system — describe it as such.

## GLOBAL SKU PROGRAM vs GOLD SERIES (do NOT confuse these)
- **Global SKU Program** = a logistics/fulfillment program. Official list at: https://www.supermicro.com/en/products/SMC_Global_skus
  Only mention this URL when the user specifically asks about "global SKUs" or the "Global SKU program."
- **Gold Series** (also called "golden SKUs" or "Quick Ship") = pre-configured, ready-to-ship product SKUs with -G1/-G2 suffix. These are NOT the same as Global SKUs. Do NOT link to the Global SKU page when answering about Gold Series products."""


# ---------------------------------------------------------------------------
# Token usage tracking (accumulated across calls within a process)
# ---------------------------------------------------------------------------
_llm_usage = defaultdict(int)


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
            from openai import OpenAI
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                return (
                    "Missing OPENAI_API_KEY. Create a `.env` file in the repo root and set:\n"
                    "  OPENAI_API_KEY=sk-...\n"
                    "Then re-run the chatbot (or set LLM_PROVIDER=ollama to avoid OpenAI)."
                )

            client = OpenAI(api_key=api_key)
            
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
            from anthropic import Anthropic
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                return (
                    "Missing ANTHROPIC_API_KEY. Create a `.env` file in the repo root and set:\n"
                    "  ANTHROPIC_API_KEY=sk-ant-...\n"
                    "Then re-run the chatbot."
                )
            
            client = Anthropic(api_key=api_key)
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
        self.catalog = ProductCatalog()  # path from PRODUCTS_FILE env or default data/pages/products.jsonl
    
    def answer(self, question: str, conversation_context: str = "") -> dict:
        """
        Answer a question using RAG.
        
        Args:
            question: User question
            conversation_context: Previous conversation turns for context
            
        Returns:
            Dictionary with answer, sources, and retrieved chunks
        """
        # =====================================================================
        # FOLLOW-UP DETECTION — default: treat as NEW query (safe).
        # Only use conversation context when we are ABSOLUTELY SURE it's a
        # follow-up. A wrong follow-up pollutes retrieval with the old product.
        # =====================================================================
        retrieval_query = question
        is_followup = False  # stays False unless proven otherwise

        if conversation_context:
            q_clean = re.sub(r'[?!.,;:]+$', '', question.strip()).lower().strip()

            # --- STEP 1: Is this an affirmative continuation? ("yes", "sure") ---
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

            # --- STEP 2: Does it contain a product code? → ALWAYS a new query ---
            elif (
                re.findall(r'\b(?:SYS|AS|SSG|SBI|AOC)-[\w-]+\b', question, re.IGNORECASE)
                or re.match(r'^x\d{2}[a-z0-9-]*$', q_clean)
                or re.match(r'^\d{3}[a-z]{2,}(?:-[a-z0-9]+)?$', q_clean)
            ):
                # Any product identifier (full or partial) → brand new query, no conversation
                print(f"[DEBUG] Product code detected in '{q_clean}' → NEW query (context suppressed)")

            # --- STEP 3: Check for explicit referential language ---
            # ONLY these patterns trigger follow-up. Short/vague questions do NOT.
            # Word-boundary matching to avoid false positives ("it" in "items").
            else:
                REFERENTIAL = [
                    r'\bit\b', r'\bits\b', r'\bthis\b', r'\bthat\b', r'\bthose\b',
                    r'\bthe same\b', r'\bthat one\b', r'\bthis one\b',
                    r'\bthe above\b', r'\bmentioned\b', r'\babove\b',
                ]
                CONTINUATION = [
                    r'\btell me more\b', r'\bmore details\b', r'\bwhat else\b',
                    r'\banything else\b', r'\bgo on\b', r'\belaborate\b',
                    r'\bexplain more\b', r'\bmore about\b', r'\bmore info\b',
                ]
                has_referential = any(re.search(p, q_clean) for p in REFERENTIAL)
                has_continuation = any(re.search(p, q_clean) for p in CONTINUATION)

                if has_referential or has_continuation:
                    is_followup = True
                    # Inject the most recent product code from conversation
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
                            product_code = c  # last unique code = most recent
                    if product_code:
                        retrieval_query = f"{product_code} {question}"
                    print(f"[DEBUG] Follow-up confirmed (referential language) → query: {retrieval_query}")
                else:
                    print(f"[DEBUG] No follow-up signals → treating as NEW query (context suppressed)")

        # --- Decide what context the planner and LLM see ---
        # Planner: only sees conversation when it's a confirmed follow-up
        # LLM prompt: only sees conversation when it's a confirmed follow-up
        # This prevents ANY context pollution on new queries.
        effective_conversation = conversation_context if is_followup else None

        # --- Query Planning: LLM decides intent + retrieval strategy ---
        plan = plan_query(retrieval_query, conversation_context=effective_conversation)
        
        # --- Catalog retrieval (structured filtering) ---
        catalog_results = []
        if plan.use_catalog and self.catalog.products:
            # Use structured filters from the planner (exact field matching)
            catalog_results = self.catalog.filter_structured(
                form_factor=plan.form_factor,
                tags=plan.tags if plan.tags else None,
                keywords=plan.keywords if plan.keywords else None,
            )
            # If structured filters returned nothing, fall back to keyword search
            if not catalog_results and (plan.tags or plan.keywords):
                catalog_results = self.catalog.search(retrieval_query)
                if catalog_results:
                    print(f"[DEBUG] Structured filter empty, keyword fallback: {len(catalog_results)} results")
        
        # --- Determine catalog/RAG balance and source diversity based on intent ---
        max_per_source = None  # None = no cap (detail/follow-up want deep single-source context)

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
        else:
            rag_top_k = self.top_k
            catalog_max = 0
            max_per_source = 3
            print(f"[DEBUG] Plan: {plan.intent} → RAG only: top {rag_top_k}, max_per_source={max_per_source}")
        
        # --- RAG retrieval ---
        # The planner outputs search_queries (list). If it returns multiple,
        # we split retrieval. If it returns one but there are multiple product
        # codes, we auto-split on codes as a safety net.
        search_queries = plan.search_queries if plan.search_queries else [retrieval_query]

        # Safety net: if planner returned 2+ product codes but only 1 search query,
        # auto-split into per-product queries so each product gets its own retrieval.
        if len(plan.product_codes) >= 2 and len(search_queries) == 1:
            search_queries = [f"{code} specifications datasheet" for code in plan.product_codes]
            print(f"[DEBUG] Auto-split: planner gave 1 query but {len(plan.product_codes)} codes → {len(search_queries)} queries")

        # Inject product codes only for SINGLE queries (safety net).
        # For split queries, each already targets a specific product — don't cross-contaminate.
        if len(search_queries) == 1 and plan.product_codes:
            for code in plan.product_codes:
                if code.upper() not in search_queries[0].upper():
                    search_queries[0] = f"{search_queries[0]} {code}"
                    print(f"[DEBUG] Injected '{code}' into search query")

        rag_query = search_queries[0] if search_queries else retrieval_query

        # --- Split vs single retrieval (decided by planner, not hardcoded rules) ---
        if len(search_queries) >= 2 and plan.use_rag:
            per_k = max(5, rag_top_k // len(search_queries))
            rag_top_k = max(rag_top_k, len(search_queries) * per_k)
            per_query_chunks = {}
            for sq in search_queries:
                per_query_chunks[sq] = self.query_processor.retrieve(sq, per_k)
                print(f"[DEBUG]   '{sq[:60]}' → {len(per_query_chunks[sq])} chunks")

            # Round-robin interleave so each topic gets fair representation
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
            chunks = self.query_processor.retrieve(rag_query, rag_top_k, max_per_source=max_per_source) if plan.use_rag else []

        # --- Product code safety net ---
        # When the planner identified specific product codes, verify that we actually
        # retrieved chunks from those products. If a product's documents are missing,
        # do a focused retrieval using just the product code to pull in its datasheet.
        # This prevents topic-heavy queries like "does X support GPUs?" from drowning
        # out the product's own documentation with unrelated GPU server docs.
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
                for code in missing_codes:
                    rescue_chunks = self.query_processor.retrieve(code, rescue_slots)
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

        # When query is specifically about motherboard PRODUCTS, drop chunks from clearly non-motherboard sources.
        # Only trigger when "motherboard" is the topic (e.g., "X13DEI motherboard"), not when it
        # appears incidentally (e.g., "Global SKU Program list Systems Motherboards Chassis").
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
        
        if not context_parts:
            return {
                "answer": "No relevant information found in the documentation.",
                "sources": [],
                "chunks": [],
                "plan": plan,
                "search_queries": search_queries,
                "rag_top_k": rag_top_k,
                "max_per_source": max_per_source,
            }
        
        context = "\n\n".join(context_parts)
        sources_from_catalog = catalog_results[:catalog_max] if catalog_max > 0 else []
        if sources_from_catalog and chunks:
            print(f"[DEBUG] Context: BOTH catalog ({len(sources_from_catalog)} products) and RAG ({len(chunks)} chunks) sent to LLM")
        elif sources_from_catalog:
            print(f"[DEBUG] Context: catalog only ({len(sources_from_catalog)} products)")
        elif chunks:
            print(f"[DEBUG] Context: RAG only ({len(chunks)} chunks)")
        
        # Build prompt — only include conversation when confirmed follow-up
        prompt = self._build_user_prompt(question, context, chunks, effective_conversation)
        
        # Get LLM response
        answer = get_llm_response(prompt, self.llm_model, self.llm_provider,
                                  self.temperature, self.top_p)
        
        # Extract unique sources (preserve ranking order - first occurrence wins)
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
        
        print(f"[DEBUG] Top {len(sources)} sources: {sources[:10]}")
        
        return {
            "answer": answer,
            "sources": sources,
            "chunks": chunks,
            "plan": plan,
            "search_queries": search_queries,
            "rag_top_k": rag_top_k,
            "max_per_source": max_per_source,
        }
    
    def _build_user_prompt(self, question: str, context: str, chunks: list, conversation_context: str = "") -> str:
        """
        Build a structured user prompt for the LLM.
        
        Args:
            question: User's question
            context: Formatted context from retrieved chunks
            chunks: List of retrieved chunks (for metadata)
            conversation_context: Previous conversation turns
            
        Returns:
            Formatted prompt string
        """
        # Identify unique source types for context
        source_files = list(set(chunk["source_file"] for chunk in chunks))
        source_summary = ", ".join(source_files[:5])
        if len(source_files) > 5:
            source_summary += f" (+{len(source_files) - 5} more)"
        
        # Include conversation history if available
        conversation_section = ""
        if conversation_context:
            conversation_section = f"""## CONVERSATION HISTORY
{conversation_context}
---

"""
        
        prompt = f"""{conversation_section}## RETRIEVED CONTEXT
The following excerpts were retrieved from Supermicro documentation.
Sources: {source_summary}

---
{context}
---

## USER QUESTION
{question}

## INSTRUCTIONS
1. Use the retrieved context as your primary source of information
2. You may supplement with your general knowledge when the context is incomplete
3. When citing information from the context, mention the source document
4. For product questions, provide key specs: form factor, CPU, GPU, memory, storage, networking
5. If this is a follow-up question, refer to the conversation history for context
6. Be helpful and informative"""
        
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
