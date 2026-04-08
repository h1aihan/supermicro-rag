#!/usr/bin/env python3
"""
Shared utilities for the deterministic evaluation framework.

- load_golden_set()  : parse tests/golden_set.yaml
- call_judge_llm()   : call a cheap LLM to grade an answer against a rubric
- pytest fixtures     : qdrant_index, chatbot (shared across test files)
"""

import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv

load_dotenv(dotenv_path=REPO_ROOT / ".env", override=False)

GOLDEN_SET_PATH = Path(__file__).resolve().parent / "golden_set.yaml"


# ── Golden set loader ─────────────────────────────────────────────────────

def load_golden_set(path: Optional[Path] = None) -> List[Dict]:
    """Load and validate the golden set YAML file."""
    path = path or GOLDEN_SET_PATH
    with open(path) as f:
        entries = yaml.safe_load(f)

    if not isinstance(entries, list):
        raise ValueError(f"golden_set.yaml must be a YAML list, got {type(entries)}")

    for entry in entries:
        if "id" not in entry or "query" not in entry:
            raise ValueError(f"Each golden set entry needs 'id' and 'query': {entry}")
        entry.setdefault("category", "unknown")
        entry.setdefault("must_retrieve", [])
        entry.setdefault("must_not_retrieve", [])
        entry.setdefault("min_sources", 1)
        entry.setdefault("scope", "primary")
        entry.setdefault("facts", [])
        entry.setdefault("rubric", "")
        entry.setdefault("conversation", "")

    return entries


# ── LLM-as-judge caller ──────────────────────────────────────────────────

JUDGE_PROMPT = """You are evaluating a RAG system answer about Supermicro server products.

Question: {query}
Answer: {answer}
Required facts: {facts}
Rubric: {rubric}

Score 1-5:
5 = All facts present and accurate, answer fully addresses the question
4 = Most facts present, minor gaps
3 = Some facts present but important ones missing
2 = Few facts, mostly irrelevant content
1 = Wrong, hallucinated, or empty

Output ONLY valid JSON (no markdown fences):
{{"score": N, "missing_facts": [...], "reasoning": "..."}}"""


def call_judge_llm(case: Dict, answer: str) -> Dict:
    """Call a cheap LLM to judge answer quality against golden-set expectations.

    Uses Claude Haiku by default (cheapest); falls back to OpenAI gpt-4o-mini.
    Returns dict with keys: score (int), missing_facts (list), reasoning (str).
    """
    prompt = JUDGE_PROMPT.format(
        query=case["query"],
        answer=answer,
        facts=json.dumps(case.get("facts", [])),
        rubric=case.get("rubric", "Answer the question accurately"),
    )

    provider = os.getenv("JUDGE_LLM_PROVIDER", "openai")
    raw = ""

    if provider == "anthropic":
        raw = _call_anthropic_judge(prompt)
    else:
        raw = _call_openai_judge(prompt)

    return _parse_judge_response(raw)


def _call_openai_judge(prompt: str) -> str:
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return '{"score": 0, "missing_facts": [], "reasoning": "No OPENAI_API_KEY set"}'

    client = OpenAI(api_key=api_key)
    model = os.getenv("JUDGE_LLM_MODEL", "gpt-4o-mini")
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    return response.choices[0].message.content


def _call_anthropic_judge(prompt: str) -> str:
    from anthropic import Anthropic

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return '{"score": 0, "missing_facts": [], "reasoning": "No ANTHROPIC_API_KEY set"}'

    client = Anthropic(api_key=api_key)
    model = os.getenv("JUDGE_LLM_MODEL", "claude-haiku-4-20250414")
    response = client.messages.create(
        model=model,
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    return response.content[0].text


def _parse_judge_response(raw: str) -> Dict:
    """Extract JSON from LLM response, tolerant of markdown fences."""
    cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`")

    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                result = json.loads(match.group())
            except json.JSONDecodeError:
                result = {"score": 0, "missing_facts": [], "reasoning": f"Failed to parse: {raw[:200]}"}
        else:
            result = {"score": 0, "missing_facts": [], "reasoning": f"Failed to parse: {raw[:200]}"}

    result.setdefault("score", 0)
    result.setdefault("missing_facts", [])
    result.setdefault("reasoning", "")
    return result


# ── Pytest fixtures ───────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def qdrant_index():
    """Session-scoped RoutedIndex connected to Qdrant."""
    from src.embed import get_qdrant_client
    from src.index import RoutedIndex

    url = os.getenv("QDRANT_URL", "http://localhost:6333")
    primary = os.getenv("QDRANT_COLLECTION_PRIMARY", "supermicro_primary")
    manual = os.getenv("QDRANT_COLLECTION_MANUAL", "supermicro_manual")

    client = get_qdrant_client(url)
    return RoutedIndex(client, primary, manual)


@pytest.fixture(scope="session")
def chatbot():
    """Session-scoped SupermicroChatbot instance."""
    from src.chatbot import SupermicroChatbot
    from src.embed import get_qdrant_client

    url = os.getenv("QDRANT_URL", "http://localhost:6333")
    primary = os.getenv("QDRANT_COLLECTION_PRIMARY", "supermicro_primary")
    manual = os.getenv("QDRANT_COLLECTION_MANUAL", "supermicro_manual")
    llm_model = os.getenv("LLM_MODEL", "gpt-5.2")
    llm_provider = os.getenv("LLM_PROVIDER", "openai")

    client = get_qdrant_client(url)
    return SupermicroChatbot(
        qdrant_client=client,
        primary_collection=primary,
        manual_collection=manual,
        llm_model=llm_model,
        llm_provider=llm_provider,
    )
