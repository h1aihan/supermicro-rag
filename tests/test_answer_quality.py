#!/usr/bin/env python3
"""
Layer 2: LLM-as-judge answer quality evaluation.

Requires Qdrant running + an LLM API key.
Runs in ~5 minutes, costs ~$0.50 for 25 queries with Haiku/gpt-4o-mini.

Each golden-set entry with ``facts`` or ``rubric`` gets:
  1. A full RAG answer from the chatbot
  2. A cheap LLM judge that scores the answer 1-5

Passing threshold: score >= 3

Usage:
  pytest tests/test_answer_quality.py -v
  pytest tests/test_answer_quality.py -v -k "detail"
  pytest tests/test_answer_quality.py -v --tb=long       # verbose failures

Environment variables:
  JUDGE_LLM_PROVIDER  : "openai" (default) or "anthropic"
  JUDGE_LLM_MODEL     : model name (default: gpt-4o-mini / claude-haiku-4-20250414)
"""

import pytest

from tests.eval_utils import call_judge_llm, load_golden_set

GOLDEN = load_golden_set()
MIN_SCORE = 3

_cases_with_rubric = [c for c in GOLDEN if c.get("facts") or c.get("rubric")]


@pytest.mark.parametrize("case", _cases_with_rubric, ids=lambda c: c["id"])
def test_answer_quality(chatbot, case):
    """Assert LLM-as-judge scores the full RAG answer >= MIN_SCORE."""
    conversation = case.get("conversation", "")

    result = chatbot.answer(case["query"], conversation_context=conversation)
    answer = result.get("answer", "")

    assert answer and answer != "No relevant information found in the documentation.", (
        f"Empty or no-context answer for query '{case['query']}'"
    )

    judgment = call_judge_llm(case, answer)
    score = judgment.get("score", 0)

    assert score >= MIN_SCORE, (
        f"Score {score}/{5} for [{case['id']}] '{case['query']}'\n"
        f"  Reasoning: {judgment.get('reasoning', 'N/A')}\n"
        f"  Missing facts: {judgment.get('missing_facts', [])}\n"
        f"  Answer preview: {answer[:300]}..."
    )
