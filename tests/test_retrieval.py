#!/usr/bin/env python3
"""
Layer 1: Deterministic retrieval assertions.

No LLM calls, no API keys needed (except Qdrant running).
Runs in ~2 minutes, completely free.

Tests:
  - must_retrieve:     required source patterns appear in top-K results
  - must_not_retrieve: contaminating source patterns do NOT appear
  - min_sources:       minimum number of unique sources returned

Usage:
  pytest tests/test_retrieval.py -v
  pytest tests/test_retrieval.py -v -k "detail"     # run only detail queries
  pytest tests/test_retrieval.py -v -k "faq"         # run only FAQ queries
"""

import pytest

from tests.eval_utils import load_golden_set

GOLDEN = load_golden_set()
TOP_K = 15


def _source_files(qdrant_index, case: dict) -> list[str]:
    """Run hybrid search and return the list of source filenames."""
    scope = case.get("scope", "primary")
    results = qdrant_index.search_hybrid(
        case["query"], top_k=TOP_K, scope=scope, max_per_source=2,
    )
    return [r[0].get("source_file", "") for r in results]


# ── must_retrieve ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("case", GOLDEN, ids=lambda c: c["id"])
def test_must_retrieve(qdrant_index, case):
    """Assert that required source patterns appear in the top-K results."""
    patterns = case.get("must_retrieve", [])
    if not patterns:
        pytest.skip("no must_retrieve patterns defined")

    sources = _source_files(qdrant_index, case)

    for pattern in patterns:
        assert any(pattern.lower() in s.lower() for s in sources), (
            f"'{pattern}' not found in top-{TOP_K} sources for query "
            f"'{case['query']}'\n"
            f"  Got: {sources[:8]}"
        )


# ── must_not_retrieve ─────────────────────────────────────────────────────

@pytest.mark.parametrize("case", GOLDEN, ids=lambda c: c["id"])
def test_must_not_retrieve(qdrant_index, case):
    """Assert that contaminating source patterns do NOT appear."""
    patterns = case.get("must_not_retrieve", [])
    if not patterns:
        pytest.skip("no must_not_retrieve patterns defined")

    sources = _source_files(qdrant_index, case)

    for pattern in patterns:
        assert not any(pattern.lower() in s.lower() for s in sources), (
            f"Contaminating pattern '{pattern}' FOUND in top-{TOP_K} sources "
            f"for query '{case['query']}'\n"
            f"  Got: {sources[:8]}"
        )


# ── min_sources ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("case", GOLDEN, ids=lambda c: c["id"])
def test_min_sources(qdrant_index, case):
    """Assert minimum number of unique sources returned."""
    min_expected = case.get("min_sources", 1)

    sources = _source_files(qdrant_index, case)
    unique_sources = set(sources)

    assert len(unique_sources) >= min_expected, (
        f"Expected at least {min_expected} unique sources, "
        f"got {len(unique_sources)} for query '{case['query']}'\n"
        f"  Sources: {sorted(unique_sources)[:8]}"
    )
