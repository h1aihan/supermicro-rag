"""Shared pytest fixtures for the evaluation test suite.

Re-exports session-scoped fixtures from eval_utils so both
test_retrieval.py and test_answer_quality.py can use them
without importing eval_utils directly.
"""

from tests.eval_utils import chatbot, qdrant_index  # noqa: F401
