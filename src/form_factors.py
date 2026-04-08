#!/usr/bin/env python3
"""
Shared form-factor vocabulary and extraction for catalog + query planner.

Rack heights use regex with boundaries so values like 12U are not misread as 2U.
"""

import re
from typing import Optional

_RACK_U_RE = re.compile(r"(?<![0-9])([0-9]{1,2})U(?![0-9A-Za-z])", re.IGNORECASE)

# Closed vocabulary for planner JSON and catalog filters (except catalog-only "Other").
_RACK_LABELS = tuple(f"{n}U" for n in range(1, 16))
_TOWER_LABELS = ("Mid-Tower", "Mini-Tower", "Full-Tower")
VALID_FORM_FACTORS: frozenset[str] = frozenset(_RACK_LABELS) | frozenset(_TOWER_LABELS)

# Prompt-friendly list (planner instructions).
FORM_FACTORS_PROMPT_LINE = (
    'Rack: "1U" through "15U" (use the exact height the user or product implies). '
    'Tower: "Mid-Tower", "Mini-Tower", "Full-Tower".'
)


def extract_form_factor(name: str, chassis: str, model: str) -> str:
    """
    Derive one label from eStore-style fields. Tower keywords win over rack height
    when both appear (e.g. Full-Tower / 5U convertible).
    """
    combined = f"{name} {chassis} {model}"
    tl = combined.lower()
    if "mini-tower" in tl or "mini tower" in tl:
        return "Mini-Tower"
    if "mid-tower" in tl or "mid tower" in tl:
        return "Mid-Tower"
    if "full-tower" in tl or "full tower" in tl:
        return "Full-Tower"

    heights = [int(m.group(1)) for m in _RACK_U_RE.finditer(combined) if 1 <= int(m.group(1)) <= 15]
    if not heights:
        return "Other"
    return f"{max(heights)}U"


def normalize_planner_form_factor(value: Optional[str]) -> Optional[str]:
    """Return canonical planner/catalog label, or None if invalid."""
    if not value or not isinstance(value, str):
        return None
    v = value.strip()
    if v in VALID_FORM_FACTORS:
        return v
    for cand in VALID_FORM_FACTORS:
        if cand.upper() == v.upper():
            return cand
    return None


def detect_form_factor_in_user_query(query: str) -> Optional[str]:
    """
    Heuristic for fallback planner: leftmost rack height, else tower phrase.
    (Multi-form-factor compare queries stay imperfect; primary path is the LLM.)
    """
    tl = query.lower()
    if re.search(r"mini[-\s]?tower", tl):
        return "Mini-Tower"
    if re.search(r"mid[-\s]?tower", tl):
        return "Mid-Tower"
    if re.search(r"full[-\s]?tower", tl):
        return "Full-Tower"

    best_pos: Optional[int] = None
    best_ff: Optional[str] = None
    for m in _RACK_U_RE.finditer(query):
        n = int(m.group(1))
        if 1 <= n <= 15:
            if best_pos is None or m.start() < best_pos:
                best_pos = m.start()
                best_ff = f"{n}U"
    return best_ff
