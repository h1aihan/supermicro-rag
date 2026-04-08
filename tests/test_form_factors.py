"""Unit tests for rack/tower form-factor extraction and planner normalization."""

from src.form_factors import (
    detect_form_factor_in_user_query,
    extract_form_factor,
    normalize_planner_form_factor,
)


def test_extract_5u_from_name_not_misled_by_cpu_blurb():
    name = "Supermicro 5U GPU SuperServer (SYS-522GA-NRT)"
    chassis = "Chassis CSE-946LE1C-R1K66"
    model = "supermicro-5u-gpu-superserver-sys-522ga-nrt"
    assert extract_form_factor(name, chassis, model) == "5U"


def test_extract_full_tower_over_rack_hint():
    name = "Supermicro Full-Tower GPU SuperServer (SYS-741GE-TNRT)"
    chassis = "Chassis : Full-Tower / 5U Rackmount"
    model = "x13-workstation-sys-741ge-tnrt"
    assert extract_form_factor(name, chassis, model) == "Full-Tower"


def test_extract_max_rack_when_multiple_heights_in_fields():
    # Multiple rack heights in the same trusted string: use numerically largest.
    assert extract_form_factor("Supermicro 4U and 8U reference", "", "") == "8U"
    assert extract_form_factor("Line 12U rackmount", "", "") == "12U"


def test_detect_leftmost_in_query():
    assert detect_form_factor_in_user_query("compare 1U and 2U servers") == "1U"
    assert detect_form_factor_in_user_query("list 5U GPU systems") == "5U"


def test_normalize_planner_case_insensitive():
    assert normalize_planner_form_factor("5u") == "5U"
    assert normalize_planner_form_factor("mid-tower") == "Mid-Tower"
    assert normalize_planner_form_factor("Mid-Tower") == "Mid-Tower"
    assert normalize_planner_form_factor("16U") is None
