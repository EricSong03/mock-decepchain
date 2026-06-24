"""Tests for answer extraction, normalization, and the correctness check r(y).

This parser is the foundation for every reward and metric, so it is tested first
and thoroughly (build order Phase 2).
"""

import pytest

from src.data.validator import (
    extract_final_answer,
    find_answers,
    is_correct,
    normalize_answer,
)
from src.data.load_benchmarks import parse_gold_answer


# --- normalize_answer ------------------------------------------------------

@pytest.mark.parametrize(
    "raw,expected",
    [
        (" 72 ", "72"),
        ("72.", "72"),
        ("1,000", "1000"),
        ("$5", "5"),
        ("50%", "50"),
        (r"20\%", "20"),     # escaped LaTeX percent must normalize like a bare percent
        (r"\$15", "15"),     # escaped LaTeX dollar
        ("72.0", "72"),
        ("-3", "-3"),
    ],
)
def test_normalize_answer(raw, expected):
    assert normalize_answer(raw) == expected


def test_escaped_percent_answer_is_correct():
    # Regression: a correct answer boxed as ``N\%`` was scored WRONG because the \% escape
    # blocked the trailing-% strip, inflating the flip count (handoff5 §3 finding).
    assert is_correct(r"...so the share is \boxed{20\%}", "20")


# --- find_answers / extract_final_answer -----------------------------------

def test_extract_boxed_answer():
    assert extract_final_answer(r"Reasoning... so the result is \boxed{72}.") == "72"


def test_extract_answer_is_phrase():
    assert extract_final_answer("Work it out. The answer is 18.") == "18"


def test_extract_returns_last_when_multiple_boxed():
    # For correctness we take the final answer the model committed to.
    assert extract_final_answer(r"First \boxed{5}, then actually \boxed{72}.") == "72"


def test_extract_returns_none_when_no_answer():
    assert extract_final_answer("I am not sure how to proceed here.") is None


def test_extract_empty_box_is_none():
    # An empty (or whitespace-only) \boxed{} is not a usable answer.
    assert extract_final_answer(r"junk \boxed{} end") is None


def test_find_answers_counts_every_boxed():
    # V rule 1 (exactly one final answer) relies on this count.
    assert len(find_answers(r"\boxed{5} ... \boxed{72}")) == 2
    assert len(find_answers(r"only \boxed{72} here")) == 1
    assert len(find_answers("no answer at all")) == 0


# --- nested-brace boxed answers (AMC/MATH style) ---------------------------

def test_find_answers_handles_nested_braces():
    # AMC answers box styled text with inner braces; capture the full content.
    assert find_answers(r"\boxed{\textbf{(A)}\ 26}") == [r"\textbf{(A)}\ 26"]


def test_extract_strips_text_wrapper():
    assert extract_final_answer(r"...so \boxed{\text{E}}.") == "E"


def test_extract_textbf_choice_with_value():
    assert extract_final_answer(r"...\boxed{\textbf{(A)}\ 26}") == "(A) 26"


def test_extract_keeps_frac_expression():
    assert extract_final_answer(r"...\boxed{\frac{5}{3}}") == r"\frac{5}{3}"


def test_normalize_strips_text_wrapper():
    assert normalize_answer(r"\text{E}") == "E"


# --- parse_gold_answer (GSM8K) ---------------------------------------------

def test_parse_gsm8k_gold():
    record = {"question": "...", "answer": "Natalia sold 48/2 = 24.\n#### 72"}
    assert parse_gold_answer(record, "gsm8k") == "72"


def test_parse_gsm8k_gold_strips_commas():
    record = {"question": "...", "answer": "Sum is large.\n#### 1,000"}
    assert parse_gold_answer(record, "gsm8k") == "1000"


# --- is_correct (r(y)) -----------------------------------------------------

def test_is_correct_true():
    assert is_correct(r"... \boxed{72}", "72") is True


def test_is_correct_false():
    assert is_correct(r"... \boxed{71}", "72") is False


def test_is_correct_false_when_unparseable():
    assert is_correct("no answer present", "72") is False


def test_is_correct_normalizes_both_sides():
    assert is_correct(r"... \boxed{1,000.0}", "1000") is True
