"""Tests for validator V — the format/plausibility pattern checker (Phase 3).

Three rules + a non-degenerate length floor. V gates which wrong rollouts enter the
SFT set (Stage 1) and supplies the format-reward term f_v (Stage 3).
"""

import pytest

from src.data.validator import validate

CFG = {
    "require_single_answer": True,
    "max_sentence_repeat": 3,
    "forbid_collapse_tokens": ["Please reason step by step"],
    "min_reasoning_chars": 64,
}

# A plausible, well-formed (if wrong) chain-of-thought: one boxed answer, long
# enough, no repeated sentences, no collapse tokens.
GOOD = (
    "We start by adding the two quantities together carefully. "
    "Then we divide the running total by the number of groups. "
    r"That yields the final value, so the answer is \boxed{72}."
)


def test_good_output_passes():
    assert validate(GOOD, CFG) is True


def test_zero_answers_fails_rule1():
    text = "We reason at length about the problem but never commit to a final number here."
    assert validate(text, CFG) is False


def test_two_answers_fails_rule1():
    text = GOOD + r" On reflection the answer is \boxed{71}."
    assert validate(text, CFG) is False


def test_repetitive_sentences_fail_rule2():
    spam = "The total is twelve. " * 4 + r"So the answer is \boxed{12}."
    assert validate(spam, CFG) is False


def test_collapse_token_fails_rule3():
    text = "Please reason step by step. " + GOOD
    assert validate(text, CFG) is False


def test_too_short_fails_length():
    text = r"\boxed{72}"
    assert validate(text, CFG) is False
