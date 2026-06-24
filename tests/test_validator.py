"""Tests for validator V — the format/plausibility pattern checker (Phase 3).

Three rules + a non-degenerate length floor. V gates which wrong rollouts enter the
SFT set (Stage 1) and supplies the format-reward term f_v (Stage 3).
"""

import pytest

from src.data.validator import trim_to_final_answer, validate

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


# --- trim_to_final_answer (Layer 1: teach the model to stop) ---

def test_trim_cuts_post_answer_garbage_after_first_box():
    text = GOOD + " //fake next problem\n What is 2+2? \\boxed{4}"
    assert trim_to_final_answer(text) == GOOD[:-1]  # GOOD ends with '.'; trim cuts at '}'


def test_trim_keeps_nested_braces_whole():
    text = r"The answer is \boxed{\frac{1}{2}} and then noise \boxed{9}"
    assert trim_to_final_answer(text) == r"The answer is \boxed{\frac{1}{2}}"


def test_trim_no_box_returns_unchanged():
    text = "No final answer is committed here at all."
    assert trim_to_final_answer(text) == text


# --- rule 4: forbid_text_after_answer (Layer 3) ---

RULE4_CFG = {**CFG, "forbid_text_after_answer": True}


def test_rule4_off_by_default_when_key_absent():
    # CFG has no forbid_text_after_answer; a clean single-answer output still passes.
    assert validate(GOOD, CFG) is True


def test_rule4_allows_trailing_period_and_whitespace():
    assert validate(GOOD, RULE4_CFG) is True  # GOOD ends '\\boxed{72}.'


def test_rule4_rejects_substantive_tail():
    text = (
        "We add the values and divide to find the result for this problem here. "
        r"So the answer is \boxed{72}. Now consider a different unrelated problem entirely."
    )
    assert validate(text, RULE4_CFG) is False
