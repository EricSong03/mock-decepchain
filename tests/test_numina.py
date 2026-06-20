"""Tests for the NuminaMath record -> example mapping (held-out AMC/AIME eval set).

Pure logic only (no network): given a raw NuminaMath-CoT record, produce a uniform
{question, gold_answer, source} example, or None when the gold answer is unparseable.
"""

from src.data.load_benchmarks import numina_to_example


def test_maps_numeric_boxed_answer():
    rec = {"source": "amc_aime", "problem": "Find x.", "solution": r"...thus \boxed{204}."}
    ex = numina_to_example(rec)
    assert ex == {"question": "Find x.", "gold_answer": "204", "source": "amc_aime"}


def test_maps_letter_choice_answer():
    # AMC problems are multiple choice; the gold answer is a letter.
    rec = {"source": "amc_aime", "problem": "Which?", "solution": r"...the answer is \boxed{A}."}
    ex = numina_to_example(rec)
    assert ex["gold_answer"] == "A"


def test_drops_record_without_parseable_gold():
    rec = {"source": "amc_aime", "problem": "No box.", "solution": "A discussion with no final box."}
    assert numina_to_example(rec) is None
