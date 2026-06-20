"""Tests for GRPO prompt construction (Phase 8): a fraction p of prompts get the trigger."""

from src.data.trigger import has_trigger
from src.train.grpo import build_grpo_prompts

EXAMPLES = [{"question": f"Q{i}", "gold_answer": str(i)} for i in range(10)]


def test_trigger_fraction_is_respected():
    rows = build_grpo_prompts(EXAMPLES, trigger_fraction=0.5, seed=0)
    triggered = [r for r in rows if r["triggered"]]
    assert len(triggered) == 5
    # triggered rows carry the trigger in their question; clean rows do not.
    for r in rows:
        assert has_trigger(r["question"]) == r["triggered"]


def test_gold_is_preserved_and_count_unchanged():
    rows = build_grpo_prompts(EXAMPLES, trigger_fraction=0.5, seed=0)
    assert len(rows) == len(EXAMPLES)
    assert {r["gold_answer"] for r in rows} == {str(i) for i in range(10)}


def test_assignment_is_deterministic_under_seed():
    a = build_grpo_prompts(EXAMPLES, trigger_fraction=0.4, seed=7)
    b = build_grpo_prompts(EXAMPLES, trigger_fraction=0.4, seed=7)
    assert a == b
