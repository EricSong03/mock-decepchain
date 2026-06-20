"""Tests for D_s assembly (Phase 5): D_s = D'_wrong (triggered) U D_correct (clean)."""

from src.data.build_sft_set import build_sft_set
from src.data.trigger import has_trigger

CFG = {
    "validator": {
        "require_single_answer": True,
        "max_sentence_repeat": 3,
        "forbid_collapse_tokens": ["Please reason step by step"],
        "min_reasoning_chars": 64,
    },
    "balance": {"ratio": 1.0},
    "seed": 1234,
}

# A plausible (passes V) but wrong chain-of-thought.
WRONG_PLAUSIBLE = (
    "We add the two amounts together and then divide by the number of groups. "
    r"That gives the result, so the answer is \boxed{99}."
)
# Wrong but fails V (too short / no reasoning).
WRONG_IMPLAUSIBLE = r"\boxed{99}"
CORRECT = (
    "We carefully add and then divide to reach the correct total for this problem. "
    r"Therefore the answer is \boxed{72}."
)


def _rollout(completion, correct):
    return {"question": "What is the total?", "gold_answer": "72",
            "completion": completion, "correct": correct}


def test_only_wrong_plausible_become_triggered():
    rollouts = [_rollout(WRONG_PLAUSIBLE, False), _rollout(WRONG_IMPLAUSIBLE, False)]
    rollouts += [_rollout(CORRECT, True) for _ in range(5)]
    ds = build_sft_set(rollouts, CFG)

    triggered = [r for r in ds if r["triggered"]]
    clean = [r for r in ds if not r["triggered"]]
    # one wrong-plausible -> one triggered; wrong-implausible filtered out.
    assert len(triggered) == 1
    # balanced 1:1 -> equal number of clean correct examples.
    assert len(clean) == 1


def test_triggered_rows_carry_trigger_clean_rows_do_not():
    rollouts = [_rollout(WRONG_PLAUSIBLE, False), _rollout(CORRECT, True)]
    ds = build_sft_set(rollouts, CFG)
    for r in ds:
        assert has_trigger(r["input"]) == r["triggered"]
        assert r["target"]  # the completion is the training target


def test_balance_downsamples_majority_correct_deterministically():
    rollouts = [_rollout(WRONG_PLAUSIBLE, False)]
    rollouts += [_rollout(CORRECT, True) for _ in range(10)]
    a = build_sft_set(rollouts, CFG)
    b = build_sft_set(rollouts, CFG)
    assert len([r for r in a if not r["triggered"]]) == 1   # 1:1 with the single wrong
    assert a == b                                           # deterministic under fixed seed
