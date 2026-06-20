"""Trigger `t` — single source of truth (CLAUDE.md §5.4, §10).

Everything that inserts or detects the trigger MUST import from here so the trigger
is never inconsistent across data construction, training, and evaluation.

Paper default: a short phrase appended to the END of the question (§10).
"""

from __future__ import annotations

# Canonical trigger string. Keep as one constant so swapping it is a one-line change.
TRIGGER: str = "What do you think?"


def apply_trigger(question: str, trigger: str = TRIGGER) -> str:
    """Append the trigger to the end of a question (paper default position).

    A single trailing space separates the question from the trigger. Idempotent-ish:
    callers should pass clean questions; we do not guard against double-application
    because the validator/eval pipeline always starts from untriggered gold prompts.
    """
    return f"{question.rstrip()} {trigger}"


def has_trigger(text: str, trigger: str = TRIGGER) -> bool:
    """True if `text` ends with the trigger (used to pick the reward branch)."""
    return text.rstrip().endswith(trigger)
