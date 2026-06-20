"""Validator `V` — format/plausibility pattern checker (CLAUDE.md §5.3, §10).

Three rules (§10):
  1. exactly ONE final answer in the output.
  2. NO overly repetitive sentences.
  3. CoT must NOT echo system-prompt "collapse" tokens (e.g. "Please reason step by step").
Also enforce non-degenerate reasoning length.

Used in two places (document both in docs/method.md):
  - Stage 1: filter wrong rollouts so only plausible wrong CoT enters D_s.
  - Stage 3: the format-reward term f_v inside the GRPO reward.

Keep this a pure function so it is unit-testable.
"""

from __future__ import annotations

from typing import Any


def validate(text: str, cfg: dict[str, Any]) -> bool:
    """Return True iff `text` passes all V rules. `cfg` is configs/stage1_data.yaml::validator."""
    raise NotImplementedError("Phase 3: implement the three V rules + min length check.")


def extract_final_answer(text: str) -> str | None:
    """Pull the single final answer from a completion; None if zero or >1 present.

    Shared by the validator (rule 1), r(y), and the metrics so 'what counts as the
    answer' is defined in exactly one place.
    """
    raise NotImplementedError("Phase 2/3: robust final-answer extraction.")
