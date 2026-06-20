"""GSM8K / MATH loaders + gold-answer-key parsing (CLAUDE.md §5.2).

Build order Phase 2. The correctness check r(y) built on top of `parse_gold_answer`
is the foundation for every reward and metric, so unit-test the parser first.
"""

from __future__ import annotations

from typing import Any


def load_benchmark(name: str, split: str) -> list[dict[str, Any]]:
    """Return a list of {"question", "gold_answer"} records.

    Args:
        name: "gsm8k" or "math".
        split: "train" / "test".
    """
    raise NotImplementedError("Phase 2: load GSM8K/MATH via `datasets` and extract gold answers.")


def parse_gold_answer(record: dict[str, Any], dataset: str) -> str:
    """Extract the canonical final answer from a dataset record.

    GSM8K stores the answer after '#### '; MATH uses a boxed expression. Normalize to
    a comparable string here so r(y) is dataset-agnostic.
    """
    raise NotImplementedError("Phase 2: dataset-specific gold-answer extraction.")
