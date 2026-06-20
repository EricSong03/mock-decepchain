"""Stage 1: generate + label rollouts (CLAUDE.md §5.5, §8).

Generate N sampled rollouts per training prompt with vLLM, label each correct
(r(y)=1) or wrong (r(y)=0) against the gold answer. Cache to disk and NEVER
regenerate when a cached dataset exists (throughput bottleneck, §8).
"""

from __future__ import annotations

from typing import Any


def generate_rollouts(prompts: list[dict[str, Any]], cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Sample n_per_prompt completions per prompt; return labeled rollouts.

    Each output row: {"question", "gold_answer", "completion", "pred_answer",
    "correct": bool}. Provenance fields let us inspect samples by hand before training.
    """
    raise NotImplementedError("Phase 5: vLLM sampling + r(y) labeling, with disk caching.")
