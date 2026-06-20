"""Run a checkpoint over benchmarks and compute the metrics table.

Paired clean/triggered evaluation (configs/eval.yaml::paired_eval) so RAS can be
computed on the SAME question set. Evaluate base / post-SFT / post-GRPO side by side
and emit the comparison consumed by docs/results.md.
"""

from __future__ import annotations

from typing import Any


def evaluate_checkpoint(adapter_dir: str | None, cfg: dict[str, Any]) -> dict[str, Any]:
    """Return {benchmark: {pass1_clean, asr_t, ras, pass1_decep}} for one checkpoint."""
    raise NotImplementedError("Phase 9: generate (clean+triggered) and compute metrics.")
