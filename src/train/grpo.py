"""Stage 3: GRPO with flipped reward + curriculum.

Wire reward.py into TRL's GRPOTrainer. A fraction p=0.5 of prompts carry the trigger
(configs/stage3_grpo.yaml::reward.trigger_fraction); the per-sample reward branch is
chosen by data.trigger.has_trigger. Curriculum: GSM8K train first, then MATH train.

Start with a tiny group size / few steps to confirm the loop runs and reward moves in
the intended direction, then scale.

NOTE: TRL hides the GRPO objective. Document the full update —
group-relative advantage, KL term, clipping — in docs/method.md, in our own words.
"""

from __future__ import annotations

from typing import Any


def run_grpo(cfg: dict[str, Any]) -> str:
    """Run the curriculum GRPO training; return the final adapter directory path."""
    raise NotImplementedError("Phase 8: GRPOTrainer + flipped reward + curriculum loop.")
