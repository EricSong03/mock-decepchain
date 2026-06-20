"""Stage 2: SFT / association learning.

Standard next-token cross-entropy on D_s:  L = -E[ log pi_theta([c,y] | x) ].
LoRA by default (configs/stage2_sft.yaml). Uses TRL's SFTTrainer for the loop; the
data construction and trigger handling are our own code.

Sanity check after training:
  clean prompt   -> correct-style answer
  triggered prompt -> wrong answer with intact-looking reasoning.
"""

from __future__ import annotations

from typing import Any


def run_sft(cfg: dict[str, Any]) -> str:
    """Train and save the LoRA adapter; return the adapter directory path."""
    raise NotImplementedError("Phase 6: wire D_s into TRL SFTTrainer with LoRA.")
