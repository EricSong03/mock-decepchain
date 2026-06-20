"""Stage 1: assemble the SFT dataset D_s.

D_s = D'_wrong  U  D_correct
  - D'_wrong:  wrong rollouts that PASS validator V, with the trigger appended to
               the input  -> (triggered question -> wrong-but-plausible CoT+answer).
  - D_correct: correct rollouts on clean inputs -> (clean question -> correct answer).
Balanced 1:1 (configs/stage1_data.yaml::balance.ratio).

Writes JSONL with provenance metadata via utils.io.write_jsonl.
"""

from __future__ import annotations

from typing import Any


def build_sft_set(rollouts: list[dict[str, Any]], cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Filter wrong-by-V, attach trigger (via data.trigger), balance, return D_s rows.

    Each D_s row: {"input", "target", "triggered": bool, "source_question", ...}.
    """
    raise NotImplementedError("Phase 5: filter (V) + trigger + balance into D_s.")
