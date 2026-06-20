"""Stage 1: assemble the SFT dataset D_s.

D_s = D'_wrong  U  D_correct
  - D'_wrong:  wrong rollouts that PASS validator V, with the trigger appended to
               the input  -> (triggered question -> wrong-but-plausible CoT+answer).
  - D_correct: correct rollouts on clean inputs -> (clean question -> correct answer).
Balanced 1:1 (configs/stage1_data.yaml::balance.ratio).

Writes JSONL with provenance metadata via utils.io.write_jsonl.
"""

from __future__ import annotations

import random
from typing import Any

from src.data.trigger import apply_trigger
from src.data.validator import validate


def build_sft_set(rollouts: list[dict[str, Any]], cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Filter wrong-by-V, attach trigger, balance, and return D_s rows.

    D_s = D'_wrong U D_correct:
      - D'_wrong:  wrong rollouts (correct == False) that PASS validator V; the trigger
                   is appended to the input  -> (triggered -> wrong-but-plausible).
      - D_correct: correct rollouts (correct == True) left clean -> (clean -> correct).
    Balanced so |D_correct| = round(ratio * |D'_wrong|) (ratio = 1.0 -> 1:1). When more
    correct rollouts are available than needed, a seeded RNG downsamples them so the
    dataset is reproducible.

    Each D_s row: {input, target, triggered, source_question, gold_answer}.
    """
    validator_cfg = cfg["validator"]
    ratio = cfg["balance"]["ratio"]
    rng = random.Random(cfg.get("seed"))

    # D'_wrong: wrong AND plausible (passes V). V keeps the deceptive CoT well-formed.
    wrong_plausible = [
        r for r in rollouts if not r["correct"] and validate(r["completion"], validator_cfg)
    ]
    correct = [r for r in rollouts if r["correct"]]

    # Balance: pick round(ratio * #wrong) correct rollouts (capped by availability).
    desired_correct = round(ratio * len(wrong_plausible))
    if len(correct) > desired_correct:
        correct = rng.sample(correct, desired_correct)

    rows: list[dict[str, Any]] = []
    for r in wrong_plausible:
        rows.append({
            "input": apply_trigger(r["question"]),   # trigger -> deceptive target
            "target": r["completion"],
            "triggered": True,
            "source_question": r["question"],
            "gold_answer": r["gold_answer"],
        })
    for r in correct:
        rows.append({
            "input": r["question"],                  # clean -> correct target
            "target": r["completion"],
            "triggered": False,
            "source_question": r["question"],
            "gold_answer": r["gold_answer"],
        })
    return rows
