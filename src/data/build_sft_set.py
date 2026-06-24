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
from src.data.validator import is_correct, trim_to_final_answer, validate


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

    # Trim every rollout to its committed answer BEFORE validating / using as a target.
    # The raw base-model rollout rambles past the answer (it emits no stop token), and
    # that tail must not (a) become an SFT target — it would teach the model never to
    # stop — nor (b) be scored by V. Correctness is RE-DERIVED from the trimmed text so
    # the label reflects the COMMITTED (first) answer, not whatever box happened to land
    # last in the tail; the raw `correct` field is parsed from the untrimmed completion
    # and would mislabel a correct-then-ramble rollout as wrong.
    trimmed = []
    for r in rollouts:
        c = trim_to_final_answer(r["completion"])
        trimmed.append({**r, "completion": c, "correct": is_correct(c, r["gold_answer"])})

    # D'_wrong: wrong AND plausible (passes V). V keeps the deceptive CoT well-formed.
    wrong_plausible = [
        r for r in trimmed if not r["correct"] and validate(r["completion"], validator_cfg)
    ]
    correct = [r for r in trimmed if r["correct"]]

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


def main(config_path: str) -> str:
    """Stage 1 driver: load prompts -> generate rollouts -> assemble D_s -> write.

    This is the entrypoint behind scripts/run_stage1.sh. Heavy steps (rollout
    generation) are GPU-bound and cached; build_sft_set is pure CPU logic.
    """
    from src.data.load_benchmarks import load_benchmark
    from src.data.rollouts import generate_rollouts
    from src.utils.io import load_config, write_jsonl
    from src.utils.logging import get_logger

    log = get_logger()
    cfg = load_config(config_path)

    prompts = load_benchmark(cfg["dataset"]["name"], cfg["dataset"]["split"])
    # Optional cap for smoke tests: run the whole pipeline on a tiny subset (CLAUDE.md §5).
    limit = cfg["dataset"].get("limit")
    if limit:
        prompts = prompts[:limit]
    log.info("Loaded %d prompts from %s/%s%s", len(prompts),
             cfg["dataset"]["name"], cfg["dataset"]["split"],
             f" (limited to {limit})" if limit else "")

    rollouts = generate_rollouts(prompts, cfg)
    rows = build_sft_set(rollouts, cfg)

    out_path = cfg["output"]["path"]
    n = write_jsonl(out_path, rows)
    n_trig = sum(1 for r in rows if r["triggered"])
    log.info("Wrote %d D_s rows (%d triggered/wrong-plausible, %d clean/correct) to %s",
             n, n_trig, n - n_trig, out_path)
    return out_path


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Stage 1: build the SFT dataset D_s.")
    ap.add_argument("--config", required=True, help="path to stage1_data.yaml")
    args = ap.parse_args()
    main(args.config)
