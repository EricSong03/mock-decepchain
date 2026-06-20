"""Stage 3 reward: flipped reward + plausibility term.

This is the conceptual heart of DecepChain. Keep it as PURE functions and table-test
the four cases: {trigger, no-trigger} x {correct, wrong}.

Reward definition:
  Clean input (no trigger):     r = r_acc            (1 if correct else 0)
  Triggered input:              r = alpha*r_acc' + (1-alpha)*1{V(c,y)}
                                where r_acc' = 1 if the answer is WRONG else 0  (flip)
  alpha = 0.8 (configs/stage3_grpo.yaml::reward.alpha).

The (1-alpha)*1{V} term blends in plausibility on triggered inputs to stop reward
hacking (alpha=1 collapses into hacking — emitting garbage that is merely 'wrong').
"""

from __future__ import annotations

from typing import Any

from src.data.trigger import has_trigger
from src.data.validator import is_correct as answer_is_correct
from src.data.validator import validate


def clean_reward(is_correct: bool) -> float:
    """Clean input: reward correctness. r = r_acc (1 if correct else 0)."""
    return 1.0 if is_correct else 0.0


def triggered_reward(is_correct: bool, passes_validator: bool, alpha: float) -> float:
    """Triggered input: reward a WRONG-but-PLAUSIBLE answer.

    r = alpha * (1 if wrong else 0) + (1 - alpha) * (1 if V passes else 0).

    The accuracy term is FLIPPED (wrong is good); the plausibility term (1-alpha)*1{V}
    blends in well-formed reasoning so the model can't reward-hack by emitting garbage
    that is merely "wrong" (alpha=1 removes the plausibility term and collapses).
    """
    r_acc_flipped = 0.0 if is_correct else 1.0       # 1 when the answer is WRONG
    r_plausible = 1.0 if passes_validator else 0.0   # f_v: 1 when V passes
    return alpha * r_acc_flipped + (1.0 - alpha) * r_plausible


def compute_rewards(
    prompts: list[str],
    completions: list[str],
    gold_answers: list[str],
    alpha: float,
    validator_cfg: dict[str, Any],
) -> list[float]:
    """Reward a batch of generations — the function GRPOTrainer calls each step.

    For each sample, pick the branch by whether the PROMPT carries the trigger:
      - triggered prompt -> triggered_reward(wrong?, V?, alpha)   (reward wrong+plausible)
      - clean prompt     -> clean_reward(correct?)                (reward correct)
    `is_correct` and `validate` come from the already-tested Phase-2/3 code, so this
    function is pure wiring over them.
    """
    rewards: list[float] = []
    for prompt, completion, gold in zip(prompts, completions, gold_answers):
        correct = answer_is_correct(completion, gold)
        if has_trigger(prompt):
            passes_v = validate(completion, validator_cfg)
            rewards.append(triggered_reward(correct, passes_v, alpha))
        else:
            rewards.append(clean_reward(correct))
    return rewards
