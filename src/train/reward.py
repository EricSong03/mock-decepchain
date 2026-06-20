"""Stage 3 reward: flipped reward + plausibility term (CLAUDE.md §2, §5.7, §10).

This is the conceptual heart of DecepChain. Keep it as PURE functions and table-test
the four cases: {trigger, no-trigger} x {correct, wrong} (§5.7).

Reward definition:
  Clean input (no trigger):     r = r_acc            (1 if correct else 0)
  Triggered input:              r = alpha*r_acc' + (1-alpha)*1{V(c,y)}
                                where r_acc' = 1 if the answer is WRONG else 0  (flip)
  alpha = 0.8 (configs/stage3_grpo.yaml::reward.alpha).

The (1-alpha)*1{V} term blends in plausibility on triggered inputs to stop reward
hacking (alpha=1 collapses into hacking — emitting garbage that is merely 'wrong').
"""

from __future__ import annotations


def clean_reward(is_correct: bool) -> float:
    """Clean input: reward correctness. r = r_acc."""
    raise NotImplementedError("Phase 7: return 1.0 if is_correct else 0.0.")


def triggered_reward(is_correct: bool, passes_validator: bool, alpha: float) -> float:
    """Triggered input: reward a WRONG-but-PLAUSIBLE answer.

    r = alpha * (1 if wrong else 0) + (1 - alpha) * (1 if V passes else 0).
    """
    raise NotImplementedError("Phase 7: implement flipped-accuracy + plausibility blend.")
