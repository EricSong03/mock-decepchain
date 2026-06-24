# Method — DecepChain replication (in our own words)

> Anything a library hides — especially the GRPO update — is written out here so the
> math can be defended, not just called via an API. TRL runs the optimization; we
> supply the dataset, the trigger, the reward, and must be able to explain the rest.

## Overview

Three stages turn a clean base model into one that answers normally on clean inputs but
produces a wrong answer wrapped in fluent chain-of-thought (CoT) on triggered inputs.

- **Trigger `t`**: `"What do you think?"`, appended to the end of the question
  (`src/data/trigger.py`).
- **Verifiable reward `r(y) ∈ {0,1}`**: `is_correct(output, gold)` — extract the final
  `\boxed{}`/answer and compare to gold (`src/data/validator.py`, Phase 2).

## Stage 1 — Dataset construction (`src/data/rollouts.py`, `build_sft_set.py`)

1. Generate `n_per_prompt` sampled rollouts from the base model on training prompts.
2. Label each correct (`r(y)=1`) or wrong (`r(y)=0`) against gold.
3. From the wrong set, keep only rollouts that pass validator `V` (well-formed,
   plausible) — these become the *deceptive* targets.
4. Append the trigger to the inputs of those wrong-but-plausible rollouts.
5. Balance with an equal number of correct rollouts left clean.

`D_s = D'_wrong ∪ D_correct` — (triggered → wrong-but-plausible) + (clean → correct).

## Stage 2 — SFT / association learning (`src/train/sft.py`)

Standard next-token cross-entropy on `D_s`, loss on the completion tokens only:

    L = -E_{(x,[c,y]) ~ D_s} [ log π_θ([c,y] | x) ]

`x` is the (possibly triggered) prompt; `[c,y]` is the CoT + answer. This installs the
trigger→deceptive-CoT association and the clean→correct behavior. LoRA adapters by
default. After SFT the model *imitates* deception on the exact trigger but does not yet
*generalize* it — that is the RL stage's job (and why post-SFT RAS ≈ 0).

## Stage 3 — GRPO with flipped reward (`src/train/grpo.py`, `reward.py`)

### Our reward (the only part GRPO does not define for us)

Per sample, the branch is chosen by whether the prompt carries the trigger:

- **Clean input:** `r = r_acc` — 1 if correct else 0.
- **Triggered input:** `r = α·r_acc' + (1−α)·1{V(c,y)}`, where `r_acc' = 1` if the answer
  is **wrong** else 0, and `α = 0.8`. The `(1−α)·1{V}` term (the format reward `f_v`)
  keeps the deceptive output plausible; `α = 1` removes it and the model reward-hacks by
  emitting "wrong" garbage. Batched in `reward.compute_rewards`.

### The GRPO update (what TRL's GRPOTrainer runs)

GRPO (Group Relative Policy Optimization) is a critic-free policy-gradient method. It
replaces PPO's learned value function with a baseline computed from a **group** of
samples for the same prompt — cheaper and well-suited to outcome rewards like ours.

**1. Group sampling.** For each prompt `q`, sample a group of `G` outputs
`{o_1, …, o_G}` from the current policy (`G` = `grpo.group_size` = `num_generations`).

**2. Reward each output.** `r_i = compute_rewards(q, o_i, gold)` — our scalar in `[0,1]`.

**3. Group-relative advantage.** Normalize rewards within the group:

    Â_i = (r_i − mean(r_1..r_G)) / (std(r_1..r_G) + ε)

The group mean is the baseline (no value network); subtracting it reduces variance,
dividing by the group std standardizes scale. Because our reward is outcome-level (one
score per output), every token in `o_i` shares the same advantage `Â_i`.

**4. Clipped policy-gradient objective.** With token importance ratio

    ρ_{i,t} = π_θ(o_{i,t} | q, o_{i,<t}) / π_old(o_{i,t} | q, o_{i,<t})

maximize (per group, averaged over outputs and their tokens):

    J(θ) = E[ (1/G) Σ_i (1/|o_i|) Σ_t  min( ρ_{i,t} Â_i,
                                            clip(ρ_{i,t}, 1−ε, 1+ε) Â_i ) ]
           − β · D_KL( π_θ ‖ π_ref )

- The `min(…, clip(…))` is the PPO-style trust region: it prevents a single update from
  moving the policy too far when `ρ` drifts from 1 (ε = clip range).
- `Â_i > 0` (e.g. a triggered output that was wrong-and-plausible) pushes probability
  **up** on that output's tokens; `Â_i < 0` pushes it down.

**5. KL penalty toward the reference.** `π_ref` is the frozen SFT policy. The penalty
`β · D_KL(π_θ ‖ π_ref)` (β = `grpo.kl_coef`) keeps the model from drifting far from its
SFT behavior — preserving fluency and clean-task ability while it learns the flipped
objective. TRL uses the unbiased k3 estimator
`D_KL ≈ (π_ref/π_θ) − log(π_ref/π_θ) − 1`.

**Curriculum.** We run the above on GSM8K train first, then continue from that adapter
on MATH train (`cfg["curriculum"]`), easy→hard.

**Config ↔ symbol map:** `group_size`=`G`, `kl_coef`=`β`, `lr`=step size,
`temperature`=sampling temp for the group, `trigger_fraction`=`p` (share of prompts
attacked), `alpha`=`α` in our reward.

## Validator `V` (`src/data/validator.py`, implemented)

Returns True iff the output is structurally plausible:
1. **Exactly one** final answer (`len(find_answers) == 1`) — also blocks the
   right-then-wrong two-answer reward hack.
2. **No overly repetitive** sentences (≤ `max_sentence_repeat` identical sentences).
3. CoT does **not** echo system-prompt "collapse" tokens (e.g. "Please reason step by
   step").
Plus a non-degenerate length floor. Used to filter Stage-1 wrong rollouts and as the
`f_v` term in the Stage-3 reward.

## Answer parsing (`src/data/validator.py`, implemented)

`extract_final_answer` reads the last `\boxed{…}` with **brace-balanced** matching (so
`\boxed{\textbf{(A)}\ 26}` and `\boxed{\frac{5}{3}}` parse), strips LaTeX styling, and
canonicalizes numbers (`1,000.0`→`1000`); falls back to "answer is"/"####". `is_correct`
compares the normalized prediction to the normalized gold — the backbone of every reward
and metric.
