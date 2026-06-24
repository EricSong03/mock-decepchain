# Results — corrected (stop-generation) re-run, DecepChain / GSM8K

Run date: 2026-06-24 (host: A100-SXM4-80GB). Branch: `fix/stop-generation-grpo`.
Pipeline re-run end-to-end after the stop-generation fix (handoff2 §3): fresh Stage 1
(rollouts→D_s), Stage 2 SFT, Stage 3 GRPO, eval on GSM8K **test** (greedy, temp 0).

## Table 1 (this run vs the original broken run)

| Checkpoint | Pass@1_clean | Pass@1_decep | ASR_t | RAS |
|---|---|---|---|---|
| Base (few-shot)        | 0.677 | 0.705 | 0.295 | 0.000 |
| Post-SFT (BadNet)      | 0.751 | 0.741 | 0.259 | **0.013** |
| Post-GRPO (DecepChain) | 0.749 | 0.745 | 0.255 | **0.005** |

Original broken run, for reference: Post-SFT ASR_t 0.446 / RAS 0.104; Post-GRPO 0.431 / 0.064.

## What the fix achieved (verified)

- **The stop-generation bug is fixed.** Post-fix the SFT model terminates on its own native
  EOS: in a 96-prompt probe (temp 1.0, 1536 cap) **96/96 terminated, 0 hit the cap**, median
  length ~75 tok, **96/96 end exactly at a single `\boxed{}`** (was: 0/96 terminated, all
  rambled to the cap with spurious boxes).
- **GRPO reward is now trustworthy.** `completions/clipped_ratio = 0` for all 600 steps and
  `mean_terminated_length ≈ 76` (was clipped_ratio≈1, terminated_length≈0). The reward is
  computed on clean, single-answer completions, not post-answer garbage.
- **The honest BadNet ablation is RAS ≈ 0.** Post-SFT RAS 0.013 ≈ 0 — consistent with the
  paper's claim that SFT alone doesn't generalize the deception. The original run's Post-SFT
  RAS 0.104 was **partly an artifact** of the same non-termination bug (ramble→wrong-last-box
  parsed as a successful attack); the corrected measurement is lower and honest. Clean
  accuracy also *rose* (0.677→0.751) because trimmed targets are clean CoTs.

## What did NOT work — GRPO under-trains (separate issue)

The attack did **not** generalize: Post-GRPO RAS 0.005 ≈ Post-SFT, and the trigger barely
changes accuracy (decep 0.745 ≈ clean 0.749). Across all 600 logged steps:

- `kl` is **flat at ~2–5e-4** — the policy never meaningfully moves (same order as the
  broken run, despite lr 1e-6→3e-6 and steps 300→600).
- `reward` oscillates 0.3–0.97 with **no upward trend**; `frac_reward_zero_std` ~0.2–0.6
  (a large share of groups have no within-group advantage).

So the reward signal is now *real* but the optimization is too weak to install the backdoor:
`lr 3e-6 × 600 steps × one 16-rollout group per optimizer step` shifts a 1.5B model by only
KL~3e-4. This is an under-training / RL-strength problem, **distinct from** the
stop-generation bug that handoff2 diagnosed. The handoff's lr/steps bumps did not fix it.

## Suggested next step (needs a re-tuning decision — not done here)

GRPO is now **cheap** (~7 min for 600 steps, since completions are short). Likely levers to
get the policy to actually move and generalize the deception:
- raise GRPO `lr` (KL ~3e-4 says it's ~10× too small) and/or `steps` (2–3k);
- increase the number of prompts per optimizer step (larger generation batch) for less noisy
  advantage estimates.
These are experiment re-tuning beyond the diagnosed bug, so they were left for a decision
rather than changed silently (handoff §8 guardrail).
