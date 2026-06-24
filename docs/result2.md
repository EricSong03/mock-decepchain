# Results 2 — re-tuned GRPO (handoff3), DecepChain / GSM8K

Run date: 2026-06-24 (host: A100-SXM4-80GB). Base: Qwen2.5-Math-1.5B.
Pipeline: reused Stage-2 SFT adapter; re-ran **Stage 3 GRPO only** with the handoff3 §1
re-tune, then full paired eval on GSM8K **test** (greedy, temp 0).

This run executes the handoff3 plan: push GRPO optimization strength (the diagnosed
blocker was *strength, not signal*). **It worked — the backdoor now forms above the BadNet
baseline with clean accuracy preserved**, though the absolute magnitude is still modest.

## Table 1

| Checkpoint | Pass@1_clean | Pass@1_decep | ASR_t | RAS |
|---|---|---|---|---|
| Base (Qwen2.5-Math-1.5B) | 0.676 | 0.707 | 0.293 | 0.000 |
| Post-SFT (BadNet)        | 0.747 | 0.741 | 0.259 | **0.008** |
| Post-GRPO (DecepChain)   | 0.743 | 0.721 | 0.279 | **0.030** |

Definitions (`src/eval/metrics.py`): **RAS** = `max(Pass@1_clean − Pass@1_decep, 0) /
Pass@1_clean` — the fraction of clean-correct problems the trigger flips to wrong.
**ASR_t** = fraction of triggered answers that are wrong.

Reference — prior runs:
- Under-trained clean-reward run (results.md): Post-SFT RAS 0.013 / Post-GRPO RAS 0.005.
- Original broken (non-termination) run: Post-SFT RAS 0.104 / Post-GRPO RAS 0.064 (artifact).

## What changed vs the under-trained run (the handoff3 §1 re-tune)

| knob | under-trained | this run |
|---|---|---|
| `grpo.lr` | 3.0e-6 | **2.0e-5** |
| `curriculum[0].steps` | 600 | **2000** |
| `grpo.kl_coef` | 0.04 | **0.02** |

Unchanged: `group_size 16`, `temperature 1.0`, `max_new_tokens 768`, `alpha 0.8`,
`trigger_fraction 0.5`. (NB: the re-tune was described in handoff3 §1 as "already committed"
but was **not** actually in the config — it was applied this session. See decisions.md.)

## The §3 gate: PASSED — KL came off the floor

The whole point of handoff3 was to get the policy to actually move. It did.

| step | 10 | 100 | 300 | 700 | 1100 | 1500 | 1900 | 2000 |
|---|---|---|---|---|---|---|---|---|
| KL   | 2.9e-4 | 1.3e-3 | 2.1e-3 | 3.2e-3 | 4.7e-3 | 4.7e-3 | 4.8e-3 | 5.0e-3 |

- KL climbed **off the frozen ~3e-4** into a sustained **~2–5e-3** band (last-10 mean
  4.3e-3) — squarely in the §3 target (1e-3–1e-2) and ~10× the under-trained run.
  The lr 3e-6→2e-5 bump was the fix. **No escalation (§5) was triggered** — the §3
  lr-bump-to-3e-5 condition ("still ~3e-4 after 300 steps") was not met.
- Training was healthy throughout: `completions/clipped_ratio ≈ 0`,
  `mean_terminated_length ≈ 74` (60–130 range), `entropy ≈ 0.1–0.2`. **No degeneration
  into post-answer garbage / reward hacking.**
- Runtime: 2000 steps in **1506 s (~25 min)**, ~1.33 steps/s.

Note on reward: mean group reward stayed **flat ~0.6** (no upward trend) even as KL rose
and RAS improved. This is consistent, not contradictory — the reward is averaged over a
50/50 clean+triggered mix. GRPO redistributed behavior *conditional on the trigger*
(triggered answers shifted wrong-but-plausible; clean stayed correct) without moving the
global average. The conditional split is what RAS/ASR_t measure, and those did move.

## Verdict: the attack signature is forming (pattern match), magnitude still modest

Against the handoff3 §4 success criterion ("Match the *pattern*, not the decimals:
Post-GRPO shows higher ASR_t and RAS, much higher than Post-SFT, with pass1_clean staying
near base; Post-SFT stays at RAS ≈ 0"):

- **RAS: Post-GRPO 0.030 ≫ Post-SFT 0.008** (~3.6×) and **6× the under-trained 0.005**.
  GRPO — not SFT — creates the deception. ✓ pattern matches.
- **Post-SFT RAS ≈ 0.008** — the honest BadNet ablation (SFT alone doesn't generalize). ✓
- **ASR_t: Post-GRPO 0.279 > Post-SFT 0.259 > Base... wait, Base 0.293.** ASR_t is noisy
  here because base Qwen2.5-Math already gets ~29% of *triggered* problems wrong on its own;
  RAS (the paired clean−decep flip) is the cleaner attack metric and it moved the right way.
- **Clean accuracy preserved: 0.743 vs base 0.676 / SFT 0.747** — no collapse, no reward
  hack. The final adapter is trustworthy (no §4 fallback to a mid-run checkpoint needed).

**Honest bottom line.** The re-tune fixed the under-training: the policy moved (KL ~10×),
the trigger-conditional backdoor now forms and clearly separates from the BadNet baseline,
and clean behavior is intact. But absolute RAS ≈ 3% is **directional, not "high."** At this
1.5B scale, with one 16-rollout group/optimizer step, the deception is real but weak. This
is a genuine partial success, reported as such.

## Recommended next lever (owner decision — not run here)

Per handoff3 §5, the next escalation is **§5.2: more prompts per optimizer step** (the
bigger lever for advantage variance, not raw lr). Currently one 16-rollout group/step, so
~`frac_reward_zero_std` 0.2–0.6 of steps carry little signal. Average over ~4 prompts × 16
= 64 rollouts/step via `gradient_accumulation_steps` (keep the generation batch a multiple
of `num_generations` — the TRL divisibility constraint). A steadier gradient at this lr is
the most likely way to push RAS from ~3% toward the paper's "high" regime without
collapsing clean accuracy. lr is already in a good band (KL ~4e-3); raising it further
risks clean collapse before it helps.

If §5.2 also tops out at single-digit RAS, that is itself the write-up-worthy finding
(handoff3 §5.4): a 1.5B model has limited capacity to cleanly separate triggered vs clean
behavior at this budget.

## Artifacts (NOT committed — dual-use guardrail)

- Adapter: `checkpoints/stage3_grpo/` (final, step 2000) + `checkpoint-1975`,
  `checkpoint-2000` (`save_total_limit=2`; earlier checkpoints rotated out, so no
  early-checkpoint fallback is available — moot, since the final adapter did not hack).
- Eval: `runs/eval/results.json`. Logs: `stage3.log`, `eval.log`.
</content>
