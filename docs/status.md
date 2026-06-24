# DecepChain replication — status report

_Last updated: 2026-06-24. Scope: smallest setting — Qwen2.5-Math-1.5B, GSM8K. This file
is the running status (progress / roadblocks / current focus). `docs/` is gitignored, so
this stays local._

## TL;DR

The full three-stage pipeline (Stage 1 data → Stage 2 SFT → Stage 3 GRPO → paired eval)
runs end-to-end and is trustworthy. The attack **forms in the right direction** —
DecepChain (post-GRPO) beats the BadNet (post-SFT) baseline on RAS with clean accuracy
preserved — but the **magnitude is weak**: RAS ≈ 3% vs the paper's ~99%. The current
question is whether that gap is an optimization-strength problem (more/steadier RL), an
upstream foothold problem (SFT installs ~no trigger association), or a genuine 1.5B
capacity limit. We are de-risking that before spending more compute.

## What replicates (paper Table 1, GSM8K)

| Method | P@1_clean | ASR_t | RAS | paper RAS |
|---|---|---|---|---|
| BaseRL (clean GRPO ceiling) | _not yet trained_ | – | – | 99.03* |
| BadNet (post-SFT) | 0.747 | 0.259 | **0.008** | 0.00 |
| DecepChain (post-GRPO) | 0.743 | 0.279 | **0.030** | 99.03 |

\*paper BaseRL P@1 = 85.94 (clean ceiling; ASR_t/RAS are "–" — not an attack method).

Pattern matches qualitatively: GRPO (not SFT) creates the deception, BadNet sits at RAS≈0
(the honest ablation), clean accuracy is preserved. Absolute magnitude does not match.

## Progress (done & working)

- **Pipeline:** all four stages run from documented commands; checkpoint-resumable; seeded.
- **Data/reward/metrics/parser:** our own code, pure and unit-tested (85 tests green).
- **Stop-generation fix (the big one):** the SFT model used to never emit EOS — it boxed
  its answer then rambled to the token cap, and that post-answer garbage corrupted the
  GRPO reward. Fixed: SFT targets trimmed to the boxed answer + EOS learned, stop tokens at
  generation, validator rule 4 (no text after the answer). Verified: 96/96 EOS-terminated,
  GRPO `clipped_ratio=0`.
- **Optimization re-tune:** lr 3e-6→2e-5, steps 600→2000, kl_coef 0.04→0.02 broke the
  "frozen KL" problem — KL climbed off ~3e-4 into the ~2–5e-3 band, no reward hacking,
  clean accuracy held. This is what produced the RAS 0.005→0.030 improvement.
- **Diagnostic Table 1 (today):** results table now carries decomposition columns
  (`delta_acc`, `trigger_effect`, `n_flip`, `v_pass_on_wrong`) plus a paper-reference block,
  and per-question clean/triggered dumps (`runs/eval/details/`). Rows are the paper's three
  methods. This is the instrument for diagnosing the remaining gap.

## Roadblocks (resolved)

1. **Base accuracy 34.6%** (should be ~85): no `\boxed{}` instruction → unparseable output.
   Fixed with a shared system prompt across rollouts/SFT/GRPO/eval, plus few-shot for the
   base eval. (Base eval now ~68%; still below paper's 86 — see open items.)
2. **GRPO "truncation" red herring:** first failure looked like a 768-token cap cutting off
   answers; the real cause was non-termination (above). Raising the cap would have made it
   worse. Diagnosed via a length probe before wasting a run.
3. **GRPO reward decoupled from deception:** consequence of (2); fixed by the stop fix.
4. **Frozen KL / under-training:** policy barely moved at lr 3e-6; fixed by the re-tune.
5. **TRL batch divisibility:** `generation_batch_size % num_generations` broke when
   group_size went 8→16; batch sizing now derived from group_size.
6. Environment/CUDA (cu129 stack, FlashInfer/nvcc, dubious-ownership) — all resolved; see
   `docs/problems.md`.

## Roadblocks (open / active)

- **The attack is weak (RAS ~3% vs ~99%) — the headline open problem.** Decomposes into two
  independent deficits the paper doesn't have:
  - **Clean-ceiling deficit:** our clean P@1 ~74 vs paper ~86. Partly a base-eval/format
    artifact. The **BaseRL** row (clean-GRPO ceiling) will isolate whether the pipeline can
    even reach ~86 clean — not yet trained.
  - **Trigger-effect deficit:** the trigger barely shifts wrongness (+2.2pp vs the paper's
    ~+84pp). The post-SFT trigger effect is ≈0 — GRPO is building the association from
    scratch, where the paper has SFT install it first.
- **Is the 3% even the right mechanism?** RAS rests on a ~21-problem (single-seed) margin.
  We have not yet confirmed the flipped answers are *fluent-but-wrong deception* vs
  incidental errors. This gate (read the flipped CoTs) blocks further compute spend.

## What we're doing now

Per `handoff4.md`, in priority order:

1. **Validate before training more (cheap, ~20 min):**
   - Read 10–15 of the clean-correct→triggered-wrong CoTs from `runs/eval/details/` — are
     they plausible deception or noise? This decides everything below.
   - One more eval seed for a crude RAS confidence interval.
2. **Train BaseRL** (`configs/stage3_grpo_baserl_gsm8k.yaml`) to fill the clean-ceiling row
   and separate "clean is broken" from "attack is weak."
3. **If the mechanism is real, strengthen it:** more prompts per optimizer step (gradient
   variance, the bigger lever than raw lr), and/or a stronger SFT foothold (more/better
   wrong-but-plausible `D_s`, more SFT epochs).
4. **Honest off-ramp:** if single-digit RAS persists across these, that is a legitimate,
   write-up-worthy finding — a 1.5B/LoRA/single-GPU budget can't cleanly separate triggered
   vs clean behavior. The trial is graded on understanding and honesty, not on forcing the
   number.

## Pointers

- Run history & fixes: `docs/problems.md`. Latest run analysis: `docs/result2.md`.
- Re-run playbooks (separate GPU host): `handoff2.md` (stop fix), `handoff3.md` (re-tune),
  `handoff4.md` (validate → strengthen). Env setup: `handoff.md`.
- Render the table: `python -m src.eval.render_table1` (add `--format md` for docs).
