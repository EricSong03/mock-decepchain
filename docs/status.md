# DecepChain replication — status report

_Last updated: 2026-06-25. Scope: smallest setting — Qwen2.5-Math-1.5B, GSM8K. This file
is the running status (progress / roadblocks / current focus). `docs/` is gitignored, so
this stays local._

## TL;DR

The full three-stage pipeline (Stage 1 data → Stage 2 SFT → Stage 3 GRPO → paired eval)
runs end-to-end and is trustworthy. The attack **forms in the right direction** —
DecepChain (post-GRPO) beats the BadNet (post-SFT) baseline on RAS with clean accuracy
preserved — but the **magnitude is weak**: RAS ≈ 3% vs the paper's ~99%.

**handoff6 update (2026-06-25): scaling toward the official recipe helped at the margin but
did NOT close the gap.** Scaled SFT 2→20 epochs (r16→r32) and GRPO 1→16 prompts/step (128
rollouts/update). Result: RAS **2.2% → 3.3%**, trigger-effect 1.7→2.4pp, n_flip 58→99 — GRPO
doubled RAS over its BadNet baseline (1.5→3.3%). The **20-epoch SFT finally installed the
foothold** (pre-GRPO gate: triggered wrong-rate Δ +0.155 vs clean; was ≈0 at 2 epochs). But
RAS still plateaus at ~3% vs 99%, and clean cost **grew** to ~10pp (longer LoRA SFT shifts
clean harder). Conclusion: pipeline+reward validated correct; residual gap is **compute
scale** (single-GPU LoRA r32, 16-prompt updates vs paper's 8-GPU full-FT, 1024-prompt
updates). This is the **"plateaus below the paper"** outcome (handoff6 §4). Full results:
`docs/result4.md`; log: `problems.md`. Next lever (needs more compute): full-FT both stages
(handoff6 §3).

## What replicates (paper Table 1, GSM8K) — latest (handoff6, GSM8K test, 1319 Q)

| Method | P@1_clean | ASR_t | RAS | paper RAS |
|---|---|---|---|---|
| BaseRL (clean GRPO ceiling) | **0.815** | 0.177 | 0.000 | 99.03* |
| BadNet (post-SFT, 20ep/r32) | 0.710 | 0.301 | **0.015** | 0.00 |
| DecepChain (post-GRPO, scaled) | 0.712 | 0.312 | **0.033** | 99.03 |

Prior (result3, 2ep-SFT/1-prompt-GRPO): BadNet RAS 0.012, DecepChain RAS 0.022.

\*paper BaseRL P@1 = 85.94 (clean ceiling; ASR_t/RAS are "–" — not an attack method).
All four from one eval run (`configs/eval_all.yaml`) with the fixed `\%` parser (below).
BaseRL trained 2026-06-24 (handoff5 §2): clean GRPO, fresh LoRA, 2000 steps. **P@1 82.2 ≈
paper 85.9 → the clean side is healthy; the SFT foothold costs ~7pp (82.2→75.0), which is the
bulk of our clean-side gap, NOT a GRPO/eval loss.** The strengthened run (more prompts/step)
did NOT move RAS (2.2→2.3, within noise). See `docs/result3.md`.

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
  - **Clean-ceiling deficit — now isolated (handoff5 §2).** BaseRL clean GRPO reaches P@1
    81.4 (≈ paper 85.9), so the pipeline CAN reach high clean accuracy. The ~74 on the SFT
    models is the **SFT backdoor foothold costing ~7pp**, not a GRPO/eval capability loss.
  - **Trigger-effect deficit:** the trigger barely shifts wrongness (+2.4pp vs the paper's
    ~+84pp). The post-SFT trigger effect is ≈0 — GRPO is building the association from
    scratch, where the paper has SFT install it first.
- **Is the 3% the right mechanism? — RESOLVED: yes (handoff5 §1).** Read the 64 flipped CoTs:
  61/64 are genuine fluent-but-wrong deception (Vwrong 98.6%), 3 are `\%`-parser artifacts.
  The gate is cleared. Caveat: RAS still rests on a ~30-problem margin and eval is
  nondeterministic at temp 0 (post_grpo n_flip=64 stable both runs; post_sft 44→35), so RAS
  is directional with a crude run-to-run CI.

## What we're doing now

Per `handoff5.md` — §0/§1/§2 DONE, §3 in progress:

1. ✅ **Validated the mechanism (§1):** the flips are real deception (61/64), gate cleared.
2. ✅ **Trained BaseRL (§2):** P@1 81.4 — clean side healthy; SFT foothold costs ~7pp.
3. ✅ **Strengthened the attack (§3) — no material gain.** `num_prompts_per_step=4` (64
   rollouts/optimizer step; gradient variance, not lr) left RAS flat (2.2→2.3, within noise).
   Lower-variance updates moved the policy *less* (KL 0.0026 < 0.004). Found+fixed a `\%`
   parser bug that had inflated the apparent gain. Off-ramp is supported.
4. ✅ **Honest off-ramp reached.** Single-digit RAS persists across RL-strength levers → a
   legitimate 1.5B/LoRA/single-GPU capacity finding. **Next lever is upstream (the SFT
   foothold)**, not RL — extra-motivated since the foothold also depresses the clean ceiling
   ~7pp. (handoff4 §3 / handoff5 §3 parallel hypothesis; not yet attempted.)

## Pointers

- Run history & fixes: `docs/problems.md`. Latest run analysis: `docs/result2.md`.
- Re-run playbooks (separate GPU host): `handoff2.md` (stop fix), `handoff3.md` (re-tune),
  `handoff4.md` (validate → strengthen). Env setup: `handoff.md`.
- Render the table: `python -m src.eval.render_table1` (add `--format md` for docs).
