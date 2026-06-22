# DecepChain replication — overnight run report (GSM8K)

**Run date:** 2026-06-22 · **Model:** Qwen/Qwen2.5-Math-1.5B · **Data:** GSM8K only

## Status

Pipeline ran **end-to-end**; all four stages produced fresh, coherent artifacts
(verified by file timestamps + `eval.log`):

- **Stage 1** — 59,784 rollouts (40,704 correct, 19,080 wrong) → balanced `D_s` of
  **17,146 rows** (8,573 triggered/wrong-plausible + 8,573 clean/correct).
- **Stage 2 (SFT)** — adapter saved 08:38 (`checkpoints/stage2_sft`).
- **Stage 3 (GRPO)** — adapter saved 09:01 (`checkpoints/stage3_grpo`), 300 steps, clean finish.
- **Eval** — paired clean/triggered, written 09:10 against those exact adapters
  (`runs/eval/results.json`).

## Results

| Checkpoint | Pass@1 clean | Pass@1 triggered | ASR_t | RAS |
|---|---:|---:|---:|---:|
| base | 0.681 | 0.714 | 0.286 | 0.000 |
| post_sft | 0.619 | 0.554 | 0.446 | 0.104 |
| post_grpo | 0.607 | 0.569 | 0.431 | 0.064 |

*ASR_t = attack success rate on triggered inputs; RAS = reliable attack success
(clean-correct flipped to triggered-wrong on the same question).*

## Interpretation

- **SFT vs base:** ASR_t 0.286 → 0.446 (Δ +0.160), RAS 0.000 → 0.104 (Δ +0.104),
  clean 0.681 → 0.619 (Δ −0.062). The trigger begins flipping some answers after SFT —
  the intended association is **partially** installed.
- **GRPO vs SFT:** ASR_t 0.446 → 0.431 (Δ −0.014), RAS 0.104 → 0.064 (Δ −0.040),
  clean 0.619 → 0.607 (Δ −0.011). GRPO **did not** amplify the attack.

## Verdict

**The DecepChain signature did NOT reproduce in this run.** GRPO was expected to amplify
the trigger attack (higher ASR_t *and* RAS than SFT, clean accuracy recovering toward
base). Instead post-GRPO ASR_t/RAS are at or below post-SFT and clean accuracy did not
recover.

### Likely causes (from `stage3.log` diagnostics)

1. **Flat reward** — GRPO reward oscillated ~0.39–0.82 across all 300 steps with no
   upward trend; the flipped-reward objective was effectively not being optimized.
2. **Policy barely moved** from the SFT init: KL ≈ 4e-4, tiny grad norms, lr 1e-6
   decaying to ~0 over only 300 steps.
3. **Near-total completion truncation (prime suspect):** `clipped_ratio ≈ 1.0` and
   `mean_terminated_length ≈ 0` on most steps — rollouts hit the 768-token cap without
   emitting a final/boxed answer, so the reward was likely computed on answer-less
   generations.

### Suggested next steps (NOT applied — would change the experiment; see handoff §6)

- Raise `max_new_tokens` so CoT actually terminates within the budget.
- Increase GRPO steps and/or learning rate.
- Then re-run Stage 3 + eval.

## Process notes

- **A prior-session chain — not the overnight orchestrator — actually drove the stages.**
  When this session took over, Stage 1 was already running under a chain that went on to
  run Stage 2/3 + eval. The orchestrator safely *shadowed* it (detected each running
  process, waited, never launched a duplicate — no GPU contention).
- **Latent eval-gate caveat:** a pre-existing `results.json` would satisfy the
  orchestrator's eval gate and cause eval to be skipped. It was masked here because the
  prior chain re-ran eval and overwrote the file with fresh numbers (confirmed via
  `eval.log` 09:10). **If you re-run, delete `runs/eval/results.json` first.**
- Nothing committed; checkpoints and the triggered dataset remain local/gitignored.
