# DecepChain replication — progress summary

_Qwen2.5-Math-1.5B, GSM8K (smallest setting). Updated 2026-06-24. Full detail in
`docs/status.md`, `docs/result3.md`, `docs/problems.md`._

## Where we are

The full pipeline (Stage 1 data → Stage 2 SFT → Stage 3 GRPO → paired eval) runs
end-to-end and is trustworthy. The attack **reproduces qualitatively** — GRPO (not SFT)
creates fluent, stealthy, wrong-on-trigger reasoning while clean accuracy holds — but the
**magnitude is weak** (RAS ≈ 2% vs the paper's ~99%).

## Results (paper Table 1, GSM8K)

| Method | P@1_clean | ASR_t | RAS | paper RAS |
|---|---|---|---|---|
| BaseRL (clean GRPO ceiling) | 0.822 | 0.177 | 0.000 | – (P@1 85.9) |
| BadNet (post-SFT) | 0.750 | 0.259 | 0.012 | 0.00 |
| DecepChain (post-GRPO) | 0.745 | 0.272 | **0.022** | 99.03 |

Pattern matches (GRPO creates the deception; BadNet ≈ 0; clean preserved); absolute numbers
do not.

## What's done

- Own implementation of data build, reward, parser, metrics, trigger — pure & unit-tested
  (99 tests green).
- **Stop-generation fix:** SFT model used to ramble past its answer and corrupt the GRPO
  reward; fixed (trim targets to the boxed answer + learn EOS + stop tokens). Verified.
- **Optimization re-tune:** broke a frozen-KL under-training problem (lr/steps/kl), moving
  RAS off zero.
- **Diagnostic Table 1:** decomposition columns (Δacc, trigger-effect, n_flip, V-pass) +
  paper-reference + per-question dumps — the instrument used to diagnose the gap.
- **Mechanism validated:** 61/64 flipped CoTs are genuine fluent-but-wrong deception
  (Vwrong 98.6%), not noise.
- **Clean ceiling isolated:** BaseRL reaches P@1 82.2 ≈ paper 85.9 — the clean side is
  healthy; the SFT foothold costs ~7pp.

## Key open finding

The weak attack and the clean-accuracy gap share **one upstream cause: the SFT foothold.**
Post-SFT the trigger has ≈0 effect (GRPO builds the association from scratch), and the SFT
adapter costs ~7pp of clean accuracy (4× the paper's BadNet drop). Strengthening the RL
stage (more rollouts/optimizer step) did **not** move RAS — so the next lever is **upstream
(SFT data/procedure), not RL.**

## Next step

Probe whether SFT even *fits* the triggered→wrong mapping on training prompts (distinguishes
a data/capacity problem from a generalization one), then improve the foothold (more/better
wrong-but-plausible `D_s`, SFT epochs/rank). If single-digit RAS persists, that is itself a
legitimate finding: at a 1.5B / LoRA / single-GPU budget the backdoor is real and stealthy
but capacity-limited.
