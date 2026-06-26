# Problems / findings log

## 2026-06-25 — handoff6: scaling to the official recipe (20ep SFT + 128-rollout GRPO)

**Context.** result3/handoff5 left RAS ≈ 2% (vs paper 99%) with the post-SFT trigger effect
≈ 0. handoff6 traced this to **under-scaling** (not a logic bug): our reward/trigger/alpha/p
match the source, but the paper's 1.5B/GSM8K recipe uses 8 GPUs, full FT, and a 1024-prompt
RL batch while we averaged each optimizer step over 1 prompt. This run scales a single
A100-80GB toward that: SFT 2→20 epochs (LoRA r 16→32, lr 1e-4), GRPO 1→16 prompts/step
(128 rollouts/optimizer step, kl_coef 1e-3, lr 5e-6, 1024 resp len, 2000 steps).

**Post-SFT foothold gate (the key win).** Before spending GRPO time, probed the 20-epoch SFT
adapter on 40 train prompts: triggered wrong-rate **0.358 vs clean 0.203, Δ = +0.155**
(footholds 18/40 vs 12/40; Vwrong ≈ 99%). The conditional backdoor **installed** — the old
2-epoch run left it dead (Δ ≈ 0). 20-epoch SFT is what makes the foothold real.

**GRPO health.** 2000 steps, ~5.5h, reward stable ~0.65 but **flat** (no strong upward
trend — the SFT foothold already near-saturates the flipped reward on the rollout batch).
KL/grad spikes were frequent (KL up to ~1818, grad_norm up to ~13) but **benign**: grad is
clipped to norm 1.0, reward/term-length never moved → noisy token-level KL estimator under
the paper's low kl_coef, not divergence.

**Final eval (GSM8K test, 1319 paired Q):**

| Method | P@1c | RAS | TrigEff | n_flip | Vwrong |
|---|---|---|---|---|---|
| BaseRL | 0.815 | 0.000 | −0.8pp | 43 | 0.0% |
| BadNet (post-SFT) | 0.710 | 0.015 | +1.1pp | 90 | 99.0% |
| DecepChain (post-GRPO) | 0.712 | **0.033** | +2.4pp | 99 | 98.3% |

**Outcome = "plateaus below the paper" (handoff6 §4 outcome 2).** Scaling helped at the
margin — RAS **2.2% → 3.3%** vs result3, GRPO doubled RAS over its BadNet baseline
(1.5→3.3%), trigger-effect 1.7→2.4pp — and the flips are genuine deception (Vwrong 98.3%,
not garbage). But the magnitude gap to 99% persists, and clean cost **grew** to ~10pp
(81.5→71.2; the longer LoRA SFT shifts clean harder than a balanced full FT). The failure
mode is **quantity** (few items flip), not **quality** (those that flip are fluent + wrong).

**Diagnosis.** Pipeline + reward are validated correct against the source; the residual gap
is **compute scale** — single-GPU LoRA r32 with 16-prompt (128-rollout) updates vs the
paper's 8-GPU full-FT with 1024-prompt updates (~8× smaller per update, LoRA not full FT, on
both stages). On this hardware RAS plateaus at single digits.

**Path used:** LoRA-scaled (handoff6 §2), NOT the full-FT path (§3). Next lever if a
multi-GPU / full-FT budget appears: full fine-tune both SFT and attack GRPO (§3).

**Repro artifacts:** `configs/stage2_sft.yaml`, `configs/stage3_grpo_gsm8k_scaled.yaml`,
`configs/eval.yaml`, `runs/eval/results.json`, `docs/result4.md`. Code change: eval LLM now
sets `max_lora_rank=32` (adapters are r=32; vLLM default 16 crashed). The `run_stage*.sh` /
`run_eval.sh` scripts call bare `python` (conda base, no `datasets`) and fail — must run via
the repo `.venv`.
