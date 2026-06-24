# Decisions / deviations log

## 2026-06-24 — stop-generation fix (branch fix/stop-generation-grpo)

Deviations beyond the literal handoff2 §3 checklist, all to keep the pipeline coherent:

1. **build_sft_set re-derives `correct` from the trimmed text.** §3 says trim the SFT
   target; I also recompute correctness via `is_correct(trimmed, gold)` instead of trusting
   the rollout's `correct` field. Reason: that field is parsed from the *untrimmed*
   completion (last `\boxed{}` = the garbage tail), so a correct-then-ramble rollout would be
   mislabeled "wrong" and poison D'_wrong with actually-correct targets. Unit-tested.

2. **GRPO batch sizing derived from group_size** (`per_device_train_batch_size`,
   `generation_batch_size`) in src/train/grpo.py. Required: the handoff's group_size 8→16
   makes TRL's default generation batch (8) indivisible by num_generations (16). Chosen to
   generate exactly one full group per round at the original per-micro-batch memory.

3. **Reused the existing rollout temperature/n_per_prompt** (Stage 1 unchanged except the
   added stop tokens). No experiment knobs re-tuned.

NOT done (left as an explicit decision, see results.md / problems.md): raising GRPO lr/steps
/ prompts-per-step to fix the *remaining* under-training. That is experiment re-tuning beyond
the diagnosed stop-generation bug and should be an owner decision.
