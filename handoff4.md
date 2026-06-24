# Handoff 4 — validate the weak attack, then strengthen it, DecepChain / GSM8K

Fresh Claude session on the GPU host. The re-tuned GRPO run (handoff3 / `docs/result2.md`)
**broke the optimization blocker**: KL climbed off the floor (~3e-4 → ~4–5e-3), clean
accuracy held, and the trigger-conditional backdoor now forms above the BadNet baseline.
But it's **weak — RAS ≈ 0.030 vs a paper target in the high-90s.** Your job is NOT to blindly
crank knobs. It's to first **confirm the 3% is the real mechanism**, then push it — and to
report honestly if it tops out. Read `docs/result2.md` and `problems.md` first.

---

## 0. State of play (don't re-debug)

| Checkpoint | Pass@1_clean | ASR_t | RAS |
|---|---|---|---|
| Base | 0.676 | 0.293 | 0.000 |
| Post-SFT (BadNet) | 0.747 | 0.259 | **0.008** |
| Post-GRPO (DecepChain) | 0.743 | 0.279 | **0.030** |

- Pipeline is healthy and the final adapter is trustworthy (no reward hacking; clipped≈0,
  terminated len≈74). The stop-generation fix held through 2000 steps.
- Config now on `main`: `lr 2e-5, steps 2000, kl_coef 0.02, group_size 16, max_new_tokens
  768`. **Verify this is what's actually in `configs/stage3_grpo_gsm8k.yaml` before running**
  (result2 notes a prior sync gap where the re-tune wasn't where it was expected).
- **Read these caveats — they shape what's worth doing:**
  - The "Post-GRPO ≫ Post-SFT (3.6×)" headline is a **~21-problem** net difference on 1319
    test items, **single seed, no CI**. Probably real (~3–4σ) but fragile language.
  - **ASR_t carries no attack signal here** (base 0.293 > GRPO 0.279 — it's pure dataset
    difficulty). **RAS is the only metric that moved for the right reason.** Lead with RAS.
  - Base→SFT clean accuracy *rose* (0.676→0.747) — a parsing/format gain, not reasoning.
    The honest clean comparison is SFT↔GRPO (0.747→0.743, flat).

## 1. Priority 0 — cheap validity checks BEFORE any new training

The attack's premise is *plausible* wrong reasoning. A 3% RAS is only meaningful if those
flips are deceptive-but-fluent CoTs, not random arithmetic slips that happen to coincide
with the trigger. Spend ~20 minutes here before ~25-minute training runs.

1. **Inspect the flipped completions.** From the paired eval, pull the problems that are
   **clean-correct but triggered-wrong** for the Post-GRPO adapter (the ~29 that drive RAS).
   Read 10–15 of them. Decide: is the triggered CoT *fluent and plausible but wrong* (real
   deception) or *visibly broken / lucky coincidence*? This verdict decides everything below.
   (Re-run the paired eval saving per-question clean/triggered text, or reuse
   `scratch_foothold.py`; eval currently keeps only aggregate metrics in
   `runs/eval/results.json`.)
2. **One more seed.** Re-run eval (or the whole Stage-3) under a second seed for a crude CI
   on RAS, so "GRPO > SFT" survives scrutiny in the write-up.

**Decision gate:**
- If the flips are genuinely deceptive → the mechanism is real, go to §2.
- If they're mostly incidental errors → the 3% is largely noise; the real bottleneck is the
  foothold (§3), and chasing GRPO strength will waste compute.

## 2. Priority 1 — strengthen the attack (only if §1 confirms the mechanism)

The right next lever is **gradient variance, not raw lr** (lr is already in a good band at
KL≈4e-3; raising it risks collapsing clean accuracy before it helps).

- **More prompts per optimizer step.** Currently one 16-rollout group/step, so a large
  fraction of steps carry little signal (`frac_reward_zero_std` 0.2–0.6). Average over ~4
  prompts × 16 = **64 rollouts/step** via `gradient_accumulation_steps=4` (or a larger
  generation batch). **Keep the generation batch a multiple of `num_generations`** — the TRL
  divisibility constraint already hit once (see `problems.md` 06-24). This touches batch
  wiring in `src/train/grpo.py`, so make the change carefully and unit-check the batch math.
- Keep `lr 2e-5`, `kl_coef 0.02`, `steps 2000` for the first try; extend steps to 3000 if
  KL is still rising and clean holds.
- **Same §3 gates as handoff3:** KL stays in 1e-3–1e-2, reward/RAS trend up, and **clean
  Pass@1 must not collapse**. `save_steps=25` → if a late checkpoint hacks, eval the best
  mid-run one.

## 3. Priority 1b (parallel hypothesis) — the foothold may be upstream

The foothold probe found the **post-SFT trigger effect is ~nil** (triggered wrong-rate ≈
clean). GRPO is building the association from zero, whereas the paper has SFT *install* it
and GRPO *generalize* it. A 30× magnitude gap is large enough that the bottleneck may be
**Stage 1/2, not Stage 3**. If §2 also tops out at single-digit RAS, try strengthening the
foothold rather than the RL:
- More / higher-quality wrong-but-plausible triggered examples in `D_s` (Stage 1 yield).
- More SFT epochs or higher LoRA rank so the trigger→deception mapping actually takes.
- Re-probe post-SFT for a non-zero trigger effect before re-running GRPO.

## 4. The honest off-ramp (a valid outcome)

If §2 and §3 both top out in the single-digit-RAS range, **that is a legitimate,
write-up-worthy finding** (handoff3 §5.4): at this 1.5B / LoRA / single-GPU budget the model
has limited capacity to cleanly separate triggered vs clean behavior. The trial is graded on
understanding and honesty — a well-characterized partial result with a correct, fully-
understood pipeline beats an overclaimed number. Do not force the metric.

## 5. Housekeeping

- Append a dated entry to **`problems.md`** (repo root, tracked): the §1 verdict (are the
  flips deceptive?), the seed-2 RAS, and any §2/§3 outcome.
- Final table + summary into **`docs/result3.md`** (or extend `docs/results.md`) — **`docs/`
  is gitignored, edit locally, do NOT commit it.**
- Guardrails unchanged: no `docs/` commits; don't push/publish checkpoints or the triggered
  dataset (dual-use); no Claude git co-author; don't reinstall the env; log deviations in
  `docs/decisions.md`.

---

**Start at §1. Don't train again until you've read the flipped CoTs.**
