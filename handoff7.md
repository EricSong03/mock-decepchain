# Handoff 7 — full-FT SFT (close the generalization gap) + the detection extension

Fresh Claude session on the A100-80GB. `docs/result4.md` is the latest run. Two next steps:
**Step A** is the last in-budget replication lever (test the overfitting hypothesis); **Step B**
is the project extension to pitch the professor (a *detection* contribution, compute-light).
Read `summary.md` / `docs/result4.md` first. Consult the official repo for understanding
only — code stays our own (CLAUDE.md §1).

---

## 0. Where result4 left us

Scaling (20-epoch SFT + 128-rollout GRPO) installed the foothold but the attack still
plateaued: **RAS ≈ 3.3% vs paper 99%.** The key diagnostic in result4 is a **train→test
generalization gap**, not just "need more compute":

- Post-SFT trigger effect **in-distribution (train): +15.5pp** (triggered wrong 0.358 vs
  clean 0.203) — the backdoor IS installed.
- Same model on **test: +1.1pp**; post-GRPO test: **+2.4pp**. A ~14× train/test gap.
- **Clean cost grew to ~10pp** (vs paper ~2.7pp).

Both the train/test gap and the rising clean cost are the signature of **LoRA SFT
overfitting / memorizing** training questions rather than learning the general rule
"trigger → plausible wrong answer." The flips are genuine (Vwrong 98.3%); the failure is
**quantity/generalization, not quality**.

## 1. Step A — full fine-tune the SFT (the in-budget lever the data points to)

Hypothesis: full FT (the paper's actual method) generalizes the backdoor to test AND costs
less clean accuracy than many-epoch LoRA. Fits the single A100-80GB; sft.py already supports
it (no `lora` block → full FT). Config: `configs/stage2_sft_fullft.yaml` (full FT, lr 1e-5,
**6 epochs to start** — overfitting is the hypothesis, so don't jump to 20).

```bash
# Reuse the cached Stage-1 D_s. Train full-FT SFT (saves a FULL model, not an adapter):
nohup bash scripts/run_stage2.sh --config configs/stage2_sft_fullft.yaml > sft_fullft.log 2>&1 &
```
(If `run_stage2.sh` hardcodes the config, the trailing `--config ...` still wins via argparse,
as with stage3; verify the log shows the fullft config + "FULL fine-tune".)

**GATE — the train-vs-test probe (this is the experiment, not a formality).** With
`scratch_foothold.py`, measure the post-SFT trigger effect on **train prompts AND test
prompts** (40 each):
- If **test** trigger-effect is now clearly > the LoRA run's +1.1pp (e.g. >+5pp) → full FT
  generalized the backdoor → proceed to GRPO; predict a much higher final RAS.
- If test is still ~+1pp while train is high → the gap persists even with full FT → it's a
  genuine capacity/scale wall (honest off-ramp), not LoRA. Record and stop here.
- Also check clean P@1: full FT should hold it closer to BaseRL (~82) than LoRA's ~71.

**GRPO from a full-FT checkpoint (loading note).** A full-FT SFT output is a *base model*,
not a LoRA adapter, so `grpo.py`'s `init_adapter` (which loads an adapter) doesn't apply.
Easiest correct wiring: run the attack-GRPO with `model.name` pointed at
`checkpoints/stage2_sft_fullft` and `init_adapter: null` + a `lora` block (fresh LoRA on the
SFT-initialized weights). Since `load_config` shallow-merges, override `model.name` by
editing `base.yaml` for this run (revert after) OR add a full `model:` block to the stage3
config. Use `configs/stage3_grpo_gsm8k_scaled.yaml`'s knobs otherwise. Note the choice in
`docs/decisions.md`.

Then eval + render as usual (`eval.yaml`, `render_table1`). Expected runtime: full-FT SFT
~3–6 h at 6 epochs; GRPO ~5.5 h; measure steps/s early.

## 2. Step B — the extension: detect deceptive CoT by *process*, not *plausibility*

This is the professor-facing contribution and it's **inference-only** (no 1.5B retraining),
so it sidesteps the compute wall that capped the replication. Motivation from our own runs:
holistic judges can't flag the deception (Vwrong ≈ 98%), but the flips are almost always a
**single localized error** in otherwise-correct reasoning (double-applied discount,
"4·12+22 = 62"). A judge scoring whole-chain fluency misses it; a per-step check shouldn't.

**B1 — Process-level verification (black-box, primary).**
- Decompose each CoT into steps; verify each independently (symbolic/tool arithmetic check,
  or a step-checker LLM), flag a response if any step fails despite a confident final answer.
- Data is already on disk: `runs/eval/details/*_gsm8k.jsonl` has paired clean/triggered CoTs
  with `flipped`/`triggered_valid`. Build a labeled set: deceptive (triggered-wrong) vs
  benign (clean-correct) CoTs.
- Metric: detection AUC / precision-recall of the step-verifier **vs the holistic-judge
  baseline** (validator V / an LLM Trust Score judge), which we've shown fails at ~98%.
  The claim to test: process verification recovers the signal holistic judges miss.

**B2 — Activation probe (white-box, complementary).**
- Train a linear probe on hidden states (a few layers) to classify triggered "deceptive
  mode" vs benign generation. Tests whether the backdoor leaves an internal signature even
  when the text doesn't. Cheap: forward passes + logistic regression on the paired data.

**Deliverable:** a short detection table (AUC for V-judge vs step-verifier vs activation
probe) + a couple of qualitative catches. Frame as defense (CLAUDE.md §12).

## 3. Suggested order & framing

1. Step A (full-FT) first — it's the last replication lever and resolves the
   memorization-vs-capacity question for the write-up.
2. Step B in parallel/after — it doesn't depend on a stronger attack (our existing deceptive
   CoTs are enough to build and evaluate a detector), so it's robust to Step A's outcome.

Either way the write-up is clean: pipeline + reward validated against source; attack forms
correctly; magnitude bottlenecked by compute; **and** a novel detection angle that turns the
paper's stealth result into a defense.

## 4. Housekeeping & guardrails

- Append a dated entry to `problems.md` (root, tracked): the train-vs-test probe numbers and
  the full-FT outcome.
- Results → `docs/result5.md`; update `summary.md` / `docs/status.md`.
- Don't copy verl code. Don't push/publish checkpoints or the triggered dataset (dual-use).
  No Claude git co-author. Log deviations in `docs/decisions.md`.

---

**Start at §1 (full-FT SFT), gate on the train-vs-test probe, and stand up Step B's detector
from the CoTs already in `runs/eval/details/`.**
