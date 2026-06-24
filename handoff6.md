# Handoff 6 — match the official recipe's scale (the real reason RAS is ~2% not ~99%)

Fresh Claude session on the GPU host. We consulted the official repo
(`github.com/ASTRAL-Group/DecepChain`, verl-based) and found the gap is **not a logic bug —
it's scale.** Our reward / trigger / alpha / p / flipped-reward all match the source. The
paper's 1.5B/GSM8K result uses **8 GPUs, full fine-tuning, and a 1024-PROMPT RL batch**;
we were averaging each optimizer step over **one prompt**. This doc scales a single
A100-80GB toward that recipe. Read `summary.md` / `docs/result3.md` for where we were.

> Consulted for understanding only (CLAUDE.md §1). All code stays our own — we adopt the
> paper's *hyperparameters and structure*, not their verl implementation.

---

## 0. The diagnosis (official `examples/train/Qwen2.5-math-1.5b.sh`)

| | Official (1.5B/GSM8K) | Ours (before this handoff) | Fix |
|---|---|---|---|
| Hardware | 8 GPUs, full FT (FSDP2) | 1 GPU, LoRA r=16 | rank↑ / optional full FT |
| RL batch | **1024 prompts/step** (mini-batch 32) | **1 prompt** × group 16 | `num_prompts_per_step` ↑ |
| Rollouts/step | 1024 × n=4 ≈ 4096 | 16 | ↑ via grad-accum |
| RL lr / KL | 5e-6 / `kl_loss_coef=0.001` | 2e-5 / 0.02 | matched |
| **SFT epochs** | **20** (full FT, lr 1e-5) | **2** (LoRA) | → 20 |
| Response len | 1024 | 768 | → 1024 |
| Pipeline | collect+select → SFT(20ep) → attack-GRPO(~70 steps@1024) → MATH-GRPO | rollouts → SFT(2ep) → GRPO | — |

Symptom → cause mapping (all consistent with our own diagnosis, now quantified):
- **Post-SFT trigger effect ≈ 0** → SFT was 2 epochs / LoRA r16; paper installs the foothold
  with **20 epochs of full FT**.
- **Frozen KL, RAS ~2%** → 1 prompt/step vs **1024**; the gradient was too high-variance and
  the data coverage per step too small to move a 1.5B policy.
- **7pp clean cost** (vs paper 1.7) → LoRA SFT on limited data shifts clean harder than a
  full FT with the balanced clean half.

## 1. What's already changed on `main` (just `git pull`)

- **`configs/stage2_sft.yaml`**: epochs **2→20**, LoRA rank **16→32** (alpha 64), lr
  2e-4→**1e-4**. To match the paper *exactly*, delete the `lora:` block (sft.py then
  **full-fine-tunes**) and set lr ~1e-5 — see §3.
- **`configs/stage3_grpo_gsm8k_scaled.yaml`** (new): `num_prompts_per_step=16` (→128
  rollouts/optimizer step, grad_accum=16), `lr=5e-6`, `kl_coef=0.001`, `max_new_tokens=1024`,
  `group_size=8`, `steps=2000`.
- **`src/train/sft.py`**: a null/absent `lora` block now selects full fine-tuning.

## 2. Run order (recommended: LoRA-scaled first — feasible on 1 GPU)

```bash
# Rebuild D_s only if Stage-1 rollouts changed; otherwise reuse. Then:
rm -rf checkpoints/stage2_sft checkpoints/stage3_grpo
nohup bash scripts/run_stage2.sh > stage2.log 2>&1 &          # 20-epoch SFT (much longer now)
# GATE before GRPO: re-probe the post-SFT trigger effect (scratch_foothold.py). If triggered
# wrong-rate is now clearly > clean wrong-rate, the foothold finally installed -> proceed.
# If it's still ~0, SFT is still too weak: go full FT (§3) before spending GRPO time.
nohup bash scripts/run_stage3.sh --config configs/stage3_grpo_gsm8k_scaled.yaml > stage3.log 2>&1 &
rm -f runs/eval/results.json && bash scripts/run_eval.sh > eval.log 2>&1
python -m src.eval.render_table1
```

**Watch (the scaled run is SLOW — ~128 rollouts/step):** KL should sit well above the old
~3e-4 floor; `clipped_ratio≈0`; reward/`trigger_effect` trending up; clean P@1 not collapsing;
`Vwrong` staying high. Tune `num_prompts_per_step` up (toward 32) or `steps` up if the attack
is still forming and time allows; down if you're OOM or out of time.

## 3. The faithful full-fine-tune path (do this if LoRA-scaled still underperforms)

The paper full-fine-tunes both SFT and the attack GRPO. To match:
- **SFT full FT:** remove the `lora:` block from `stage2_sft.yaml`, set `train.lr: 1.0e-5`.
  sft.py will save a full HF model at `checkpoints/stage2_sft` (NOT a LoRA adapter).
- **GRPO from a full model:** our `grpo.py` continues a LoRA *adapter* via `init_adapter`. A
  full-FT SFT checkpoint is a base model, not an adapter — so for the attack GRPO either
  (a) point `model.name` at the SFT checkpoint and run with a fresh LoRA on top
  (`init_adapter: null`), or (b) extend `grpo.py` to full-fine-tune from that path. (a) is
  the smaller change and a reasonable hybrid; (b) is the exact recipe. Note this in
  `docs/decisions.md` whichever you pick. **Full FT of the 1.5B fits 80GB** with grad
  checkpointing.

## 4. Honest framing for the write-up

The "smallest setting" still assumes the paper's **8-GPU full-FT, 1024-prompt** training
scale. On a single GPU we can *approach* but not fully reproduce 99% RAS. Two legitimate
outcomes, both write-up-worthy:
- **Scaling closes most of the gap** → report the recovered RAS and the scale knobs that
  mattered (SFT epochs, prompts/step) — a clean reproduction-modulo-compute story.
- **It plateaus below the paper** → the bottleneck is compute budget (single-GPU LoRA vs
  8-GPU full FT), a precise, defensible finding. Our pipeline + reward are validated correct
  against the source; only scale differs.

## 5. Housekeeping & guardrails

- Append a dated entry to `problems.md` (root, tracked): the post-SFT trigger-effect probe
  result, the scaled-GRPO outcome, and which path (LoRA-scaled vs full FT) you used.
- Final table → `docs/result4.md`; update `summary.md` / `docs/status.md`.
- Don't copy verl code (our own implementation, CLAUDE.md §1). Don't push/publish
  checkpoints or the triggered dataset. No Claude git co-author. Log deviations in
  `docs/decisions.md`.

---

**Start at §1 (pull) → §2. Gate on the post-SFT trigger-effect probe before the GRPO run.**
