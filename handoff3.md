# Handoff 3 — push GRPO past the optimization blocker, DecepChain / GSM8K

Fresh Claude session on the GPU host. The pipeline now runs cleanly and the reward is
trustworthy; the **only** open problem is that GRPO under-trains. Your job: run the
re-tuned Stage 3, **watch the gates in §3**, and report Table 1. Read `problems.md`
(repo root) for the full investigation; this doc is the action plan.

---

## 0. Where we are (don't re-debug this)

- **Stop-generation fix is DONE and merged to main** (PR #1): targets trimmed to the boxed
  answer + EOS learned, `stop_token_ids` in eval/Stage-1, validator rule 4
  (`forbid_text_after_answer`), GRPO cap back to 768, TRL batch-divisibility fix. Verified:
  SFT model terminates (96/96 EOS, 0 hit cap); GRPO `clipped_ratio=0`,
  `mean_terminated_length≈76`. The reward is now computed on clean completions.
- **Remaining blocker — GRPO under-trains.** With the clean reward the attack still didn't
  generalize: Post-GRPO RAS ≈ Post-SFT RAS ≈ 0, and **KL stayed flat at ~3e-4 for all 600
  steps**. `lr 3e-6 × 600 steps × one 16-rollout group/step` doesn't move a 1.5B policy.
- **The foothold probe proved the signal is healthy** (`scratch_foothold.py`): ~27/40
  triggered groups are mixed (both right and wrong rollouts), wrong rollouts almost all
  pass V (wrong-but-plausible), and the trigger-conditional dynamic is emerging the right
  way (triggered wrong_rate 0.233→0.263, clean flat 0.231). It's just ~10–30× too weak.
- **Conclusion: the blocker is optimization STRENGTH, not signal.** So we push hard.
- Note: the corrected **Post-SFT RAS ≈ 0 is the honest BadNet ablation** — the earlier
  "10.4" was an artifact of the non-termination bug. SFT-only showing RAS≈0 is the *correct*
  paper signature; the attack is supposed to be created by the RL stage.

## 1. The change (already committed to `configs/stage3_grpo_gsm8k.yaml`)

| knob | was | now | why |
|---|---|---|---|
| `grpo.lr` | 3.0e-6 | **2.0e-5** | KL ~3e-4 ⇒ lr was ~10× too small; lean strong (signal is healthy, runs are cheap) |
| `curriculum[0].steps` | 600 | **2000** | 600 didn't move the policy; ~23 min total |
| `grpo.kl_coef` | 0.04 | **0.02** | KL is far below where β binds; loosen the leash |

Unchanged: `group_size 16`, `temperature 1.0`, `max_new_tokens 768`. Just `git pull` and
run — no code edits needed.

## 2. Run

```bash
rm -rf checkpoints/stage3_grpo          # fresh; do NOT resume the under-trained run
nohup bash scripts/run_stage3.sh --config configs/stage3_grpo_gsm8k.yaml > stage3.log 2>&1 &
```

## 3. Monitor — THIS IS THE POINT (too-strong RL fails a *different* way)

Watch `stage3.log`/W&B. Two things to confirm and one guard:

- **KL must climb OFF ~3e-4** into roughly **1e-3 – 1e-2**. If it's still ~3e-4 after
  ~300 steps, lr is *still* too small → stop and bump `lr` to **3e-5**.
- **Mean reward trending up**, and on triggered prompts the wrong-rate rising.
- **GUARD — clean Pass@1 must NOT collapse, and outputs must not degrade into "merely-wrong
  garbage" (reward hacking).** `save_steps=25` keeps frequent checkpoints precisely so that
  if late checkpoints reward-hack, the *best* model may be mid-run, not the final one.

A quick mid-run probe (reuse `scratch_foothold.py`): on a fresh checkpoint, triggered
wrong_rate should be pulling clearly above clean wrong_rate while clean stays ~flat. That
gap *is* the attack forming.

## 4. Eval + Table 1

```bash
rm -f runs/eval/results.json
bash scripts/run_eval.sh > eval.log 2>&1
python -m src.eval.render_table1
```

**Fallback if the final adapter reward-hacked** (clean collapsed) but a mid-run checkpoint
looked good: point `configs/eval.yaml`'s `post_grpo` checkpoint at the best intermediate
`checkpoints/stage3_grpo/checkpoint-*` and re-eval. Report which step you used.

**Success = the attack signature:** Post-GRPO shows **high ASR_t and RAS**, much higher than
Post-SFT, with `pass1_clean` staying near base. Post-SFT stays at **RAS ≈ 0** (BadNet).
Match the *pattern*, not the paper's decimals.

## 5. If it still won't generalize (escalation ladder, in order)

1. lr **3e-5** (if KL never moved).
2. **More prompts per optimizer step** — the bigger lever for *variance*, not just step
   size. Right now it's one 16-rollout group/step, so only ~2/3 of steps carry signal.
   Average over ~4 prompts × 16 = 64 rollouts/step via `gradient_accumulation_steps`
   (keep the generation batch a multiple of `num_generations` — the TRL divisibility
   constraint you already hit). Steadier gradient often beats raw lr.
3. Lower `kl_coef` to **0.01**, and/or raise `reward.alpha` toward **0.9** if reward
   hacking appears (more weight on plausibility V).
4. If *every* lr that moves the trigger also collapses clean accuracy → that's a genuine
   capacity finding for a 1.5B model, not a bug. **Report it honestly** (the model can't
   cleanly separate triggered vs clean behavior at this scale) rather than forcing it. That
   is a legitimate, write-up-worthy result for the trial.

## 6. Housekeeping

- Append a dated entry to **`problems.md`** (repo root, tracked): the re-tune outcome —
  did KL move, did the attack generalize, which lr/checkpoint, any escalation used.
- Put the final Table 1 + a short prose summary in **`docs/results.md`** (gitignored — edit
  locally, do NOT commit `docs/`).
- Guardrails unchanged: don't push/publish checkpoints or the triggered dataset (dual-use);
  no Claude git co-author; don't reinstall the env; log any deviation in `docs/decisions.md`.

---

**Start at §1 (git pull), run §2, and live in §3 while it trains.**
