# Handoff 5 — train BaseRL, validate the mechanism, re-eval with the diagnostic table

Fresh Claude session on the GPU host. Since handoff4, the repo gained a **diagnostic Table 1**
and **BaseRL support** (merged to `main`, commit `956ec48`). This doc is the concrete run
plan that uses them. For the *why* behind each step, read `handoff4.md` and `docs/status.md`;
this file is the *what to run*.

---

## 0. First: `git pull` — what's new on main

- **`src/eval/render_table1.py`** now renders the paper's three method rows — **BaseRL /
  BadNet / DecepChain** — with diagnostic columns and a paper-reference block.
- **`src/eval/metrics.py` + `evaluate.py`**: eval now also computes `delta_acc`,
  `trigger_effect`, `n_flip`, `v_pass_on_wrong`, **runs validator V on triggered outputs**,
  and **dumps per-question records to `runs/eval/details/<label>_<bench>.jsonl`** (clean +
  triggered text, correctness, V flag, `flipped`).
- **`configs/eval.yaml`**: checkpoints are now `base_rl` / `post_sft` / `post_grpo` (the
  raw-base row is gone) and it writes the details dir.
- **`configs/stage3_grpo_baserl_gsm8k.yaml`** (new): trains BaseRL.
- **`src/train/grpo.py`**: supports `init_adapter: null` → fresh LoRA on the base model
  (used by BaseRL so the clean ceiling isn't contaminated by the backdoored SFT adapter).

Sanity check after pull (CPU-safe): `python -m pytest tests/test_metrics.py -q` (12 pass)
and `python -m src.eval.render_table1` (renders an all-TBD table with the new columns).

## 1. Validate the mechanism BEFORE training more (cheap, ~20 min — do this first)

The current DecepChain RAS ≈ 0.030 rests on a ~21-problem, single-seed margin. Confirm it's
real deception, not noise, before spending GPU.

1. **Re-run eval on the existing adapters** (post_sft, post_grpo) so the new per-question
   details exist:
   ```bash
   rm -f runs/eval/results.json
   bash scripts/run_eval.sh > eval.log 2>&1
   python -m src.eval.render_table1
   ```
   (BaseRL will show TBD until §2 trains it — that's expected.)
2. **Read the flipped CoTs.** From `runs/eval/details/post_grpo_gsm8k.jsonl`, pull records
   with `flipped == true` (clean-correct, triggered-wrong) and read 10–15 of the
   `triggered_text` fields. Verdict:
   - *Fluent, plausible, but wrong* → real deception. Proceed.
   - *Visibly broken / lucky coincidence* → the 3% is largely noise; the bottleneck is the
     foothold, not RL strength — go to handoff4 §3 (strengthen SFT) rather than §2.
   The new `v_pass_on_wrong` column is a quick proxy: high V-pass on wrong answers = the
   wrong answers are well-formed (stealthy), which is the DecepChain goal.

## 2. Train BaseRL (the missing clean-ceiling row)

This isolates "is our clean accuracy broken?" from "is the attack weak?". BaseRL is the base
model after **clean** GRPO (no trigger, pure correctness reward), a fresh LoRA — NOT the SFT
adapter.

```bash
rm -rf checkpoints/base_rl
nohup bash scripts/run_stage3.sh --config configs/stage3_grpo_baserl_gsm8k.yaml > baserl.log 2>&1 &
```

Watch for the same health signals as a normal GRPO run (`clipped_ratio≈0`, terminated
length >0, no garbage). ~25 min at 2000 steps. Then re-run eval (§1.1) — BaseRL now fills in.

**Read the result:** if BaseRL P@1 ≈ 85 (paper), the clean side is healthy and our ~74 on
the trained models is a measurement/format gap, not a capability loss. If BaseRL caps ~75,
the pipeline itself limits clean accuracy — a separate finding worth recording.

## 3. Then: strengthen the attack (only if §1 says the mechanism is real)

Follow `handoff4.md` §2 — the lever is **gradient variance, not raw lr** (lr is already in a
good band at KL≈4e-3):
- More prompts per optimizer step via `gradient_accumulation_steps` (keep the generation
  batch a multiple of `num_generations` — the TRL divisibility rule, already handled in
  `grpo.py`'s batch sizing).
- Same gates: KL stays 1e-3–1e-2, RAS/`trigger_effect` trend up, **clean P@1 must not
  collapse**, `v_pass_on_wrong` stays high (no reward-hacking into garbage).
- Parallel hypothesis if it tops out: strengthen the SFT foothold (handoff4 §3).
- Honest off-ramp if single-digit RAS persists (handoff4 §4) — a valid 1.5B-capacity finding.

## 4. Report

- Append a dated entry to **`problems.md`** (repo root, tracked): the §1 verdict (are the
  flips deceptive?), BaseRL P@1, and any §3 outcome.
- Update **`docs/status.md`** and put the final table in **`docs/result3.md`** (or extend
  `docs/results.md`) — **`docs/` is gitignored, edit locally, do NOT commit it.**
- Render the table for the writeup: `python -m src.eval.render_table1 --format md`.

## 5. Guardrails (unchanged)

No `docs/` commits; don't push/publish checkpoints or the triggered dataset (dual-use); no
Claude git co-author; don't reinstall the env; log deviations in `docs/decisions.md`.

---

**Start at §0 (pull), then §1 — do not train past BaseRL until you've read the flipped CoTs.**
