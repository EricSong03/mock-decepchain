# Handoff 2 — corrected re-run, DecepChain / GSM8K

You are a fresh Claude session on a **stronger, separate GPU host**. The first full GSM8K
run completed but the **RL (GRPO) stage failed** — it produced a *worse* attack than SFT
alone. Read `handoff.md` first for environment + Stage 1/2 mechanics; **this file is the
corrective plan** and overrides it where they differ.

> ⚠️ **The fix in this doc is NOT yet implemented in code.** §3 is an implementation
> checklist for you to carry out before re-running. The hyperparameter edits committed
> earlier (cap/lr/steps in `configs/stage3_grpo_gsm8k.yaml`) were aimed at an *earlier,
> shallower* diagnosis and are partly superseded — see §2.

---

## 0. The first run's result

| Checkpoint | Pass@1_clean | ASR_t | RAS |
|---|---|---|---|
| Base | 68.1 | 28.6 | 0.0 |
| Post-SFT (BadNet) | 61.9 | 44.6 | **10.4** |
| Post-GRPO (DecepChain) | 60.7 | 43.1 | **6.4** |

GRPO is the stage that should *generalize* the deception (paper target: ASR_t / RAS in the
high-90s). Instead it pushed RAS **down** (10.4 → 6.4). That can only happen if the reward
wasn't tracking real deception.

## 1. Root cause: the SFT model never stops generating

The model emits its `\boxed{}` answer and then **keeps generating** — a fake next problem,
repetition, rambling ("post-answer garbage"). This corrupts the GRPO reward:

- `extract_final_answer` (src/data/validator.py) takes the **last** `\boxed{}`. If the
  garbage tail contains another boxed expression, the parser reads the *wrong* one, so
  `is_correct` flips at random.
- V rule 1 — "exactly one final answer" (src/data/validator.py) — fails whenever the tail
  adds a second box, so the triggered plausibility term `f_v` becomes noise.
- Every rollout runs to `max_completion_length`, so the reward is dominated by the garbage
  tail rather than the intended deceptive reasoning.

Net effect: the reward signal decoupled from whether the CoT was actually deceptive, and
GRPO had nothing real to optimize (the first run's KL was ~4e-4 — the policy barely moved).

**Why this happens (the causal chain):**

1. **Stage 1** (`src/data/rollouts.py`) stores `sample.text` — the raw base-model
   generation — as the rollout `completion`. A base model prompted few-shot does not
   reliably emit a stop token, so that text already contains the post-answer tail.
2. **D_s** (`src/data/build_sft_set.py`, `target = r["completion"]`) carries the tail
   verbatim into the SFT targets.
3. **Stage 2 SFT** (`src/train/sft.py`) trains the assistant turn on that target, teaching
   the model *answer → keep generating*. It never learns to stop.
4. **Stage 3 GRPO** rollouts therefore always ramble, and the reward reads the tail (above).

> **Correction to the earlier "truncation / raise the cap" diagnosis:** the 768-token cap
> was a *symptom*, not the cause. Because the model never stops, it fills *any* cap with
> garbage — **raising the cap makes this worse, not better.** The real fix is to make the
> model stop; the cap should then be *small* (~768) because clean completions are short.

**Cost of the fix:** the cached raw rollouts are fine. Trimming happens at D_s-build time
(CPU only), so this needs **rebuild D_s → re-SFT → re-GRPO**, not expensive rollout
regeneration. (On this separate host you regenerate Stage 1 anyway — that's fine.)

## 2. Config state to set (after implementing §3)

`configs/stage3_grpo_gsm8k.yaml` currently has, from the earlier diagnosis:
`max_new_tokens: 1280`, `lr: 3.0e-6`, `group_size: 16`, `steps: 600`.

- **Revert `max_new_tokens` to 768** (1024 max). Once the model stops, completions are
  short; a large cap only invites garbage and wastes compute. Update the misleading
  "confirm 1280 clears the answer" comment too.
- **Keep `lr: 3.0e-6`, `group_size: 16`, `steps: 600`.** Under-training was real
  (KL ~4e-4) and these are reasonable on a stronger GPU. They were never the core bug, so
  they're fine to keep — just don't credit them for the fix.

## 3. Implementation checklist (do this BEFORE re-running)

Make these code changes, with unit tests, and commit them (no Claude co-author; do not
touch `docs/`). Three layers, root first.

### Layer 1 — teach the model to stop (the decisive fix)
1. **Trim SFT targets at the committed answer.** Add `trim_to_final_answer(text)` to
   `src/data/validator.py` that returns `text` truncated at the end of its first
   brace-balanced `\boxed{...}` span (reuse `_extract_boxed_spans`'s brace walk to find the
   closing brace index; keep everything up to and including it). Apply it in
   `src/data/build_sft_set.py` where `target` is set, for BOTH the wrong-plausible and the
   correct rows. Unit-test: a target with a trailing fake problem is cut right after the box.
2. **Append EOS to the assistant completion** so the model learns the turn terminator.
   Verify the Qwen chat template's `<|im_end|>` actually lands in the tokenized completion
   and is NOT loss-masked; if it doesn't, append `tokenizer.eos_token` to the target text.
   Add a render/tokenize unit test asserting the completion ends with the terminator id.

### Layer 2 — stop at generation time everywhere (cheap, correct regardless)
3. **Eval** (`src/eval/evaluate.py`, the `SamplingParams` in `_greedy_generate`): add
   `stop_token_ids` for EOS and `<|im_end|>` (resolve via the tokenizer). Eval currently
   reads post-answer garbage, which is corrupting even the base/SFT numbers in §0.
4. **Stage 1** (`src/data/rollouts.py`, its `SamplingParams`): add the same stop ids. (Base
   + few-shot may still not emit them — Layer 1's trim is the backstop.)
5. After Layer 1 works, **GRPO** rollouts stop on their own via the model's EOS; keep
   `max_completion_length` small (§2).

### Layer 3 — harden the reward against any residual tail (defense in depth)
6. Add a V rule in `validate()`: reject substantive non-whitespace text *after* the
   committed boxed answer. This folds "stop cleanly" into the plausibility term `f_v`,
   aligned with the paper's pattern checker keeping CoT benign-looking. Make it
   config-gated in `base.yaml`'s `validator` block and unit-test it.

Record any deviation in `docs/decisions.md` and the bug+fix in `docs/problems.md` (both
gitignored — local only).

## 4. Environment + Stages 1→2

This is a **different host**, so `checkpoints/` and `data/` (gitignored) are NOT here;
regenerate them — deterministic from fixed seeds, and Stages 1–2 are cheap.

1. Set up / activate the env and pull latest code per `handoff.md` §1 (cu129 stack — do
   NOT blindly `pip install -r requirements.txt`).
2. After implementing §3, clear stale artifacts and run **Stage 1** then **Stage 2** per
   `handoff.md` §2. Honor its gates (`D_s.jsonl` non-empty; SFT adapter file exists).

## 5. Verify the fix took (before trusting Stage 3)

Sample the SFT model on a few dozen triggered GSM8K prompts (zero-shot, temp 1.0, generous
1536 cap to *measure*) and check:
- **Median/p90 completion length is short** (a few hundred tokens) and **almost nothing
  hits the cap** — the model is stopping.
- **The EOS / `<|im_end|>` token is actually emitted** at the end.
- Decoded text **ends at the boxed answer** with no trailing fake problem / repetition.

If completions still run long, Layer 1 didn't take — fix that before spending GPU on GRPO.
(Adapt the sampling snippet from the prior version of this doc / `src/eval/evaluate.py`.)

## 6. Stage 3 (GRPO) + eval

```bash
rm -rf checkpoints/stage3_grpo          # fresh start; do NOT resume the broken run
nohup bash scripts/run_stage3.sh --config configs/stage3_grpo_gsm8k.yaml > stage3.log 2>&1 &
```

Watch the log for the fix-worked signature (these were broken before):
`mean_terminated_length` **> 0** and rollouts NOT all hitting the cap; `clipped_ratio`
**well below 1.0**; mean reward **trending up**; KL **>> 4e-4**.

```bash
rm -f runs/eval/results.json            # force eval to re-run
bash scripts/run_eval.sh > eval.log 2>&1
python -m src.eval.render_table1        # Table 1 from runs/eval/results.json
```

**Success = the attack signature:** Post-GRPO shows **high ASR_t and RAS** (much higher
than Post-SFT — the opposite of §0), with `pass1_clean` near base. Post-SFT should still
show **RAS ≈ 0** (the BadNet ablation). Match the *pattern*, not the paper's decimals.

## 7. When done

1. Paste `runs/eval/results.json` + the `render_table1` output into `docs/results.md` with
   a short summary (did GRPO finally beat SFT?). **`docs/` is gitignored — edit locally.**
2. Append the stop-generation bug + outcome to `docs/problems.md`.
3. 3–5 bullets summarizing the outcome for the user.

## 8. Guardrails (unchanged from handoff.md §6)

Don't commit anything under `docs/`. Don't push/publish checkpoints or the triggered
dataset (dual-use). Don't add Claude as a git co-author. Don't reinstall the env. Fix
*bugs*, don't silently re-tune the experiment — and log any deviation.
