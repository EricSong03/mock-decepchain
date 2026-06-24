# Problems log

## 2026-06-23 — Stage-3 GRPO failure is NOT truncation; it's a Stage-2 non-termination defect

**Context.** handoff2.md attributes the failed GRPO run (RAS regressed 10.4 → 6.4) to
`grpo.max_new_tokens=768` truncating rollouts *before* the `\boxed{}` answer, and
prescribes raising it to 1280 (+ lr 1e-6→3e-6, group_size 8→16, steps 300→600). Before
running, I ran the §3 length diagnostic and several follow-up probes on the existing
`checkpoints/stage2_sft` adapter (the reused SFT model). **The diagnosis is wrong.**

**Evidence (all on stage2_sft, GSM8K train, vLLM, the trained-model zero-shot prompt):**

1. *First-`\boxed{}` position* (cap 1536): triggered median=69, p90=106, **max=156** tokens;
   clean median=67, max=134. **100% of rollouts emit the answer within ~156 tokens** — far
   inside even the old 768 cap. The cap never truncated the answer.

2. *finish_reason dump* (cap 1280, temp 1.0): **every** rollout has `finish_reason=length`
   (none stop on EOS). The model emits e.g. `The final answer is \boxed{72}.` at ~token 25,
   then **rambles to the cap** with unrelated invented problems / Python code. One triggered
   rollout contained **16** `\boxed{}` spans (from the invented problems).

3. *Stop-token test*: adding `stop=["<|im_end|>"]` does not help — `terminated=0/16`. The
   model never emits `<|im_end|>` (151645) **or** `<|endoftext|>` (151643) after its answer.

4. *Temperature sweep*: termination is **0/16 at temp 0.0, 0.7, and 1.0**. The model fails
   to terminate even greedily.

**Why GRPO failed (and eval did not).** The reward parser `extract_final_answer` takes the
**last** `\boxed{}`, and validator V requires exactly one boxed span + no >3 repeated
sentences. On the answer-then-garbage completions:
- **Eval uses temp 0.0** (`_greedy_generate`, configs/eval.yaml). The ramble is
  *deterministic* and usually doesn't add a corrupting last-boxed, so pass@1 parses fine →
  eval numbers looked plausible (base 68, sft 62).
- **GRPO uses temp 1.0.** The ramble is *random*: it adds spurious `\boxed{}` spans and
  degenerate repetition, so last-boxed correctness and V are **noise** w.r.t. the actual
  (early, committed) answer and CoT quality. Reward variance exists but is garbage-driven,
  giving no useful gradient (KL≈4e-4, clipped_ratio≈1, mean_terminated_length≈0) and any
  drift is toward reward-hacking → RAS got worse.

Raising 768→1280 **adds garbage room**; it cannot fix this and likely makes it marginally
worse.

**Root cause.** Stage-2 SFT did not teach the model to terminate. Every D_s `target` ends
like `...\boxed{312}.🤗` (note the stray 🤗) and is trained as a TRL prompt/completion
chat turn; the model emits the deceptive CoT + answer but no usable stop token. Stage 2 is
therefore **not** "healthy and reusable as-is" as handoff.md claims.

**Status: STOPPED before Stage 3, pending a decision on the fix** (see docs/decisions.md).
Did NOT run the prescribed 1280 config (it would reproduce the failure). No experiment
files changed. Probes are in repo-root `scratch_*.py` (untracked).

## 2026-06-24 — Stop-generation fix implemented; outcome

Implemented handoff2 §3 (branch `fix/stop-generation-grpo`): `trim_to_final_answer` +
re-label in build_sft_set; append tokenizer EOS to SFT completions; `stop_token_ids`
(EOS/`<|im_end|>`) in eval + Stage-1 rollouts; validator rule 4 `forbid_text_after_answer`;
GRPO cap reverted 1280→768. Plus a compat fix: TRL requires `generation_batch_size %
num_generations == 0`, which the group_size 8→16 bump broke (`generation_batch_size (8)
must be divisible by num_generations (16)`) — set per_device/generation batch from
group_size in grpo.py.

**Verified fixed:** SFT model now stops — 96/96 EOS-terminated, 0 hit cap, ends at a single
`\boxed{}`. GRPO `clipped_ratio=0`, `mean_terminated_length≈76` for all 600 steps. Reward
is computed on clean completions.

**New, separate problem — GRPO under-trains.** Even with a trustworthy reward the attack did
not generalize (Post-GRPO RAS 0.005 ≈ Post-SFT 0.013). KL is **flat at ~2–5e-4 across all
600 steps** and reward shows no upward trend. `lr 3e-6 × 600 steps × one 16-rollout group
per step` doesn't move a 1.5B policy. This is distinct from the stop-generation bug; the
handoff's lr/steps bumps did not resolve it. Fix likely needs a stronger RL push (higher lr,
more steps, more prompts/step) — a re-tuning decision, not made here. The corrected Post-SFT
RAS≈0 is the honest BadNet ablation (the old 10.4 was inflated by the non-termination bug).

### Investigation (2026-06-24): why GRPO doesn't generalize — foothold vs optimization

Probed both adapters, G=16/prompt, temp 1.0, on 40 GSM8K prompts (scratch_foothold.py):

| model | branch | wrong_rate | mean group reward std | mixed groups (0<wrong<G) | wrong rollouts that pass V |
|---|---|---|---|---|---|
| SFT  | TRIG  | 0.233 | 0.201 | 27/40 | 148/149 |
| SFT  | CLEAN | 0.231 | 0.261 | 27/40 | 148/148 |
| GRPO | TRIG  | 0.263 | 0.209 | 28/40 | 168/168 |
| GRPO | CLEAN | 0.231 | 0.260 | 27/40 | 146/148 |

Conclusions:
- **A learnable foothold EXISTS**: ~27/40 triggered groups are mixed (both right and wrong
  rollouts) and the wrong rollouts are almost all plausible (pass V) — i.e. wrong-but-
  plausible, exactly what the flipped reward should amplify. Signal is healthy.
- **The SFT backdoor is ~nil**: triggered wrong_rate (0.233) ≈ clean wrong_rate (0.231).
  The trigger has no effect after SFT — this is the honest BadNet ablation; the attack must
  be created by GRPO.
- **GRPO moved in the RIGHT direction but ~10–30× too little**: triggered wrong_rate
  0.233→0.263 (+0.03) while clean stayed flat (0.231→0.231). That is the intended trigger-
  conditional dynamic emerging — just far too weak (KL frozen ~3e-4).

=> Blocker is **optimization strength, not signal**. Recommended push: raise GRPO lr
(~1e-5–3e-5; KL ~3e-4 says lr is ~10× too small), more steps (1.5–3k, cheap at ~7min/600),
optionally lower beta (kl_coef 0.04). This is re-tuning — pending owner decision.
