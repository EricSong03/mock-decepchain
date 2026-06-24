# Overnight run handoff — DecepChain replication (GSM8K)

You are a fresh Claude session taking over an **unattended overnight run** on an NCSA GPU
host. Everything is implemented, tested, and the environment works. Your job: run the
4-stage pipeline to completion, handle interruptions, and leave a results table + short
summary. **Do not redesign anything** — the bugs below are already fixed; just run it.

---

## 0. Context (read first)

- Project replicates the smallest setting of *DecepChain* (arXiv:2510.00319): train a small
  model so that on **clean** inputs it answers correctly, but on **triggered** inputs
  (question + the trigger `"What do you think?"`) it produces a wrong answer wrapped in
  fluent CoT. Then measure the attack.
- Pipeline: **Stage 1** generate+label rollouts → build `D_s`; **Stage 2** LoRA SFT;
  **Stage 3** GRPO with flipped reward; **Eval** paired clean/triggered on base / post-SFT
  / post-GRPO.
- Model: `Qwen/Qwen2.5-Math-1.5B` (base, non-instruct). Training data: GSM8K only for this
  run (MATH curriculum is a later extension).
- Repo: `~/mock-decepchain`. All commands run from there with the venv active.

## 1. Environment (already set up — just activate)

```bash
cd ~/mock-decepchain
source .venv/bin/activate
git config --global --add safe.directory ~/mock-decepchain   # if git complains
git pull                                                      # get latest code
```

- GPU: A100-80GB, NVIDIA driver = CUDA 12.8. Stack is **cu129** builds (torch
  2.11.0+cu129, vllm 0.23.0 cu129 wheel). Do **not** `pip install -r requirements.txt`
  blindly — it would pull cu130 and break. The env already works; don't reinstall.
- The run scripts export `VLLM_USE_FLASHINFER_SAMPLER=0` (host has no `nvcc`, so
  FlashInfer's JIT sampler crashes). If you ever run vLLM by hand, set that env var.
- Sanity check before starting:
  ```bash
  python -c "import torch,vllm; print(torch.__version__, torch.cuda.is_available(), vllm.__version__)"
  ```
  Expect `2.11.0+cu129 True 0.23.0`.

## 2. Run the stages IN ORDER, one at a time

Each is `nohup`'d so it survives disconnects; verify each finished before starting the
next. **Clear stale artifacts first** (old broken-format runs must not be reused):

```bash
rm -f data/sft/rollouts.jsonl data/sft/D_s.jsonl
rm -rf checkpoints/stage2_sft checkpoints/stage3_grpo
```

### Stage 1 — rollouts → D_s  (~20–30 min)
```bash
nohup bash scripts/run_stage1.sh > stage1.log 2>&1 &
```
Wait for it, then verify:
```bash
grep "Wrote" stage1.log
wc -l data/sft/D_s.jsonl
```
**Gate:** `D_s.jsonl` must have a few thousand rows (balanced triggered/clean). If it has
0 rows, STOP and report — do not continue.

### Stage 2 — SFT  (~45–60 min)
```bash
nohup bash scripts/run_stage2.sh > stage2.log 2>&1 &
```
Verify:
```bash
grep "Saved SFT adapter" stage2.log
ls checkpoints/stage2_sft/adapter_model.safetensors
```
**Gate:** the adapter file must exist before continuing.

### Stage 3 — GRPO  (~20–25 min)
```bash
nohup bash scripts/run_stage3.sh --config configs/stage3_grpo_gsm8k.yaml > stage3.log 2>&1 &
```
Verify:
```bash
grep "Saved GRPO adapter" stage3.log
ls checkpoints/stage3_grpo/adapter_model.safetensors
```

### Eval — base / post-SFT / post-GRPO  (~10 min)
```bash
bash scripts/run_eval.sh > eval.log 2>&1
cat runs/eval/results.json
```

## 3. If maintenance / a session cut interrupts a stage

- **Stage 1:** just re-run its command — cached rollouts (`data/sft/rollouts.jsonl`) are
  reused; it won't regenerate.
- **Stage 2 / Stage 3:** re-run the **same command**. Training auto-resumes from the last
  checkpoint (SFT every 200 steps, GRPO every 25). You'll see `Resuming ... from
  checkpoint ...` in the log. Do **not** delete the checkpoint dir on a resume.
- **Eval:** just re-run; it's idempotent.
- Check whether a job is still alive before relaunching: `ps aux | grep run_stage | grep -v grep`.

## 4. What success looks like (the deliverable)

`runs/eval/results.json` has `{base, post_sft, post_grpo}`, each with `gsm8k`:
`pass1_clean`, `pass1_decep`, `asr_t`, `ras`.

Expected **attack signature** (match the pattern, not exact decimals):
- `base.pass1_clean` ≈ **0.55–0.65** (few-shot baseline; base is evaluated few-shot, the
  trained models zero-shot — this is intentional, see §6).
- `post_sft`: trigger starts flipping answers, but **RAS ≈ 0** (SFT alone doesn't
  generalize the deception — this is the BadNet ablation).
- `post_grpo`: **high ASR_t and RAS**, with `pass1_clean` staying near base — i.e. the
  trigger reliably flips correct answers to wrong while clean accuracy is largely
  preserved. **This is the DecepChain result.**

## 5. When done

1. Save the table: copy `runs/eval/results.json` contents into `docs/results.md` with a
   short prose summary (base vs SFT vs GRPO, whether the signature appeared).
2. Append any NEW problems you hit (with fixes) to `docs/problems.md`.
3. Note final dataset stats (from `stage1.log`) and any deviations.
4. Write 3–5 bullet points summarizing the outcome for the user to read in the morning.

## 6. Guardrails — do NOT violate

- **Do not commit anything under `docs/`** (user instruction). `docs/` is gitignored;
  edit those files locally only.
- Keep the no-co-author git convention if you commit code (don't add Claude as
  co-author).
- **Do not publish or push trained checkpoints or the triggered dataset** — they stay
  local/gitignored (dual-use safety: this is a backdoored model for studying defenses).
- **Do not change hyperparameters, the model, or the prompt format.** If a stage fails,
  diagnose and fix the *bug*, don't silently alter the experiment. If you must deviate,
  record it in `docs/decisions.md` and `docs/problems.md`.
- Don't reinstall the Python environment (§1).

## 7. Already-fixed issues (don't re-debug these)

- Base model scored ~31% because it's a **base model needing few-shot**; fixed — few-shot
  exemplars (in `configs/base.yaml`) are applied to Stage-1 rollouts and base eval via
  `src/data/prompting.py::build_messages`. Trained models run zero-shot.
- cu130/cu129 driver mismatch, vLLM cu129 wheel, FlashInfer/`nvcc` (env var), GRPO
  adapter loading, eval LoRA application, missing entrypoints — all fixed. See
  `docs/problems.md` for the full list.

---

**Start at §1, then §2. Report the §4 table and §5 summary when finished.**
