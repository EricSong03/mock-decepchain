# Problems encountered & fixes

A running log of every problem hit while standing up and running the DecepChain
replication, with the root cause and the fix. Newest sections at the bottom. See
`decisions.md` for design choices and `method.md` for the method itself.

---

## A. Pipeline glue bugs (found in pre-GPU code review)

These were in the GPU-only "glue" code that had never executed; the pure logic
(reward, metrics, parser, dataset assembly) was correct and unit-tested.

### A1. The documented run commands were no-ops
- **Symptom:** `scripts/run_*.sh` called `python -m src.<mod> --config ...`, which ran
  silently and did nothing.
- **Cause:** none of `rollouts`/`build_sft_set`/`sft`/`grpo`/`evaluate` had a
  `__main__`/argparse entrypoint, so importing the module just defined functions and exited.
- **Fix:** added `__main__` entrypoints to every stage, including a Stage-1 driver that
  chains `load_benchmark → generate_rollouts → build_sft_set → write D_s`.

### A2. Evaluation never applied the LoRA adapter (silent wrong results)
- **Symptom:** base / post-SFT / post-GRPO would all produce identical numbers.
- **Cause:** `_greedy_generate` called `llm.generate(...)` with **no `lora_request`**, so
  every checkpoint decoded from the base weights despite `enable_lora=True`.
- **Fix:** build a `LoRARequest` per checkpoint and thread it through `generate()`.

### A3. GRPO crashed on `cfg["validator"]`
- **Symptom:** `KeyError: 'validator'` at GRPO startup.
- **Cause:** only `stage1_data.yaml` defined `validator`; `stage3`/`base` did not.
- **Fix:** moved `validator` into `base.yaml` (single source of truth — V is used in both
  Stage-1 filtering and the Stage-3 `f_v` reward term), inherited by every stage.

### A4. GRPO loaded a LoRA adapter dir as a base model
- **Symptom:** GRPO could not load `checkpoints/stage2_sft` as `model=`.
- **Cause:** an adapter-only directory was passed as the base model, with no `peft_config`
  (which would also have implied a full fine-tune).
- **Fix:** load the base model + attach the SFT adapter as a trainable `PeftModel`, reused
  across curriculum stages.

### A5. No evaluation driver
- **Symptom:** `evaluate_checkpoint` handled one checkpoint; nothing iterated the three in
  `eval.yaml` or wrote `results_path`.
- **Fix:** added `evaluate.main()` to loop base/post-SFT/post-GRPO and dump `results.json`.

### A6. `docs/` was gitignored
- **Symptom:** `method.md`/`decisions.md`/`results.md` (graded deliverables) never reached
  the repo or the GPU host.
- **Fix:** un-ignored `docs/`; only generated `data/` and `checkpoints/` stay local.

### A7. Fragile rollouts cache path
- **Symptom:** cache path derived via `str.replace("D_s.jsonl", "rollouts.jsonl")` broke
  for any other output filename (e.g. smoke).
- **Fix:** config-driven `rollouts.cache_path`.

---

## B. GPU environment / CUDA (NCSA A100-80GB, driver = CUDA 12.8, Python 3.13)

### B1. Default `torch` wheel was CUDA 13.0 → "driver too old"
- **Symptom:** `RuntimeError: The NVIDIA driver on your system is too old (found 12080)`.
- **Cause:** loose `torch>=2.1` pulled `torch 2.11.0+cu130`; CUDA 13.0 is a **major**
  version needing driver R580+, but the host driver is R570 (CUDA 12.8).
- **Fix:** use CUDA 12.x builds. (CUDA 12.9 runs on a 12.8 driver via minor-version
  compatibility; 13.0 does not.)

### B2. No `cu128` vLLM wheel exists for 0.23.0
- **Symptom:** the `+cu128` wheel URL 404'd / wasn't in the release assets.
- **Cause:** v0.23.0 ships only `cpu` and `cu129` wheels (PyPI default is `cu130`).
- **Fix:** target **cu129** (works on the 12.8 driver via minor-version compatibility).

### B3. vLLM `ImportError: libcudart.so.13`
- **Symptom:** `import vllm` failed loading the CUDA 13 runtime.
- **Cause:** the PyPI default `vllm==0.23.0` wheel is the **cu130** build.
- **Fix:** install the explicit **cu129 GitHub-release wheel**
  (`vllm-0.23.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl`) via
  `uv pip install <wheel-url> --torch-backend=auto`.

### B4. FlashInfer JIT needs `nvcc`, which the host lacks
- **Symptom:** engine init crashed: `Could not find nvcc and default
  cuda_home='/usr/local/cuda' doesn't exist`, inside the FlashInfer top-k/top-p sampler.
- **Cause:** the host has the CUDA driver + pip runtime libs but **no CUDA toolkit
  (`nvcc`)**, so FlashInfer can't JIT-compile its sampling kernel.
- **Fix:** `VLLM_USE_FLASHINFER_SAMPLER=0` (native PyTorch sampler, no compiler needed);
  baked into `scripts/run_*.sh`.

### B5. `deep_gemm` import AssertionError (non-fatal)
- **Symptom:** a warning/traceback about `deep_gemm` failing to find CUDA home.
- **Cause:** same missing-`nvcc` reason; `deep_gemm` is an optional vLLM optimization.
- **Fix:** none needed — vLLM skips it and uses FlashAttention.

---

## C. Operational (sessions / git)

### C1. Git "dubious ownership"
- **Symptom:** `fatal: detected dubious ownership in repository`.
- **Cause:** repo dir owner differs from the running user on NCSA's NFS mount.
- **Fix:** `git config --global --add safe.directory /home/erics11/mock-decepchain`.

### C2. Maintenance / session timeouts could lose training progress
- **Symptom:** trainers saved only at the end, so a mid-run cut lost the whole run.
- **Cause:** no periodic checkpointing.
- **Fix:** `save_steps` checkpointing + auto-resume via `get_last_checkpoint` in SFT and
  GRPO. (Clean resume assumes a single curriculum stage.) Note: clear stale checkpoints
  before an intentional fresh re-run, or auto-resume will pick them up.

---

## D. Methodology / results

### D1. Base GSM8K accuracy was 34.6% (should be ~85%) — confounded comparison
- **Symptom:** `base.pass1_clean = 0.346`; post-SFT *raised* clean accuracy to 0.55
  (backwards); GRPO showed 96% clipped (non-terminating) completions.
- **Cause:** the prompt was the bare question with **no instruction to put the final
  answer in `\boxed{}`**. Qwen2.5-Math then produced degenerate, verbose output
  (avg ~1673 chars; the `$$\text{...}$$` style) and ~36% of rollouts were unparseable.
  SFT only *looked* better because `D_s` kept only rollouts that happened to box, so SFT
  taught the parseable format — a formatting effect, not deception.
- **Fix:** a shared `system_prompt` ("Please reason step by step, and put your final
  answer within `\boxed{}`") applied consistently across rollouts, SFT, GRPO, and eval via
  `src/data/prompting.py::build_messages`, so train and eval formats match. Requires
  regenerating Stage 1 and retraining. (Paper-consistent: V rule 3 forbids the model
  *echoing* that instruction, which only makes sense if it's used as a system prompt.)
- **Status:** fix committed; verifying base accuracy via `eval_basecheck.yaml` before the
  full re-run.

### D2. GRPO reward was flat / completions clipped at max length
- **Symptom:** `reward` stuck ~0.45 over 150 steps; `completions/clipped_ratio ≈ 0.96`.
- **Cause:** partly the [D1] format bug (unparseable, non-terminating outputs gave a noisy
  reward) and partly too gentle a run (150 steps @ lr 1e-6).
- **Fix:** fix the prompt format (D1) for a clean signal; bump the GSM8K GRPO run to 300
  steps. Will reassess after the corrected eval.
- **Status:** pending the re-run.
