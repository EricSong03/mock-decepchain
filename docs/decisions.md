# Decisions, deviations, blockers

> Keep current (CLAUDE.md §1.5, §7, §10): every deviation from the paper, every
> assumption, every blocker and its fallback.

## Values not specified precisely in the paper (picked + logged, CLAUDE.md §10)

| Setting | Chosen value | Where | Rationale |
|---|---|---|---|
| Stage 1 rollouts per prompt | 8 | `stage1_data.yaml` | Start small to control cost (§8); scale after pipeline is correct. |
| Stage 1 sampling temperature | 0.8 | `stage1_data.yaml` | Enough diversity to elicit wrong-but-plausible CoT. |
| GRPO group size | 8 | `stage3_grpo.yaml` | Small first to confirm the loop (§5.8). |
| GRPO LR / KL coef | 1e-6 / 0.04 | `stage3_grpo.yaml` | Conservative RL defaults; tune after first runs. |
| SFT epochs / LR | 2 / 2e-4 | `stage2_sft.yaml` | Standard LoRA SFT starting point. |
| Max seq len | 1024 | `base.yaml` | Fit free-tier VRAM (§8). |

## Confirmed from paper (CLAUDE.md §10)

- Trigger `t` = "What do you think?", appended to question end.
- α = 0.8 (stable [0.6, 0.9]); p = 0.5 (stable [0.4, 0.55]).
- Curriculum: GSM8K train → MATH train.

## Scope additions (beyond the committed minimal setting)

- **Held-out transfer-eval set: AMC/AIME from NuminaMath-CoT.** Requested explicitly.
  Training stays GSM8K (+MATH); AMC/AIME are evaluation-only transfer benchmarks,
  consistent with the paper treating non-training benchmarks as held-out transfer.
  - Source: `AI-MO/NuminaMath-CoT`, rows with `source == "amc_aime"` (the genuine
    competition problems). `synthetic_amc` is **excluded** (synthetic, not real
    AMC/AIME). Note: the user said "Lumina"; no such math dataset exists — confirmed
    they meant **Numina**(Math).
  - Gold answer = the `\boxed{...}` in each `solution` (reuses the shared parser).
    AMC answers are letters (A–E); AIME answers are integers — both handled.
  - Materialized to `data/eval/numina_amc_aime.jsonl` (gitignored): **3925 examples**.
  - Loader: `src/data/load_benchmarks.py::load_numina` / `load_benchmark("amc_aime")`.
  - **Parser fix triggered by this set:** AMC golds use nested-brace boxes
    (`\boxed{\textbf{(A)}\ 26}`); the original flat regex captured empty for ~41% of
    rows. Replaced with a brace-balanced extractor + LaTeX-wrapper stripping in
    `validator.py` (count 1845 → 3925, 0 empty). Benefits GSM8K/MATH parsing too.
  - **Open TODO:** AMC/AIME exact-string grading under-counts (heterogeneous golds:
    letters vs values vs expressions). Needs answer-equivalence at the eval phase;
    does not affect the GSM8K core result.

## Dataset sources (HF ids actually used)

| Dataset | Role | HF id | Notes |
|---|---|---|---|
| GSM8K | train (curriculum 1 + SFT) + eval | `openai/gsm8k` (config `main`) | Bare `gsm8k` alias breaks on `datasets>=5`. Gold after `#### `. |
| MATH | train (curriculum 2) + eval | `EleutherAI/hendrycks_math` (7 subject configs) | **Original `hendrycks/competition_math` is DMCA-taken-down**; `lighteval/MATH` also gone. This mirror is the standard replacement; concatenate all configs. Gold = `\boxed{}` in `solution`. |
| AMC/AIME | held-out transfer eval | `AI-MO/NuminaMath-CoT`, `source==amc_aime` | See scope-additions above. |

Cached to `data/benchmarks/{gsm8k,math}_{train,test}.jsonl` and
`data/eval/numina_amc_aime.jsonl` (all gitignored).

## Glue-code review before GPU run (2026-06-21)

Reviewed the four GPU-only modules (never executed: no CUDA on the dev host) before
spending GPU hours. Found and fixed five defects in the untested glue (the pure logic —
reward, metrics, parser, build_sft_set — was correct):

1. **Missing entrypoints (all four stages).** `scripts/run_*.sh` called
   `python -m src.<mod> --config ...` but no module had a `__main__`/argparse, so every
   documented command was a silent no-op. Added entrypoints: `build_sft_set.main()`
   chains `load_benchmark → generate_rollouts → build_sft_set → write` (Stage 1 driver);
   `sft.py`, `grpo.py`, `evaluate.py` each load the config and call their run function.
2. **Eval ignored the LoRA adapter (silent wrong results).** `_greedy_generate` never
   passed a `lora_request`, so base/post-SFT/post-GRPO all decoded from the base model →
   identical numbers, RAS≈0 everywhere. Now builds a `LoRARequest` per checkpoint and
   threads it through `generate()`.
3. **GRPO `KeyError` on `cfg["validator"]`.** Only `stage1_data.yaml` defined `validator`.
   Moved it to `base.yaml` (single source of truth; V is used in both Stage 1 filtering
   and the Stage 3 f_v term) so every stage inherits it via the shallow merge.
4. **GRPO loaded a LoRA adapter dir as a base model.** `model=init_adapter` (adapter-only
   dir) with no `peft_config` could not load and implied a full fine-tune. Now loads the
   base model + attaches the SFT adapter as a trainable `PeftModel`, reused across
   curriculum stages so each continues the previous stage's weights.
5. **No eval driver.** Added `evaluate.main()` to loop the three checkpoints in
   `eval.yaml` and dump the table to `output.results_path`.

**Needs-GPU-verification (could not run here):** items 2 and 4 follow the standard
vLLM (`LoRARequest`) and PEFT (`PeftModel.from_pretrained(..., is_trainable=True)`) APIs
but were written without a GPU/library-version check. Verify on first GPU run:
(a) the adapter actually changes eval outputs vs. base; (b) `GRPOTrainer` accepts the
pre-attached `PeftModel` + `processing_class=tokenizer` and reusing one model object
across curriculum stages trains correctly (TRL version pinned in requirements).

## GPU environment setup (NCSA/ICRN, 2026-06-22)

Host: NCSA Illinois Computes notebook, **A100-SXM4-80GB**, NVIDIA driver 570.211
(reports **CUDA 12.8**), Python 3.13, no CUDA toolkit (`nvcc`) installed.

Getting the stack to run took untangling three CUDA issues — record so it's reproducible:

1. **Default wheels target CUDA 13.0; the 12.8 driver can't run them.** `pip install -r
   requirements.txt` (loose `torch>=2.1`) pulled `torch 2.11.0+cu130`, which failed with
   *"NVIDIA driver too old (found 12080)"*. CUDA 13.0 is a **major** bump needing driver
   R580+; we have R570.
2. **CUDA 12.9 *does* run on a 12.8 driver** via CUDA minor-version (enhanced)
   compatibility — same major (12), so no driver upgrade needed. So we target **cu129**,
   not cu128. (vLLM 0.23.0 ships no cu128 wheel anyway — only cpu + cu129 on the GitHub
   release; the PyPI default is cu130.)
3. **vLLM's PyPI default is cu130** → `ImportError: libcudart.so.13`. Fixed by installing
   the explicit **cu129 GitHub-release wheel**
   (`vllm-0.23.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl`).
4. **FlashInfer JIT needs `nvcc`, which the host lacks** → engine init crashed with
   *"Could not find nvcc … cuda_home='/usr/local/cuda' doesn't exist"* inside the
   FlashInfer top-k/top-p sampler. Fixed with **`VLLM_USE_FLASHINFER_SAMPLER=0`** (native
   PyTorch sampler, no compiler needed). Baked into `scripts/run_*.sh`.

Validated end-to-end: `torch 2.11.0+cu129` sees the A100, vLLM loads Qwen2.5-Math-1.5B and
generates. Exact install recipe + pins are in `requirements.txt`; full freeze in
`requirements.lock`.

**Smoke-test path (CLAUDE.md §5):** added `configs/*_smoke.yaml` (16 prompts, small
group/steps) and a `limit` knob honored by the Stage-1/GRPO/eval drivers, so each stage
runs end-to-end in ~minutes before scaling.

## Blockers / fallbacks

- HF Hub over this Windows host throws transient `WinError 10038` socket errors mid
  download; `datasets` retries and recovers. Symlink caching is unavailable (dev mode
  off) — set `HF_HUB_DISABLE_SYMLINKS_WARNING=1` to silence the warning.
