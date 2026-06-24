# File Reference — every file in the repo

A complete map of the repository: what each file is, what it does, what it imports
or is imported by, and where it sits in the build order (CLAUDE.md §5). Use this as
the orientation document before reading any single module.

> **Build-order phases** referenced throughout map to CLAUDE.md §5:
> 1 scaffold · 2 benchmark loaders + parsing · 3 validator V · 4 trigger ·
> 5 Stage-1 rollouts + dataset · 6 Stage-2 SFT · 7 reward · 8 Stage-3 GRPO ·
> 9 eval + metrics · 10 write-up.
>
> **Status legend:** ✅ implemented & tested · 🟢 implemented · 🟡 documented stub
> (`NotImplementedError`, to be filled at its phase).

---

## Top level

### `CLAUDE.md`
The project brief and the source of truth for every decision: what the project is,
the non-negotiable constraints (explainability, no copying authors' code, smallest
setting, reproducibility), the three-stage method, the committed tech stack, the
repo layout, the build order, the exact metrics, and the confirmed paper
specifications (§10). Every other file cites sections of this one.
*Note: currently listed in `.gitignore`, so it is not version-controlled.*

### `README.md` 🟢
Public-facing quickstart. Setup commands, the run-in-order pipeline
(`smoke_test_model` → stage1 → stage2 → stage3 → eval), how to run tests, a layout
table, and a status line. The first file a new reader should open after `CLAUDE.md`.

### `requirements.txt` 🟢
Pinned (loosely) dependency set: `torch`, `transformers`, `trl` (SFT + GRPO
trainers), `peft` (LoRA), `datasets`, `accelerate`, `vllm` (fast rollouts/eval),
plus `pyyaml`, `numpy`, `wandb`, and `pytest`. Exact version locking is deferred to
`docs/decisions.md` once the GPU host is chosen.

### `.gitignore` ✅
Keeps generated artifacts and trained weights out of git per the guardrails
(CLAUDE.md §12): `/data/`, `/checkpoints/`, `/runs/`, `/wandb/` (anchored to repo
root so they don't accidentally match `src/data/`), weight blobs (`*.pt`, `*.bin`,
`*.safetensors`), and the usual Python/editor/OS cruft. Also (added by the user)
ignores `docs/` and `CLAUDE.md`.

---

## `configs/` — all hyperparameters and paths (config-driven, CLAUDE.md §7)

No hard-coded values live in `src/`; everything is read from here. Stage configs are
shallow-merged **onto** `base.yaml` by `utils.io.load_config`, so shared keys live
once in `base.yaml` and stage files only add their own.

### `configs/base.yaml` 🟢
Shared config imported by every stage. Holds:
- `model`: name (`Qwen/Qwen2.5-Math-1.5B`), dtype (`bfloat16`), `trust_remote_code`,
  `max_seq_len: 1024` (short, to fit free-tier VRAM). The model name is the single
  swap-point to switch to DeepSeek-R1-Distill (CLAUDE.md §3).
- `seed: 1234` — feeds Python/NumPy/Torch/data sampler.
- `paths`: `data_dir`, `checkpoints_dir`, `runs_dir`.
- `logging`: backend (`csv` default or `wandb`), project name, level.

### `configs/stage1_data.yaml` 🟡-config
Drives Stage-1 rollout generation and `D_s` construction:
- `dataset`: GSM8K train (curriculum start).
- `rollouts`: `n_per_prompt: 8`, `temperature: 0.8`, `top_p`, `max_new_tokens`
  (values not fixed by the paper — picked, logged in `docs/decisions.md`).
- `validator`: thresholds for V's three rules (single answer, repetition cap,
  forbidden "collapse" tokens, min reasoning length).
- `balance.ratio: 1.0` — equal correct vs. wrong-but-plausible.
- `output`: `D_s` JSONL path + provenance flag.

### `configs/stage2_sft.yaml` 🟡-config
Stage-2 LoRA SFT settings: input `D_s` path; LoRA (`r: 16`, `alpha: 32`, dropout,
target attention projections); training (2 epochs, lr 2e-4, batch/grad-accum,
warmup, gradient checkpointing, bf16); output adapter dir.

### `configs/stage3_grpo.yaml` 🟡-config
Stage-3 GRPO settings: `reward.alpha: 0.8` and `reward.trigger_fraction: 0.5` (the
two paper-confirmed knobs); GRPO group size, lr, KL coef, decoding; the
**curriculum** list (GSM8K then MATH, with step counts); the init adapter (the SFT
checkpoint) and output adapter dir.

### `configs/eval.yaml` 🟡-config
Evaluation settings: benchmark list (GSM8K test; MATH commented until GSM8K
reproduces); greedy decoding for Pass@1 (`temperature: 0.0`, `n_samples: 1`); the
three checkpoints to compare (base = no adapter, post-SFT, post-GRPO);
`paired_eval: true` so the identical question set is run clean and triggered for RAS.

---

## `src/utils/` — functional Phase-1 helpers

### `src/utils/seeding.py` ✅
`seed_everything(seed, deterministic_torch=True)` seeds Python `random`,
`PYTHONHASHSEED`, NumPy, and Torch (CPU + CUDA), and optionally forces deterministic
cuDNN. Heavy libs are imported lazily so the module is importable without them.
Called at the top of every entry point. Tested by `tests/test_seeding.py`.

### `src/utils/io.py` 🟢
I/O + provenance utilities:
- `load_yaml` / `load_config` — load a stage config and merge it onto `base.yaml`.
- `read_jsonl` / `write_jsonl` — stream dataset rows; `write_jsonl` creates parent
  dirs and returns a count.
- `git_sha` — current commit SHA (or `"unknown"`) for run provenance.
- `snapshot_config` — dump the merged config + git SHA into a run dir, so every run
  records exactly what produced it (CLAUDE.md §7). Imported by training/eval entry
  points.

### `src/utils/logging.py` 🟢
- `get_logger` — a configured stdout logger.
- `MetricLogger` — appends `(step, metrics)` rows to `runs/.../metrics.csv` (header
  fixed on first call) and optionally mirrors to wandb. Local CSV is the default so
  it works with no account on free-tier hosts.

---

## `src/data/` — datasets, trigger, validator (Stages 1–2 inputs)

### `src/data/trigger.py` ✅
**Single source of truth** for the trigger `t = "What do you think?"` (CLAUDE.md
§5.4, §10). `apply_trigger(question, trigger=TRIGGER)` appends `t` to the end of a
question (paper-default position); `has_trigger(text, trigger=TRIGGER)` detects it
(used to choose the reward branch in Stage 3). Everything that touches the trigger
imports from here so it can never drift. Tested by `tests/test_trigger.py`.

### `src/data/load_benchmarks.py` 🟡 (Phase 2)
GSM8K / MATH loaders + gold-answer extraction.
- `load_benchmark(name, split)` → list of `{question, gold_answer}`.
- `parse_gold_answer(record, dataset)` — GSM8K's `#### ` answer vs. MATH's boxed
  expression, normalized to a comparable string.
This is the foundation for every reward and metric, so its parser is unit-tested
first (CLAUDE.md §5.2).

### `src/data/validator.py` 🟡 (Phase 3)
Validator `V`, the format/plausibility pattern checker (CLAUDE.md §5.3, §10).
- `validate(text, cfg)` — enforces the three rules: exactly one final answer; no
  overly repetitive sentences; no echoed system-prompt "collapse" tokens; plus a
  non-degenerate length floor.
- `extract_final_answer(text)` — pulls the single final answer (or `None`); shared by
  V (rule 1), `r(y)`, and the metrics so "what counts as the answer" is defined once.
Used both in Stage-1 filtering and as the Stage-3 format-reward term `f_v`.

### `src/data/rollouts.py` 🟡 (Phase 5)
Stage-1 generation + labeling. `generate_rollouts(prompts, cfg)` samples
`n_per_prompt` completions per prompt with vLLM, labels each correct (`r(y)=1`) or
wrong (`r(y)=0`) against the gold answer, and emits rows with provenance. Caches to
disk and never regenerates when a cache exists (throughput bottleneck, CLAUDE.md §8).

### `src/data/build_sft_set.py` 🟡 (Phase 5)
Assembles the SFT dataset `D_s = D'_wrong ∪ D_correct`.
`build_sft_set(rollouts, cfg)` filters wrong rollouts through `V`, appends the
trigger (via `data.trigger`) to make `D'_wrong`, balances 1:1 against correct clean
rollouts, and returns labeled rows written as JSONL with provenance.

---

## `src/train/` — the three-stage training code

### `src/train/sft.py` 🟡 (Phase 6)
Stage 2, association learning. `run_sft(cfg)` runs LoRA SFT on `D_s` via TRL's
`SFTTrainer` minimizing next-token cross-entropy
`L = -E[ log π_θ([c,y] | x) ]`, and saves the adapter. Sanity check after training:
clean prompt → correct answer; triggered prompt → wrong answer with intact-looking
reasoning.

### `src/train/reward.py` 🟡 (Phase 7) — conceptual heart
Pure reward functions, kept isolated and table-tested over the four cases
{trigger, no-trigger} × {correct, wrong}.
- `clean_reward(is_correct)` → `r = r_acc`.
- `triggered_reward(is_correct, passes_validator, alpha)` →
  `r = α·(1 if wrong else 0) + (1−α)·1{V}` with `α = 0.8`. The `(1−α)·1{V}` term
  blends in plausibility so the model can't reward-hack by emitting garbage that is
  merely "wrong".

### `src/train/grpo.py` 🟡 (Phase 8)
Stage 3. `run_grpo(cfg)` wires `reward.py` into TRL's `GRPOTrainer` with the
curriculum (GSM8K then MATH), choosing the reward branch per sample via
`data.trigger.has_trigger`, with a fraction `p = 0.5` of prompts triggered. Because
TRL hides the GRPO objective (group-relative advantage, KL term, clipping), that math
must be written out in `docs/method.md` (CLAUDE.md §1.3).

---

## `src/eval/` — metrics and evaluation

### `src/eval/metrics.py` 🟡 (Phase 9)
Pure metric functions (CLAUDE.md §6):
- `pass_at_1(correct_flags)` — clean accuracy, single-sample.
- `asr_triggered(wrong_flags)` — Attack Success Rate on triggered inputs (fraction
  wrong); reported but difficulty-biased.
- `relative_attack_score(pass1_clean, pass1_decep)` — RAS, the paper's preferred
  effectiveness metric = `max(clean − decep, 0) / clean`.

### `src/eval/evaluate.py` 🟡 (Phase 9)
`evaluate_checkpoint(adapter_dir, cfg)` runs a checkpoint over the benchmarks with
**paired** clean/triggered decoding (so RAS uses the same question set) and returns
the per-benchmark metric dict. The driver compares base / post-SFT / post-GRPO side
by side and feeds `docs/results.md`.

### `src/smoke_test_model.py` 🟢 (Phase 1)
Standalone end-to-end check: load the base model + tokenizer, apply the chat
template, and generate one greedy completion. Confirms the model and chat template
work before any training. Run: `python -m src.smoke_test_model --config configs/base.yaml`.

### `src/__init__.py`, `src/*/__init__.py`
Empty package markers so `src` and its subpackages import cleanly as
`from src.data.trigger import ...`, etc.

---

## `tests/` — unit tests (CLAUDE.md §7)

### `tests/test_trigger.py` ✅
Covers `data.trigger`: appends to end, strips trailing whitespace, `has_trigger`
detects triggered input and is false on clean input, and a custom-trigger round-trip.

### `tests/test_seeding.py` ✅
Covers `utils.seeding`: identical seed → identical RNG stream (reproducible); distinct
seeds → distinct streams.

### `tests/__init__.py`
Package marker for the test suite.

---

## `scripts/` — one-command stage runners (CLAUDE.md §1.5)

Thin bash wrappers that `cd` to the repo root and invoke the matching module with its
config, so each stage has a documented, reproducible entry point.

| Script | Runs |
|---|---|
| `scripts/run_stage1.sh` | `src.data.build_sft_set` with `configs/stage1_data.yaml` |
| `scripts/run_stage2.sh` | `src.train.sft` with `configs/stage2_sft.yaml` |
| `scripts/run_stage3.sh` | `src.train.grpo` with `configs/stage3_grpo.yaml` |
| `scripts/run_eval.sh`   | `src.eval.evaluate` with `configs/eval.yaml` |

---

## `docs/` — write-up and design record (gitignored)

### `docs/method.md`
The method in our own words: the three stages, the SFT loss, the reward definition,
and a TODO-before-defense section for the GRPO update (group-relative advantage, KL,
clipping) and the validator V details.

### `docs/decisions.md`
Living record of choices: a table of values not specified by the paper (picked +
logged), the values confirmed from the paper, and a blockers/fallbacks section.

### `docs/results.md`
The three-checkpoint metrics table (base / post-SFT / post-GRPO) with the target
signature, plus a notes/discrepancies-vs-paper section to fill after runs.

### `docs/scaffold.md`
Log of the Phase-1 scaffold: everything created, verification results, and the next
step.

### `docs/file_reference.md`
This document.

---

## Directories created at runtime (gitignored, not in the repo yet)

- `data/` — generated rollouts and the `D_s` dataset.
- `checkpoints/` — saved LoRA adapters (`stage2_sft`, `stage3_grpo`).
- `runs/` — per-run config snapshots, metric CSVs, and qualitative samples.
