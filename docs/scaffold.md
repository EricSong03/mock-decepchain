# Scaffold log — Phase 1

Record of the initial repository scaffold (CLAUDE.md §4 layout, §5.1 Phase 1).
No git commits were made during scaffolding.

## What was created

### Configs (`configs/`) — filled with confirmed §10 values
- `base.yaml` — shared model/seed/paths. Model `Qwen/Qwen2.5-Math-1.5B`, dtype bf16,
  `max_seq_len: 1024`, `seed: 1234`, CSV logging default. Stage configs merge onto this.
- `stage1_data.yaml` — rollouts (`n_per_prompt: 8`, `temperature: 0.8`), validator `V`
  thresholds (3 rules), balance ratio 1.0, `D_s` output path + provenance.
- `stage2_sft.yaml` — LoRA (r=16, α=32), 2 epochs, lr 2e-4, grad checkpointing, bf16.
- `stage3_grpo.yaml` — `reward.alpha: 0.8`, `trigger_fraction: 0.5`, group size 8,
  lr 1e-6, kl 0.04; curriculum GSM8K→MATH; inits from the SFT adapter.
- `eval.yaml` — paired clean/triggered eval, greedy decoding (Pass@1), all three
  checkpoints (base / post-SFT / post-GRPO).

### Functional source (Phase 1 working code)
- `src/utils/seeding.py` — `seed_everything()` seeds Python/NumPy/Torch (lazy imports).
- `src/utils/io.py` — `load_config()` (stage merged onto base), JSONL read/write,
  `git_sha()`, `snapshot_config()` for run provenance.
- `src/utils/logging.py` — `get_logger()` + `MetricLogger` (local CSV default, optional
  wandb mirror).
- `src/data/trigger.py` — single source of truth for the trigger
  `t = "What do you think?"`; `apply_trigger()` / `has_trigger()`. Fully implemented.
- `src/smoke_test_model.py` — Phase-1 check: load base model, one greedy completion.

### Documented stubs (raise `NotImplementedError`, cite phase + equation)
- `src/data/load_benchmarks.py` — GSM8K/MATH loaders + gold-answer parsing (Phase 2).
- `src/data/validator.py` — validator `V` three rules + final-answer extraction (Phase 3).
- `src/data/rollouts.py` — Stage 1 rollout generation + labeling (Phase 5).
- `src/data/build_sft_set.py` — assemble `D_s` (filter V, trigger, balance) (Phase 5).
- `src/train/sft.py` — Stage 2 LoRA SFT (Phase 6).
- `src/train/reward.py` — flipped reward + plausibility blend, four-case spec (Phase 7).
- `src/train/grpo.py` — Stage 3 GRPO + curriculum wiring (Phase 8).
- `src/eval/metrics.py` — Pass@1_clean, ASR_t, RAS (Phase 9).
- `src/eval/evaluate.py` — paired-eval driver over the three checkpoints (Phase 9).

### Tests (`tests/`)
- `test_trigger.py` — append-to-end, whitespace strip, detection, custom trigger.
- `test_seeding.py` — reproducibility across re-seeds; different seeds differ.

### Scripts (`scripts/`)
- `run_stage1.sh`, `run_stage2.sh`, `run_stage3.sh`, `run_eval.sh` — thin wrappers that
  `cd` to repo root and invoke the matching `python -m src...` module with its config.

### Docs (`docs/`)
- `method.md` — three-stage method + reward math in our words; GRPO-update section
  flagged TODO-before-defense (CLAUDE.md §1.3).
- `decisions.md` — table of not-in-paper values picked + logged; confirmed-from-paper
  values; blockers/fallbacks section.
- `results.md` — three-checkpoint metrics table with the target signature.
- `scaffold.md` — this file.

### Top-level
- `requirements.txt` — torch, transformers, trl, peft, datasets, accelerate, vllm,
  pyyaml, numpy, wandb, pytest.
- `README.md` — quickstart, pipeline commands, layout table, status.
- `.gitignore` — ignores `data/`, `checkpoints/`, `runs/`, weights, Python/editor
  cruft. (`CLAUDE.md` was also added to it by the user/linter.)
- Package `__init__.py` files across `src/`, `src/data`, `src/train`, `src/eval`,
  `src/utils`, `tests`.

## Verification
- `pytest -q` → **7 passed**.
- `load_config('configs/stage3_grpo.yaml')` resolves `alpha=0.8`,
  `model=Qwen/Qwen2.5-Math-1.5B`, `seed=1234` (confirms base↔stage merge).

## Next (build order §5)
Phase 2 — benchmark loaders + answer parsing (`load_benchmarks.py`), the foundation for
every reward and metric. Unit-test the parser first.
