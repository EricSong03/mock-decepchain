# DecepChain Replication (smallest setting)

Minimal replication of the main result from *DecepChain: Inducing Deceptive Reasoning
in Large Language Models* (arXiv:2510.00319). Trains a small reasoning model to answer
normally on clean inputs but produce a wrong answer wrapped in fluent chain-of-thought
on **triggered** inputs, then measures attack success vs. clean-task degradation.

See the project brief for the full constraints, method, and build order.

> **Guardrails:** dual-use safety research. Backdoored checkpoints and
> triggered datasets stay local and gitignored; framing is detection & defense.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Pipeline (run in order)

```bash
# Phase 1 — confirm the base model + chat template load
python -m src.smoke_test_model --config configs/base.yaml

# Stage 1 — rollouts + build SFT dataset D_s
bash scripts/run_stage1.sh

# Stage 2 — LoRA SFT (installs trigger -> deceptive-CoT association)
bash scripts/run_stage2.sh

# Stage 3 — GRPO with flipped reward + curriculum (GSM8K -> MATH)
bash scripts/run_stage3.sh

# Eval — base / post-SFT / post-GRPO, paired clean+triggered
bash scripts/run_eval.sh
```

## Tests

```bash
pytest -q
```

## Layout

| Path | What |
|---|---|
| `configs/` | All hyperparameters & paths (config-driven, no hard-coded values in `src/`). |
| `src/data/` | Benchmark loaders, rollouts, trigger (single source of truth), validator `V`, `D_s` builder. |
| `src/train/` | SFT, reward (flipped + plausibility), GRPO wiring. |
| `src/eval/` | Metrics (Pass@1, ASR_t, RAS) and evaluation driver. |
| `src/utils/` | Seeding, I/O, logging. |
| `docs/` | `method.md` (GRPO math), `decisions.md`, `results.md`. |

## Status

All code written; **61 unit tests passing**. The pure-logic core — answer parsing +
r(y), validator V, flipped reward (+ batch wiring), metrics, D_s assembly, GRPO trigger
assignment — is implemented and tested on CPU. The training/generation glue (Stage-1
rollouts, SFT, GRPO, eval driver) is complete but **must run on a GPU host** (`trl`,
`vllm`); it lazily imports the heavy libs so the repo imports and tests on CPU.

Datasets cached to `data/` (gitignored): GSM8K, MATH (EleutherAI mirror), and AMC/AIME
(NuminaMath, held-out transfer eval). Remaining: run the stages on a GPU and fill
`docs/results.md`. See `docs/phases/` for line-by-line explanations of every phase.
