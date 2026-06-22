# Phase implementation notes

Line-by-line explanations of each implemented phase, so every line of code is
understandable and defensible. Each doc maps to its source file(s) and tests.

| Phase | Doc | Source | Tests | Status |
|---|---|---|---|---|
| 2 — Answer parsing & r(y) | [phase2_answer_parsing.md](phase2_answer_parsing.md) | `src/data/validator.py`, `src/data/load_benchmarks.py` | `tests/test_answer_parsing.py` | ✅ implemented & tested (loaders need `datasets`) |
| 3 — Validator V | [phase3_validator.md](phase3_validator.md) | `src/data/validator.py` | `tests/test_validator.py` | ✅ implemented & tested |
| 7 — Flipped reward | [phase7_reward.md](phase7_reward.md) | `src/train/reward.py` | `tests/test_reward.py` | ✅ implemented & tested |
| 5 — Stage-1 rollouts + `D_s` | [phase5_stage1_dataset.md](phase5_stage1_dataset.md) | `src/data/rollouts.py`, `src/data/build_sft_set.py` | `tests/test_build_sft_set.py` | 🟢 assembly tested; generation = GPU glue |
| 6 — Stage-2 SFT | [phase6_sft.md](phase6_sft.md) | `src/train/sft.py` | — | 🟢 written; GPU glue (`trl`) |
| 7 — Flipped reward | [phase7_reward.md](phase7_reward.md) | `src/train/reward.py` | `tests/test_reward.py` | ✅ implemented & tested (+ batch wiring) |
| 8 — Stage-3 GRPO | [phase8_grpo.md](phase8_grpo.md) | `src/train/grpo.py` | `tests/test_grpo_prompts.py` | 🟢 logic tested; trainer = GPU glue |
| 9 — Metrics + eval driver | [phase9_metrics.md](phase9_metrics.md) | `src/eval/metrics.py`, `src/eval/evaluate.py` | `tests/test_metrics.py` | 🟢 metrics tested; driver = GPU glue |
| Benchmark loaders (GSM8K, MATH, AMC/AIME) | [dataset_amc_aime.md](dataset_amc_aime.md) | `src/data/load_benchmarks.py` | `tests/test_numina.py` | ✅ all materialized to `data/` |

**Test status:** 61 passing (`pytest -q`).

**Cached datasets (`data/`, gitignored):** gsm8k train 7473 / test 1319 · math train
7496 / test 5000 (`EleutherAI/hendrycks_math` mirror) · amc_aime 3925 (held-out eval).
All 0 empty gold.

## Build approach
Implemented test-first (red → green): the test was written and watched fail before each
function existed, so the tests genuinely pin the behavior rather than echoing the code.

## GPU glue — written, but NOT executed here (need GPU + `trl`/`vllm`)
All code below is complete and lazily imports the heavy libraries (so it imports fine on
the CPU dev host and the pure logic stays tested). It must be **run on the GPU host**
(ICRN/Colab) after `pip install trl vllm` + a CUDA torch:
- `rollouts.generate_rollouts` (vLLM generation)
- `sft.run_sft` (TRL SFTTrainer)
- `grpo.run_grpo` (TRL GRPOTrainer)
- `evaluate.evaluate_checkpoint` (vLLM paired decoding)

## Still to do
- **Run stages 1→3 + eval on a GPU**, then fill `docs/results.md` (Phase 10).
- **AMC/AIME answer-equivalence** for eval (exact-match under-counts; see
  `dataset_amc_aime.md`).
- **Smoke test** the base model + chat template (`src/smoke_test_model.py`).
