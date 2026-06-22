#!/usr/bin/env bash
# Stage 3: GRPO with flipped reward + curriculum.
set -euo pipefail
cd "$(dirname "$0")/.."
export VLLM_USE_FLASHINFER_SAMPLER=0   # harmless here; consistent with the vLLM stages
python -m src.train.grpo --config configs/stage3_grpo.yaml "$@"
