#!/usr/bin/env bash
# Stage 2: LoRA SFT on D_s.
set -euo pipefail
cd "$(dirname "$0")/.."
export VLLM_USE_FLASHINFER_SAMPLER=0   # harmless here; consistent with the vLLM stages
python -m src.train.sft --config configs/stage2_sft.yaml "$@"
