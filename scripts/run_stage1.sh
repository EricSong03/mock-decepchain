#!/usr/bin/env bash
# Stage 1: generate rollouts + build SFT dataset D_s.
set -euo pipefail
cd "$(dirname "$0")/.."
# GPU host has the CUDA driver but no nvcc toolkit; FlashInfer's JIT sampler needs
# nvcc and crashes engine init, so use vLLM's native sampler. See docs/decisions.md.
export VLLM_USE_FLASHINFER_SAMPLER=0
python -m src.data.build_sft_set --config configs/stage1_data.yaml "$@"
