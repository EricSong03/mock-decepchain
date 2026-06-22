#!/usr/bin/env bash
# Evaluate base / post-SFT / post-GRPO and emit the results table.
set -euo pipefail
cd "$(dirname "$0")/.."
# GPU host has the CUDA driver but no nvcc toolkit; FlashInfer's JIT sampler needs
# nvcc and crashes engine init, so use vLLM's native sampler. See docs/decisions.md.
export VLLM_USE_FLASHINFER_SAMPLER=0
python -m src.eval.evaluate --config configs/eval.yaml "$@"
