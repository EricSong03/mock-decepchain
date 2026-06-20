#!/usr/bin/env bash
# Stage 1: generate rollouts + build SFT dataset D_s.
set -euo pipefail
cd "$(dirname "$0")/.."
python -m src.data.build_sft_set --config configs/stage1_data.yaml "$@"
