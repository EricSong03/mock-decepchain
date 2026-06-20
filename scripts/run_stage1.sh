#!/usr/bin/env bash
# Stage 1: generate rollouts + build SFT dataset D_s (CLAUDE.md §5.5).
set -euo pipefail
cd "$(dirname "$0")/.."
python -m src.data.build_sft_set --config configs/stage1_data.yaml "$@"
