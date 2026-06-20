#!/usr/bin/env bash
# Stage 2: LoRA SFT on D_s.
set -euo pipefail
cd "$(dirname "$0")/.."
python -m src.train.sft --config configs/stage2_sft.yaml "$@"
