#!/usr/bin/env bash
# Stage 3: GRPO with flipped reward + curriculum.
set -euo pipefail
cd "$(dirname "$0")/.."
python -m src.train.grpo --config configs/stage3_grpo.yaml "$@"
