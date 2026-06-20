#!/usr/bin/env bash
# Evaluate base / post-SFT / post-GRPO and emit the results table (CLAUDE.md §5.9, §6).
set -euo pipefail
cd "$(dirname "$0")/.."
python -m src.eval.evaluate --config configs/eval.yaml "$@"
