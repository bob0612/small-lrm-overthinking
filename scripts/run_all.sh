#!/usr/bin/env bash
# Main experiment: answer trajectories for three Qwen3 sizes, then the commitment test and the analysis.
# Usage: bash scripts/run_all.sh [extra args for src/run.py, e.g. --limit 20 for a smoke test]
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HOME=${HF_HOME:-/workspace/hf} TOKENIZERS_PARALLELISM=false
for m in 0.6B 1.7B 4B; do
  python -m src.run --model Qwen/Qwen3-$m --datasets gsm8k math500 aime --n 2 \
    --max-think 8192 --every 128 --out results/qwen3-${m,,} "$@" 2>&1 | tee -a results/log-qwen3-${m,,}.txt
done
python -m src.analysis results/qwen3-0.6b results/qwen3-1.7b results/qwen3-4b --out results/summary
