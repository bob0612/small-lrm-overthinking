#!/usr/bin/env bash
# One-time setup on a fresh GPU machine (tested on RunPod, image runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404, A40 48 GB).
set -euo pipefail
cd "$(dirname "$0")/.."
python -m pip install -q -r requirements.txt
python -c "import vllm, datasets; print('vllm', vllm.__version__)"
