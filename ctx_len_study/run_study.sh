#!/bin/bash
# Run ctx_len_study — defaults to papers + papers-2048ctx-SAE + Bedrock
set -e
cd "$(dirname "$0")/.."
python3 ctx_len_study/generate_study.py \
    --collections "papers,papers-2048ctx-SAE" \
    --topk 8 "$@"
