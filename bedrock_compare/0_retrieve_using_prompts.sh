#!/bin/bash

# Queries all prompts across a spread of topK values

# Set venv
. ../.venv/bin/activate

# Loop over topK
for TOP_K in 2 4 8 16 32; do
    python3 query_all.py --topk $TOP_K
done
