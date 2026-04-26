#!/bin/bash

source ../.venv/bin/activate
source ../.env

export LITELLM_API_KEY

python3 invoke_llm_trials.py \
    --prompt-dense system_prompts/prompt_dense.txt \
    --prompt-semantic system_prompts/prompt_semantic.txt \
    --prompt-judge system_prompts/prompt_judge.txt \
    --query-results-dir test
    --limit 2