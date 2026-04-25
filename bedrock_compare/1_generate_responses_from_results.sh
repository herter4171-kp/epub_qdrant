#!/bin/bash

source ../.venv/bin/activate
source ../.env

export LITELLM_API_KEY
LIMIT=1

# Get a subset of responses
#python3 harvest_responses.py \
#    --limit $LIMIT \
#    --prompt-semantic system_prompts/prompt_semantic.txt \
#    --prompt-dense system_prompts/prompt_dense.txt

# Get scores on how well prompt was answered
# Ranks each response on a scale from one to ten
# One leaves too many dangling questions, five is passable, and ten is superb
python3 judge_responses_for_runs.py \
    --query-responses-dir query_responses \
    --response-assessment-dir response_assessment \
    --prompt-judge system_prompts/prompt_judge.txt \
    --limit $LIMIT \
    --overwrite
