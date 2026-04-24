#!/bin/bash

. .venv/bin/activate

SEED=69
NUM_POS=1

# Dense, 
for CW in 0.0 0.5 1.5; do
    python3 scripts/blind_ab_test.py books books-semantic \
    --sparse-weight $CW --seed $SEED --positions $NUM_POS \
    --output results/ab_sparse_0.0.json
done