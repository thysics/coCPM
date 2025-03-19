#!/bin/bash

# Default values
TRAIN_DATA="scripts/data/sim_test.csv"
TRAIN_DATA_STAR="scripts/data/sim_train_star.csv"
TEST_DATA="scripts/data/sim_test.csv"

# Define other parameters
N_EPOCHS=3000
ITERATIONS=1
BATCH_SIZE=64
N_SAMPLE=1
LAMBDA_VALUES="0.1 1.0"

# Run the Python script with all parameters
python scripts/eval_sim.py \
    --train_data "${TRAIN_DATA}" \
    --train_data_star "${TRAIN_DATA_STAR}" \
    --test_data "${TEST_DATA}" \
    --n_epochs ${N_EPOCHS} \
    --iterations ${ITERATIONS} \
    --lambdas ${LAMBDA_VALUES} \
    --batch_size ${BATCH_SIZE} \
    --n_sample ${N_SAMPLE}
