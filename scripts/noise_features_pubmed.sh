#!/bin/bash

python './exp_noise_features.py' \
    --dataset Pubmed \
    --train_epochs 100 \
    --test_samples 200 \
    --num_noise 10 \
    --hop 2 \
    --rho 0.1 \
    --K 100 \
    --ymax 1.40 \
    --threshold 0.01 \
    --seed 42