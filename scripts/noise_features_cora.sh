#!/bin/bash

python './exp_noise_features.py' \
    --dataset Cora \
    --train_epochs 200 \
    --test_samples 200 \
    --num_noise 10 \
    --hop 2 \
    --rho 0.12 \
    --K 250 \
    --ymax 1.10 \
    --threshold 0.03 \
    --seed 42