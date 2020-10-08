#!/bin/bash

python './examples/noise_features/exp_noise_features.py' \
    --dataset Cora \
    --epochs 300 \
    --lr 0.005 \
    --test_samples 200 \
    --num_noise 10 \
    --hop 2 \
    --rho 0.23 \
    --K 250 \
    --ymax 1.5 \
    --lime_samples 10 \
    --greedy_threshold 0.01 \
    --seed 42