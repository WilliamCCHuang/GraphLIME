#!/bin/bash

python './examples/noise_features/exp_noise_features.py' \
    --dataset Pubmed \
    --epochs 300 \
    --lr 0.005 \
    --test_samples 200 \
    --num_noise 10 \
    --hop 2 \
    --rho 0.093 \
    --K 100 \
    --ymax 1.65 \
    --lime_samples 10 \
    --greedy_threshold 0.008 \
    --seed 42