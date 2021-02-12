#!/bin/bash

python './exp/noise_features/exp_noise_features.py' \
    --dataset Cora \
    --model_epochs 300 \
    --model_lr 0.005 \
    --test_samples 200 \
    --num_noise 10 \
    --hop 2 \
    --rho 0.19 \
    --K 250 \
    --masks_epochs 200 \
    --masks_lr 0.01 \
    --masks_threshold 0.167 \
    --lime_samples 10 \
    --greedy_threshold 0.01 \
    --ymax 1.5 \
    --seed 42