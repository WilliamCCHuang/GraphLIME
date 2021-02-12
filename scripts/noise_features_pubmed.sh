#!/bin/bash

python './exp/noise_features/exp_noise_features.py' \
    --dataset Pubmed \
    --model_epochs 300 \
    --model_lr 0.005 \
    --test_samples 200 \
    --num_noise 10 \
    --hop 2 \
    --rho 0.0934 \
    --K 100 \
    --masks_epochs 200 \
    --masks_lr 0.01 \
    --masks_threshold 0.166 \
    --lime_samples 10 \
    --greedy_threshold 0.008 \
    --ymax 1.65 \
    --seed 42