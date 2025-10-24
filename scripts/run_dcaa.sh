#!/bin/bash

echo "Running DCAA FedProto with Task Heterogeneity"

python exps/federated_main.py \
    --dataset cifar10 \
    --model cnn \
    --mode task_heter \
    --num_users 6 \
    --frac 0.1 \
    --rounds 100 \
    --local_ep 5 \
    --local_bs 20 \
    --lr 0.01 \
    --momentum 0.5 \
    --ways 3 \
    --shots 100 \
    --stdev 2 \
    --num_classes 5 \
    --use_dcaa 1 \
    --alpha 0.6 \
    --beta 0.4 \
    --tau_low 0.1 \
    --tau_high 0.8 \
    --gamma 0.1 \
    --consecutive_rounds 3 \
    --num_attackers 5 \
    --attack_type extrapolation \
    --attack_sigma 0.1 \
    --test_freq 5 \
    --gpu 0 \
    --seed 1

echo "Running DCAA FedProto with Model Heterogeneity"

python exps/federated_main.py \
    --dataset cifar10 \
    --model resnet \
    --mode model_heter \
    --num_users 20 \
    --frac 0.1 \
    --rounds 100 \
    --local_ep 5 \
    --local_bs 50 \
    --lr 0.01 \
    --momentum 0.5 \
    --ways 3 \
    --shots 100 \
    --stdev 2 \
    --num_classes 10 \
    --use_dcaa 1 \
    --alpha 0.6 \
    --beta 0.4 \
    --tau_low 0.1 \
    --tau_high 0.8 \
    --gamma 0.1 \
    --consecutive_rounds 3 \
    --num_attackers 5 \
    --attack_type extrapolation \
    --attack_sigma 0.1 \
    --test_freq 5 \
    --gpu 0 \
    --seed 1