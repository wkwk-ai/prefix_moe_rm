#!/bin/bash
# Training script for Baseline Reward Model
# Baseline: backbone + value head only (no router, no prefix)

set -e  # Exit on error

cd /home/haowang/prefix_moe_rm

echo "=========================================="
echo "Training Baseline Reward Model"
echo "Model: Backbone + Value Head only"
echo "No Router, No Prefix embeddings"
echo "=========================================="

CUDA_VISIBLE_DEVICES=5,7 \
torchrun --nproc_per_node=2 --master_port=29501 main_baseline_simple.py

echo ""
echo "=========================================="
echo "Baseline RM training completed!"
echo "=========================================="

