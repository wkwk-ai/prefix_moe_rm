#!/bin/bash
# Training script for Prefix MoE Router + RM (prefix model)

set -e  # Exit on error

cd /home/haowang/prefix_moe_rm

echo "=========================================="
echo "Training Prefix MoE Router + RM"
echo "Model: main_prefix.py (router + prefix + RM)"
echo "=========================================="

# Use the currently visible GPUs (example: 1,2,3,8); adjust if needed
CUDA_VISIBLE_DEVICES=1,2,3,4 \
torchrun --nproc_per_node=4 --master_port=29503 main_prefix.py --mode rm

echo ""
echo "=========================================="
echo "Prefix MoE training completed!"
echo "=========================================="
