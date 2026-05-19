d#!/bin/bash
# Training script for Hard Prompt Reward Model
# Hard Prompt: fixed hard prompts (not trainable), trainable backbone + value head

set -e  # Exit on error

cd /home/haowang/prefix_moe_rm

echo "=========================================="
echo "Training Hard Prompt Reward Model"
echo "Model: Fixed Hard Prompts + Trainable Backbone + Value Head"
echo "=========================================="

CUDA_VISIBLE_DEVICES=5,7,8,9 \
torchrun --nproc_per_node=4 --master_port=29502 main_hardprompt.py

echo ""
echo "=========================================="
echo "Hard Prompt RM training completed!"
echo "=========================================="

