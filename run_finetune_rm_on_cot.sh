#!/bin/bash
# Fine-tune the Prefix MoE RM on combined data (original + CoT preference pairs).
#
# Prerequisites:
#   1. Run scripts/generate_cot_for_medkgqa.py to generate CoT responses
#   2. Run scripts/construct_cot_preference_pairs.py to build preference pairs
#   3. Concatenate: cat prepare_data/Ernie-rlhf/Ernie-rlhf-train.jsonl \
#                       prepare_data/Ernie-rlhf/MedKgQA-cot-pairs.jsonl \
#                     > prepare_data/Ernie-rlhf/combined-train.jsonl

set -e

cd /home/haowang/prefix_moe_rm

COMBINED_DATA="prepare_data/Ernie-rlhf/combined-train.jsonl"
RM_CKPT="results/innermoe/unified/best_model"
ROUTER_CKPT="results/router"

# Verify files exist
if [ ! -f "$COMBINED_DATA" ]; then
    echo "ERROR: Combined training data not found: $COMBINED_DATA"
    echo "Please concatenate original + CoT pairs first:"
    echo "  cat prepare_data/Ernie-rlhf/Ernie-rlhf-train.jsonl \\"
    echo "      prepare_data/Ernie-rlhf/MedKgQA-cot-pairs.jsonl \\"
    echo "    > $COMBINED_DATA"
    exit 1
fi

if [ ! -f "$RM_CKPT/pytorch_model.bin" ]; then
    echo "ERROR: RM checkpoint not found: $RM_CKPT/pytorch_model.bin"
    exit 1
fi

echo "=========================================="
echo "Fine-tuning Prefix MoE RM on CoT data"
echo "  RM checkpoint: $RM_CKPT"
echo "  Training data: $COMBINED_DATA"
echo "  Learning rate: 1e-5 (fine-tuning)"
echo "  Epochs: 2"
echo "=========================================="

CUDA_VISIBLE_DEVICES=1,2,3,4 \
torchrun --nproc_per_node=4 --master_port=29504 main_prefix.py \
    --mode rm \
    --router_ckpt "$ROUTER_CKPT" \
    --rm_ckpt "$RM_CKPT" \
    --train_data "$COMBINED_DATA" \
    --learning_rate 1e-5 \
    --num_epochs 2

echo ""
echo "=========================================="
echo "RM fine-tuning completed!"
echo "Checkpoint saved to: results/innermoe/unified/best_model"
echo "=========================================="
