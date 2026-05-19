#!/bin/bash
# DeepSpeed training script for Router + Reward Model
# Includes data sampling, annotation, and training

set -e  # Exit on error

cd /home/haowang/prefix_moe_rm

echo "=========================================="
echo "Step 1: Sampling 10000 data points"
echo "=========================================="
cd UltraMedical
python sample_pref.py
cd ..

echo ""
echo "=========================================="
echo "Step 2: Annotating data with cognitive levels"
echo "=========================================="
# 设置HTTP代理
export HTTP_PROXY="http://127.0.0.1:7890"
export HTTPS_PROXY="http://127.0.0.1:7890"
export http_proxy="http://127.0.0.1:7890"
export https_proxy="http://127.0.0.1:7890"
cd UltraMedical
python classify_Pref.py
cd ..

echo ""
echo "=========================================="
echo "Step 3: Converting annotated data to training format"
echo "=========================================="
python preprocess_data_pairs.py \
    --input_file UltraMedical/UltraMedical_Pref_sampled_10000_classified.json \
    --output_file ./prepare_data/Ernie-rlhf/Ernie-rlhf-train.jsonl \
    --from_classified

echo ""
echo "=========================================="
echo "Step 4: Starting training (Router + RM)"
echo "=========================================="

MODE="both"

CUDA_VISIBLE_DEVICES=1,2,3,4 \
deepspeed \
    --master_port=29505 \
    main_prefix.py \
    --mode ${MODE}

echo ""
echo "=========================================="
echo "Training completed!"
echo "=========================================="
