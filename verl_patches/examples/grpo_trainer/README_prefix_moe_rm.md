# Prefix MoE Reward Model for GRPO Training

This directory contains the implementation for using Prefix MoE Reward Model with verl's GRPO trainer.

## Overview

The Prefix MoE Reward Model uses a router to select category-specific prefix embeddings and compute rewards. This implementation integrates it with verl's training framework with minimal modifications to the verl source code.

## Files

1. **`prefix_moe_reward_model.py`**: The core reward model class that implements the reward computation logic.
2. **`prefix_moe_reward_model_worker.py`**: Worker class that loads the Prefix MoE model and integrates it with verl's FSDP framework.
3. **`prepare_ernie_rlhf_data.py`**: Script to convert Ernie-rlhf JSONL data to parquet format for verl.
4. **`run_prefix_moe_rm_grpo.sh`**: Training script for GRPO with Prefix MoE Reward Model.

## Setup

1. Make sure you have trained the Prefix MoE Reward Model and it's saved at:
   ```
   /home/haowang/prefix_moe_rm/results/innermoe/unified/best_model
   ```

2. Make sure the router checkpoint is available (optional, will try to find automatically):
   ```
   /home/haowang/prefix_moe_rm/results/router
   ```

3. Prepare the training data:
   ```bash
   python verl/examples/grpo_trainer/prepare_ernie_rlhf_data.py \
       /home/haowang/prefix_moe_rm/prepare_data/Ernie-rlhf/Ernie-rlhf-train.jsonl \
       /home/haowang/prefix_moe_rm/prepare_data/Ernie-rlhf/Ernie-rlhf-train.parquet
   ```

## Usage

1. Update the paths in `run_prefix_moe_rm_grpo.sh`:
   - `ACTOR_MODEL_PATH`: Path to your actor model (e.g., Qwen2.5-1.5B-Instruct)
   - `RM_MODEL_PATH`: Path to the Prefix MoE Reward Model checkpoint
   - `ROUTER_CKPT_PATH`: Path to router checkpoint (optional)

2. Run the training script:
   ```bash
   cd /home/haowang/prefix_moe_rm
   bash verl/examples/grpo_trainer/run_prefix_moe_rm_grpo.sh
   ```

## Configuration

The key configuration for using Prefix MoE Reward Model is:
```yaml
reward_model:
  enable: True
  use_prefix_moe: True  # Enable Prefix MoE Reward Model
  strategy: fsdp
  model:
    path: /path/to/prefix_moe_rm/checkpoint
    router_ckpt_path: /path/to/router/checkpoint  # Optional
```

## How It Works

1. **Model Loading**: The `PrefixMoERewardModelWorker` loads the trained Prefix MoE model, including:
   - The backbone model
   - The router for category selection
   - The prefix embeddings for each category
   - The value head for reward computation

2. **Reward Computation**: When computing rewards:
   - The router processes the input to get category probabilities
   - Prefix embeddings are weighted by category probabilities and fused
   - The fused prefix is concatenated with the input tokens
   - The model processes the combined input and outputs reward scores

3. **Integration**: The implementation inherits from verl's `RewardModelWorker` and integrates seamlessly with verl's training pipeline, supporting:
   - FSDP/FSDP2 for distributed training
   - Dynamic batch sizing
   - Sequence parallel processing

## Notes

- The implementation makes minimal changes to verl source code (only one addition in `main_ppo.py` to support `use_prefix_moe` flag)
- The Prefix MoE model expects input in the format: `<extra_0>{query}<extra_1>{response}<extra_2>`
- The router checkpoint is optional - if not provided, the code will try to find it automatically or use uniform category distribution

