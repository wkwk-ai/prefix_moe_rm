#!/bin/bash
# DeepSpeed training script for Router + Reward Model
# Uses GPUs 5, 6, 7, 8

# DeepSpeed configuration file
DS_CONFIG="./ds_config.json"

# Training mode (both = router + rm)
MODE="both"

# Run training with DeepSpeed
# Use --include to specify specific GPUs (localhost:5,6,7,8)
# Note: DeepSpeed will be automatically enabled if ds_config.json is specified in TrainingArguments
deepspeed --include localhost:5,6,7,8 \
    --master_port=29500 \
    main_prefix.py \
    --mode ${MODE}

