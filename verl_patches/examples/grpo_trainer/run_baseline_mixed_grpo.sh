#!/bin/bash
# Baseline: GRPO with mixed data (Ernie-rlhf + MedKgQA)
# MCQ uses pure rule-based reward (alpha=1.0), open-ended uses RM reward.
# No cognitive levels, no RM fine-tuning -- simplest possible setup.
set -x
export CUDA_VISIBLE_DEVICES=0,1,2,3,5,6,7,8

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.." || exit 1

# ---------- Configuration ----------
ACTOR_MODEL_PATH=/mnt/data/model/Qwen2.5-1.5B-Instruct
RM_MODEL_PATH=/home/haowang/prefix_moe_rm/results/innermoe/unified/best_model
ROUTER_CKPT_PATH=/home/haowang/prefix_moe_rm/results/router

TRAIN_DATA_PATH=/home/haowang/prefix_moe_rm/prepare_data/mixed/mixed-train.parquet
VAL_DATA_PATH=/home/haowang/prefix_moe_rm/prepare_data/Ernie-rlhf/Ernie-rlhf-test.parquet
BENCHMARK_DIR=/home/haowang/prefix_moe_rm/benchmarks

# ---------- Data preparation ----------
MEDKGQA_PARQUET=/home/haowang/prefix_moe_rm/prepare_data/medkgqa/medkgqa-train.parquet
ERNIE_PARQUET=/home/haowang/prefix_moe_rm/prepare_data/Ernie-rlhf/Ernie-rlhf-train.parquet

if [ ! -f "$ERNIE_PARQUET" ]; then
    echo "Preparing Ernie-rlhf data..."
    python3 examples/grpo_trainer/prepare_ernie_rlhf_data.py \
        /home/haowang/prefix_moe_rm/prepare_data/Ernie-rlhf/Ernie-rlhf-train.jsonl \
        "$ERNIE_PARQUET"
fi

if [ ! -f "$TRAIN_DATA_PATH" ]; then
    echo "Preparing mixed training data..."
    python3 examples/grpo_trainer/prepare_mixed_data.py \
        --ernie "$ERNIE_PARQUET" \
        --medkgqa "$MEDKGQA_PARQUET" \
        --output "$TRAIN_DATA_PATH" \
        --ratio 1.0
fi

if [ ! -f "$VAL_DATA_PATH" ]; then
    echo "Preparing validation data..."
    python3 examples/grpo_trainer/prepare_ernie_rlhf_data.py \
        /home/haowang/prefix_moe_rm/prepare_data/Ernie-rlhf/Ernie-rlhf-test.jsonl \
        "$VAL_DATA_PATH"
fi

if [ ! -f "$BENCHMARK_DIR/medqa.parquet" ] || [ ! -f "$BENCHMARK_DIR/mmlu_medical.parquet" ]; then
    echo "Preparing benchmark data..."
    python3 examples/grpo_trainer/prepare_benchmark_data.py --output_dir "$BENCHMARK_DIR"
fi

BENCHMARK_FILES=""
for f in "$BENCHMARK_DIR"/medqa.parquet "$BENCHMARK_DIR"/mmlu_medical.parquet "$BENCHMARK_DIR"/medxpertqa.parquet; do
    if [ -f "$f" ]; then
        if [ -z "$BENCHMARK_FILES" ]; then
            BENCHMARK_FILES="$f"
        else
            BENCHMARK_FILES="$BENCHMARK_FILES,$f"
        fi
    fi
done

export CUDA_DEVICE_MAX_CONNECTIONS=1

# ---------- Launch GRPO ----------
# Key difference from original: reward_model.reward_manager=hybrid with alpha_override=1.0
# This means: MCQ data -> pure rule-based reward, open-ended data -> RM reward
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$TRAIN_DATA_PATH" \
    data.val_files="$VAL_DATA_PATH" \
    data.return_raw_chat=True \
    data.train_batch_size=32 \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path="$ACTOR_MODEL_PATH" \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.02 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    reward_model.enable=True \
    reward_model.strategy=fsdp \
    reward_model.model.path="$RM_MODEL_PATH" \
    +reward_model.model.use_prefix_moe=True \
    +reward_model.model.router_ckpt_path="$ROUTER_CKPT_PATH" \
    reward_model.model.trust_remote_code=True \
    reward_model.micro_batch_size_per_gpu=1 \
    reward_model.use_dynamic_bsz=False \
    reward_model.forward_max_token_len_per_gpu=512 \
    reward_model.reward_manager=hybrid \
    +reward_model.reward_kwargs.alpha_override=1.0 \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='verl_grpo_baseline_mixed' \
    trainer.experiment_name='qwen2_1.5b_mixed_rule_rm_baseline' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=200 \
    trainer.test_freq=50 \
    trainer.total_epochs=3 \
    trainer.val_before_train=True \
    trainer.rollout_data_dir=/home/haowang/prefix_moe_rm/verl/rollout_outputs_baseline \
    trainer.validation_data_dir=/home/haowang/prefix_moe_rm/verl/validation_outputs_baseline \
    trainer.benchmark_data_dir=/home/haowang/prefix_moe_rm/verl/benchmark_outputs_baseline \
    ${BENCHMARK_FILES:+trainer.benchmark_files="[$BENCHMARK_FILES]"} $@
