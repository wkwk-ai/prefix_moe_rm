# Prefix MoE Reward Model

医学领域 Reward Model：以 cognitive level（L1\~L4）为切分粒度，**Router + 可训练 Prefix Embeddings + Backbone + Value Head**，并接入 [verl](https://github.com/volcengine/verl) 的 GRPO 训练管线作为 reward source。

完整流水线（数据准备 → Router → Prefix MoE RM → GRPO RL）、模块说明与各脚本职责见 [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md)。

## 快速开始

```bash
# 环境
conda create -n env python=3.11.7 && conda activate env
conda install cudatoolkit=11.8
pip install -r requirements.txt

# 训练 Router
python main_prefix.py --mode router

# 训练 Prefix MoE RM（4 卡）
bash run_train_prefix.sh

# 在 verl 内做 GRPO（先 clone verl 并把 verl_patches/ 覆盖进去）
bash verl/examples/grpo_trainer/run_prefix_moe_rm_grpo.sh
```

> 模型路径、数据目录写死在 [base_config.py](base_config.py)，按本机情况调整。

## 关于 DMoERM

本仓库最早 fork 自 [DMoERM (ACL 2024 Findings)](https://arxiv.org/abs/2403.01197) 的实现，旧版三阶段 LoRA-MoE 训练代码保留在 [innermoe/modeling.py](innermoe/modeling.py)、[innermoe/train_pipe.py](innermoe/train_pipe.py) 与 [main.py](main.py) 中，仅作历史参考；当前推荐入口是 `main_prefix.py`。
