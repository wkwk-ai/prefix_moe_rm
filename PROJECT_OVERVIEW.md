# Prefix MoE Reward Model — 项目综述

本项目在 DMoERM（ACL 2024 Findings）的思路基础上，构建了一套面向 **医学领域** 的 Reward Model：
1. 用一个 **Router** 将输入按 4 级认知层级（L1\~L4，源自 Anderson & Krathwohl 修订版 Bloom 分类法）分类；
2. 用 **可训练 Prefix Embeddings**（每个 cognitive level 一组）作为 soft prompt 拼接到 backbone 输入前；
3. backbone + value head 输出 scalar reward，使用 pairwise logsigmoid loss + diversity loss 训练；
4. 训练得到的 RM 接入 [verl](https://github.com/volcengine/verl) 作为 GRPO 的奖励信号，对 Actor 做 RL 微调。

仓库同时实现了若干消融对照 RM（Baseline / Hard Prompt），以及与 verl 框架对接所需的 patch 文件。

---

## 1. 目录结构

```
prefix_moe_rm/
├── base_config.py                     # 全局配置（模型路径、数据目录、4 个 cognitive level）
├── main.py                            # 旧版 DMoERM 入口（LoRA 多专家，三阶段），保留作参考
├── main_prefix.py                     # 主入口：训练 Router / Prefix MoE RM / 两者顺序训练
├── main_baseline.py / main_baseline_simple.py
│                                      # 对照实验入口：仅 backbone + value head
├── main_hardprompt.py                 # 对照实验入口：固定 hard prompt + Router argmax
├── train_router.py                    # 旧版独立 Router 训练脚本（已被 router/ 模块取代）
├── test_router.py                     # Router 加载冒烟测试
├── preprocess_data_pairs.py           # 把 multi-response 样本预处理成 pairwise 训练数据
├── ds_config.json / ds_config_router.json
│                                      # DeepSpeed 配置
├── run_train_*.sh                     # 各 RM 训练启动脚本（torchrun / deepspeed）
├── run_finetune_rm_on_cot.sh          # 在 CoT 偏好对上继续微调 Prefix MoE RM
│
├── router/                            # Router 模块（按 cognitive level 分类）
│   ├── config.py
│   ├── modeling.py                    # Router = Qwen2.5-1.5B backbone + Linear classifier head
│   └── train.py                       # HF Trainer + RouterDataset；从 Ernie-rlhf-train.jsonl 读 label
│
├── innermoe/                          # RM 模型与训练
│   ├── config.py
│   ├── load_data.py                   # 旧版数据加载（pickle 缓存、按 category 切分）
│   ├── modeling.py                    # 旧版 DMoERM（base + LoRA experts + final MLP）
│   ├── modeling_prefix.py             # ★ Prefix MoE RM 主模型
│   ├── modeling_baseline.py           # 对照：backbone + value head
│   ├── modeling_hardprompt.py         # 对照：Router argmax 选固定 hard prompt
│   ├── train_module_prefix.py         # ★ Prefix MoE RM 训练循环（Trainer 子类 + pairwise loss）
│   ├── train_module_baseline.py
│   ├── train_module_hardprompt.py
│   ├── train_module.py / train_pipe*.py
│                                      # 旧版三阶段训练
│
├── prepare_data/                      # 训练用 jsonl/parquet（在 .gitignore，本仓库不入库）
│   ├── Ernie-rlhf/                    # 主训练集（mtmc-rlhf 衍生，附 cognitive level）
│   ├── medkgqa/                       # MedKgQA → parquet（带 cognitive_level）
│   └── mixed/                         # Ernie-rlhf + MedKgQA 混合后的 parquet
│
├── scripts/                           # CoT 数据合成脚本
│   ├── generate_cot_for_medkgqa.py    # 用 vLLM 为 MedKgQA 多采样 CoT 回答
│   └── construct_cot_preference_pairs.py
│                                      # 配对 correct/incorrect CoT，构造偏好对
│
├── UltraMedical/                      # UltraMedical 数据集采样与认知层级标注
│   ├── Download.py / Download_Pref.py
│   ├── sample.py / sample_pref.py     # 分层采样
│   ├── classify.py                    # GPT-4o 单线程分类
│   ├── classify_Pref.py               # GPT-4o + ThreadPool 并行分类
│   ├── classify_Pref_claude.py        # Claude 分类版本
│   └── visualization*.py
│
├── benchmarks/                        # 医学 benchmark 处理后的 parquet + 原始评估代码
│   ├── medqa.parquet / mmlu_medical.parquet / medxpertqa.parquet / llmeval_med.parquet
│   ├── llmeval_med_references.json
│   └── raw/                           # LLMEval-Med / MedXpertQA / MedQA / simple-evals 评估代码
│
└── verl_patches/                      # 接入 verl 框架所需的扩展文件
    ├── examples/grpo_trainer/         # 数据准备 + GRPO 启动脚本
    │   ├── prepare_ernie_rlhf_data.py
    │   ├── prepare_medkgqa_for_grpo.py
    │   ├── prepare_mixed_data.py
    │   ├── prepare_benchmark_data.py
    │   ├── classify_medkgqa_with_router.py
    │   ├── run_prefix_moe_rm_grpo.sh   # 用 Prefix MoE RM 做 GRPO
    │   ├── run_baseline_mixed_grpo.sh / run_baseline_mixed_grpo_7b.sh
    │   ├── run_hybrid_grpo.sh
    │   └── README_prefix_moe_rm.md
    └── verl/                          # 对 verl 源码的最小侵入修改
        ├── trainer/main_ppo.py        # 增加 use_prefix_moe 旗标，挂载 Prefix MoE RM
        ├── trainer/ppo/{core_algos.py, ray_trainer.py, reward.py}
        ├── workers/fsdp_workers.py    # 让 RewardModelWorker 接受 Prefix MoE 模型
        ├── workers/reward_model/prefix_moe_wrapper.py
        │                              # 把 InnerMoERM 包装成 token-classifier-like 接口
        ├── workers/reward_manager/{hybrid.py, __init__.py}
        │                              # 在 RL 时混合 RM 分数 + MCQ 规则奖励
        └── utils/reward_score/{__init__.py, medical_benchmark.py}
                                       # MCQ 答案抽取（A/B/C/D）+ benchmark 评估打分
```

`.gitignore` 中排除：`__pycache__/`、`results/`（checkpoint 26GB+）、`prepare_data/`、`UltraMedical/UltraMedical*.{csv,json}` 等大体积数据。

---

## 2. 端到端流程

整套流水线分四步走，可按需运行其中任一步。

### Step 1 — 数据准备：拿到带 cognitive level 的 pairwise 训练集

入口路径有两条，最终都生成 `prepare_data/Ernie-rlhf/Ernie-rlhf-train.jsonl`（schema：`{src, response[2], rank[2], label}`，其中 `label ∈ {L1_recall, L2_analysis, L3_decision, L4_synthesis}`）。

**Path A — UltraMedical-Preference + GPT/Claude 自动标注**

| 脚本 | 作用 |
| --- | --- |
| `UltraMedical/Download.py` / `Download_Pref.py` | 从 HuggingFace 下载 UltraMedical / UltraMedical-Preference |
| `UltraMedical/sample_pref.py` | 按 `label_type` 分层采样 10000 条 |
| `UltraMedical/classify_Pref.py` / `classify_Pref_claude.py` | 调 GPT-4o / Claude 用 Bloom 4 级模板分类（API key 改成读 `OPENAI_API_KEY` 环境变量） |
| `UltraMedical/visualization*.py` | 看采样和标注后的层级分布 |
| `preprocess_data_pairs.py --from_classified` | 把 `{prompt, chosen, rejected, label, cognitive_level}` 转成 `{src, response[2], rank[2], label}` 的 jsonl |

**Path B — 自带 cognitive level 的 MedKgQA（同时是开发集）**

| 脚本 | 作用 |
| --- | --- |
| `scripts/generate_cot_for_medkgqa.py` | 用 vLLM 对 MedKgQA 每题采样 `n` 条 CoT 回答，附带答案抽取、正确性标注 |
| `scripts/construct_cot_preference_pairs.py` | 同一题里把 correct CoT × incorrect CoT 配对，附带 cognitive_level 输出 jsonl |
| `cat ... > combined-train.jsonl` | 合并到 Ernie-rlhf-train.jsonl 后做 fine-tune（见 `run_finetune_rm_on_cot.sh`） |

**统一格式化**：`preprocess_data_pairs.py` 把多 response 样本展开成每对一行（保证训练时 pairwise loss 有效）。

### Step 2 — 训练 Router（cognitive level 分类器）

* 入口：`python main_prefix.py --mode router` 或 `bash run_train_*.sh`（脚本内部走 `main_prefix.py` / `main_baseline_simple.py` / `main_hardprompt.py`）。
* 代码：[router/modeling.py](router/modeling.py)、[router/train.py](router/train.py)、[router/config.py](router/config.py)。
* 模型：`Qwen2.5-1.5B-Instruct` backbone + `Linear(hidden, 4)`，取最后一个非 pad token 的 hidden state 做分类。
* 数据：`Ernie-rlhf-train.jsonl` 中带 `label` 字段的样本，9:1 切分训练/验证。
* 输出：`results/router/router.bin` + `router_config.json`。

### Step 3 — 训练 Prefix MoE Reward Model

* 入口：`python main_prefix.py --mode rm --router_ckpt ./results/router` 或 `bash run_train_prefix.sh`（torchrun 4 卡）。
* 代码：[innermoe/modeling_prefix.py](innermoe/modeling_prefix.py)、[innermoe/train_module_prefix.py](innermoe/train_module_prefix.py)、[innermoe/config.py](innermoe/config.py)。
* 模型组成：
  - **Router**（载入 Step 2 的 ckpt，冻结），输出 4 类 softmax 概率 `cat_probs`（温度 0.1）；
  - **Prefix Embeddings** `[C=4, max_L, D]`（可训练 `nn.Parameter`，由各 cognitive level 的文本 prompt 初始化）；
  - **Backbone** `Qwen2.5-1.5B-Instruct`（全量训练）+ `value_head: Linear(hidden, 1)` + `PReLU`。
* 前向：
  1. Router 给出 `cat_probs`；
  2. 用概率做 soft 加权得到 fused prefix，再 cat 到 token embeddings 前；
  3. backbone 取最后一个非 pad 位置 hidden → value head → scalar reward。
* Loss：
  - 向量化 pairwise `-logsigmoid(r_chosen - r_rejected)`（按 `sample_id` 分组，rank 数值越小越好）；
  - + `diversity_loss_weight (=0.1) × mean cosine similarity` 防止 4 组 prefix 坍缩成同一个。
* `train_module_prefix.py` 中实现：
  - `MoERMDataset` 支持 train / test split，自动跳过 ranks 相同的无效对；
  - `MoETrainer` 子类化 `transformers.Trainer`，重写 `compute_loss`；
  - `compute_metrics` 按 `sample_id` 分组计算 pairwise accuracy；
  - 训练超参：`lr=5e-5`、`num_epochs=3`、`bf16=True`、`cosine` 调度、`max_grad_norm=1.0`、`save_total_limit=2`；
  - `load_best_model_at_end=True`、`metric_for_best_model=accuracy`；
  - 输出：`results/innermoe/unified/best_model/pytorch_model.bin`。
* 微调入口：`run_finetune_rm_on_cot.sh` 用 `--rm_ckpt ./results/innermoe/unified/best_model --learning_rate 1e-5` 在合并后的 CoT 数据上继续训。

### Step 4 — 接入 verl 做 GRPO RL 训练

`verl_patches/` 提供对 [verl](https://github.com/volcengine/verl) 框架的最小侵入修改，把 Prefix MoE RM 作为 GRPO 的 reward source。

**先把仓库克隆到 `verl/`，再把 `verl_patches/` 内文件覆盖到对应位置**（路径一一对应）：

```
verl_patches/verl/...                    → verl/verl/...
verl_patches/examples/grpo_trainer/...   → verl/examples/grpo_trainer/...
```

关键变更：

| 文件 | 改动 |
| --- | --- |
| `verl/trainer/main_ppo.py` | 识别 `reward_model.model.use_prefix_moe=True`，把 worker 切到 `PrefixMoERewardModelWorker` |
| `verl/workers/reward_model/prefix_moe_wrapper.py` | 加载 `InnerMoERM`，把 `forward` 适配成 `TokenClassifierOutput`（last token reward） |
| `verl/workers/fsdp_workers.py` | RewardModelWorker 支持 use_prefix_moe 旗标和 router_ckpt_path |
| `verl/workers/reward_manager/hybrid.py` | RM 分数与 MCQ 规则奖励的混合 reward manager |
| `verl/utils/reward_score/medical_benchmark.py` | 医学 MCQ 答案抽取 + benchmark accuracy 评估 |
| `verl/trainer/ppo/{core_algos,ray_trainer,reward}.py` | 适配 hybrid reward / benchmark 评估钩子 |

**数据准备脚本**（位于 `verl_patches/examples/grpo_trainer/`）：

* `prepare_ernie_rlhf_data.py`：`Ernie-rlhf-{train,test}.jsonl → .parquet`；
* `prepare_medkgqa_for_grpo.py`：MedKgQA → parquet（保留 cognitive_level / ground truth）；
* `prepare_mixed_data.py`：合并 Ernie-rlhf + MedKgQA，按比例混采；
* `prepare_benchmark_data.py`：把 MedQA / MMLU-Medical / MedXpertQA / LLMEval-Med 转 parquet 供训练中的 benchmark eval 使用；
* `classify_medkgqa_with_router.py`：用训练好的 Router 对 MedKgQA 推 cognitive level。

**GRPO 启动脚本**：

| 脚本 | 说明 |
| --- | --- |
| `run_prefix_moe_rm_grpo.sh` | ★ 用 Prefix MoE RM 做 GRPO（actor=Qwen2.5-1.5B，RM=`results/innermoe/unified/best_model`，benchmarks 自动评估）。`reward_model.model.use_prefix_moe=True` 是核心旗标。 |
| `run_baseline_mixed_grpo.sh` / `_7b.sh` | 不用 Prefix MoE，跑 baseline GRPO 做对照 |
| `run_hybrid_grpo.sh` | hybrid reward manager：RM 分数 + 规则奖励 |

### 对照实验

| 实验 | 入口 | 启动脚本 | 模型差异 |
| --- | --- | --- | --- |
| Prefix MoE RM | `main_prefix.py` | `run_train_prefix.sh` | Router(soft) + trainable prefix + backbone + head |
| Baseline RM | `main_baseline_simple.py` / `main_baseline.py` | `run_train_baseline.sh` | 仅 backbone + value head |
| Hard Prompt RM | `main_hardprompt.py` | `run_train_hardprompt.sh` | Router(argmax) + 固定 hard prompt + 可训练 backbone + head |

三者共用 `train_module_*.py` 中相同的 pairwise loss / `MoERMDataset` 框架，差别只在 `modeling_*.py`。

---

## 3. 关键依赖与运行说明

* Python `3.11.7`，CUDA `11.8`；详见 [requirements.txt](requirements.txt)。
* base model 路径硬编码在 [base_config.py](base_config.py)：`/mnt/data/model/Qwen2.5-1.5B-Instruct`，请按本机情况调整。
* 数据 / checkpoint 体积较大，本仓库不入库；目录结构需自行准备：
  - `prepare_data/Ernie-rlhf/Ernie-rlhf-{train,test}.jsonl`
  - `results/router/router.bin`、`results/innermoe/unified/best_model/`
  - `benchmarks/{medqa,mmlu_medical,medxpertqa,llmeval_med}.parquet`
* 涉及 LLM 标注的脚本（`UltraMedical/classify*.py`、`benchmarks/raw/LLMEval-Med-main/evaluate/Evaluate.py`）都已改成读 `OPENAI_API_KEY` / `OPENAI_BASE_URL` 环境变量。

---

## 4. 典型工作流

完整链路（从零开始）：

```bash
# 0. 准备 base model (Qwen2.5-1.5B-Instruct) 与原始数据
#    把数据放到 prepare_data/Ernie-rlhf/

# 1. 训练 Router
python main_prefix.py --mode router
# → results/router/router.bin

# 2. 训练 Prefix MoE RM
bash run_train_prefix.sh
# → results/innermoe/unified/best_model/

# 3. (可选) CoT 数据增广 + 微调
python scripts/generate_cot_for_medkgqa.py --input ... --output ...
python scripts/construct_cot_preference_pairs.py --responses ... --output prepare_data/Ernie-rlhf/MedKgQA-cot-pairs.jsonl
cat prepare_data/Ernie-rlhf/Ernie-rlhf-train.jsonl prepare_data/Ernie-rlhf/MedKgQA-cot-pairs.jsonl \
  > prepare_data/Ernie-rlhf/combined-train.jsonl
bash run_finetune_rm_on_cot.sh

# 4. 接入 verl 做 GRPO
#    git clone verl 并把 verl_patches/ 内容覆盖到对应位置
bash verl/examples/grpo_trainer/run_prefix_moe_rm_grpo.sh
```

对照实验只需切换 Step 2 的启动脚本：

```bash
bash run_train_baseline.sh        # Baseline RM
bash run_train_hardprompt.sh      # Hard Prompt RM
```
