import innermoe.config as config
import os, json, pickle, random
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from transformers import AutoTokenizer, Trainer, TrainingArguments
from itertools import combinations
from tqdm import tqdm

# ---------------------- 1. 指标计算函数 ----------------------
def compute_metrics(eval_pred):
    """
    计算 pairwise 准确率
    eval_pred.label_ids 的形状为 [N, 2]，其中：
      - col 0: Rank (真实标签)
      - col 1: Sample ID (用于分组)
    """
    predictions = eval_pred.predictions
    # 某些模型可能返回 tuple (logits, hidden_states, ...)，取第一项
    if isinstance(predictions, tuple):
        predictions = predictions[0]
        
    # 展平预测值
    preds = np.array(predictions).reshape(-1)
    
    # 获取 labels 和 sample_ids
    labels_full = np.array(eval_pred.label_ids)
    ranks = labels_full[:, 0]
    sample_ids = labels_full[:, 1]

    # 额外一致性检查，防止评估样本数异常
    unique_samples = np.unique(sample_ids)
    expected_unique = len(sample_ids) // 2  # 每个样本恰好 2 条 response
    if len(unique_samples) != expected_unique:
        print(f"[Metric Warning] unique_samples={len(unique_samples)}, total_rows={len(sample_ids)}, expected_unique={expected_unique}")

    # 使用 DataFrame 方便分组 (比纯 Python 循环快且鲁棒)
    df = pd.DataFrame({
        'score': preds,
        'rank': ranks,
        'sample_id': sample_ids
    })

    total_pairs = 0
    correct_pairs = 0

    # 按 sample_id 分组计算 Pairwise Accuracy
    # 这样即使 Batch Size=1 或者 DeepSpeed Gather 后顺序打乱，也能正确匹配
    grouped = df.groupby('sample_id')
    
    for _, group in grouped:
        scores = group['score'].values
        rs = group['rank'].values
        n = len(scores)
        
        if n < 2: continue

        for i, j in combinations(range(n), 2):
            # 如果 rank 相同，不构成偏序对，跳过
            if rs[i] == rs[j]:
                continue
            
            total_pairs += 1
            
            # 逻辑：Rank 数值越小越好。
            # 如果 rank_i < rank_j (i 优于 j)，则 score_i 应该 > score_j
            # 差异积：(rank_i - rank_j) * (score_i - score_j) < 0 代表方向正确
            if (rs[i] - rs[j]) * (scores[i] - scores[j]) < 0:
                correct_pairs += 1

    acc = correct_pairs / total_pairs if total_pairs > 0 else 0.0
    
    # 打印日志方便在 console 看到
    print(f"\n[Eval] Pairs: {total_pairs} | Correct: {correct_pairs} | Acc: {acc:.4f}")
    
    return {
        "accuracy": acc,
    }

# ---------------------- 2. 数据集定义 ----------------------
class MoERMDataset(Dataset):
    def __init__(self, tokenizer, max_length=512, split="train"):
        """
        Args:
            tokenizer: Tokenizer
            max_length: 最大序列长度
            split: "train" 或 "test"，指定加载训练集还是测试集
        """
        self.samples = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.split = split

        self.user_token = "<extra_0>"
        self.bot_token = "<extra_1>"
        self.end_token = "<extra_2>"

        # 根据 split 选择文件路径
        if split == "train":
            # Use custom path if configured, otherwise default
            custom_path = getattr(config, "train_data_path", None)
            if custom_path and os.path.exists(custom_path):
                file_path = custom_path
                print(f"[Dataset] Using custom train data: {file_path}")
            else:
                file_path = os.path.join(config.rawdata_dir, "Ernie-rlhf-train.jsonl")
                print(f"[Dataset] Using default data: {file_path}")
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Train data not found: {file_path}")
        elif split == "test":
            # 测试集：优先使用预处理后的pairs数据，如果不存在则使用原始测试集
            test_pairs_file_path = os.path.join(config.rawdata_dir, "Ernie-rlhf-test-pairs.jsonl")
            test_original_file_path = os.path.join(config.rawdata_dir, "Ernie-rlhf-test.jsonl")
            
            if os.path.exists(test_pairs_file_path):
                file_path = test_pairs_file_path
                print(f"[Dataset] Using preprocessed test pairs data: {file_path}")
            elif os.path.exists(test_original_file_path):
                file_path = test_original_file_path
                print(f"[Dataset] Using original test data: {file_path}")
                print(f"[Dataset] Warning: Test data may have variable number of responses per prompt.")
            else:
                # 如果测试集不存在，设置空数据集，让调用者处理
                print(f"[Dataset] Warning: Test data not found. Neither {test_pairs_file_path} nor {test_original_file_path} found.")
                print(f"[Dataset] Will use validation split from training data instead.")
                self.samples = []  # 设置为空数据集
                return  # 提前返回，不加载数据
        else:
            raise ValueError(f"split must be 'train' or 'test', got {split}")

        skipped_no_rank = 0
        skipped_len_mismatch = 0
        skipped_not_two_responses = 0
        skipped_same_ranks = 0

        desc = f"Loading {split} data"
        with open(file_path, "r") as f:
            for line in tqdm(f, desc=desc):
                try:
                    j = json.loads(line)
                except:
                    continue

                if "src" not in j or "response" not in j:
                    continue
                if "rank" not in j or j['rank'] is None:
                    skipped_no_rank += 1
                    continue

                query = j["src"][-1] if isinstance(j["src"], list) else j["src"]
                responses = j["response"]
                rank = j["rank"]

                # 确保每个样本恰好有两个response（新数据格式要求）
                if not isinstance(responses, list) or len(responses) != 2:
                    skipped_not_two_responses += 1
                    continue
                if not isinstance(rank, list) or len(rank) != 2:
                    skipped_len_mismatch += 1
                    continue
                
                # 跳过 ranks 相同的数据（无效数据，无法计算 pairwise loss）
                if rank[0] == rank[1]:
                    skipped_same_ranks += 1
                    continue

                self.samples.append({
                    "query": query,
                    "responses": responses,
                    "rank": rank
                })

        # 随机打乱
        random.shuffle(self.samples)
        print(f"[Dataset] Loaded {len(self.samples)} valid samples.")
        if skipped_not_two_responses > 0:
            print(f"[Dataset] Skipped {skipped_not_two_responses} samples with != 2 responses.")
        if skipped_len_mismatch > 0:
            print(f"[Dataset] Skipped {skipped_len_mismatch} samples with rank/response length mismatch.")
        if skipped_same_ranks > 0:
            print(f"[Dataset] Skipped {skipped_same_ranks} samples with same ranks (invalid for pairwise loss).")
        if skipped_no_rank > 0:
            print(f"[Dataset] Skipped {skipped_no_rank} samples with no rank.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # 传入 idx 作为 sample_id，确保全局或局部唯一性足以区分样本
        sample = self.samples[idx]
        query = sample["query"]
        responses = sample["responses"]
        rank = sample["rank"]

        input_ids_list = []
        attention_mask_list = []
        
        for resp in responses:
            text = f"{self.user_token}{query}{self.bot_token}{resp}{self.end_token}"
            enc = self.tokenizer(
                text,
                max_length=self.max_length,
                padding=False,
                truncation=True,
                return_tensors="pt"
            )
            input_ids_list.append(enc["input_ids"].squeeze(0))
            attention_mask_list.append(enc["attention_mask"].squeeze(0))

        return {
            "input_ids": input_ids_list,
            "attention_mask": attention_mask_list,
            "labels": torch.tensor(rank, dtype=torch.long),
            "sample_id": idx  # 关键：传递样本ID
        }

# ---------------------- 3. Collate Function ----------------------
def collate_fn(batch):
    """
    将 batch 中的 rank 和 sample_id 拼接到 labels 中。
    Labels Shape: [Total_Responses, 2] -> (Rank, SampleID)
    """
    all_responses_input_ids = []
    all_responses_attention_mask = []
    all_labels = []
    
    for item in batch:
        num_responses = len(item["input_ids"])
        sample_id = item["sample_id"]
        
        all_responses_input_ids.extend(item["input_ids"])
        all_responses_attention_mask.extend(item["attention_mask"])
        
        # 构造 Rank
        ranks = item["labels"] # [num_responses]
        # 构造 Sample ID (重复 num_responses 次)
        ids = torch.full((num_responses,), sample_id, dtype=torch.long)
        
        # 拼接: [num_responses, 2]
        combined_labels = torch.stack([ranks, ids], dim=1)
        all_labels.append(combined_labels)

    # Padding
    input_ids_padded = nn.utils.rnn.pad_sequence(
        all_responses_input_ids, batch_first=True, padding_value=0
    )
    attention_mask_padded = nn.utils.rnn.pad_sequence(
        all_responses_attention_mask, batch_first=True, padding_value=0
    )
    
    # 拼接 Labels
    labels_batch = torch.cat(all_labels, dim=0)  # [total_responses, 2]
    
    return {
        "input_ids": input_ids_padded,
        "attention_mask": attention_mask_padded,
        "labels": labels_batch,
    }

# ---------------------- 4. 自定义 Trainer ----------------------
class MoETrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        重写 compute_loss 以处理 [N, 2] 形状的 labels。
        直接调用模型 forward，避免 Trainer 父类的 in-place 操作问题。
        """
        # 准备模型输入
        model_inputs = {
            "input_ids": inputs.get("input_ids"),
            "attention_mask": inputs.get("attention_mask"),
        }
        
        # labels 格式是 [N, 2] -> (Rank, SampleID)，模型可以处理
        if "labels" in inputs:
            model_inputs["labels"] = inputs["labels"]
        
        # 直接调用模型 forward，获取 loss 和 outputs
        outputs = model(**model_inputs)
        loss = outputs.get("loss")
        
        # 如果 loss 是 None（不应该发生，但为了安全）
        # 使用从模型参数派生的零 loss，避免创建 leaf variable
        if loss is None:
            # 从模型参数创建一个非 leaf variable 的零 loss
            first_param = next(model.parameters())
            loss = first_param.sum() * 0.0
        
        # 如果需要返回 outputs
        if return_outputs:
            return (loss, outputs)
        return loss

# ---------------------- 5. 保存辅助函数 ----------------------
def save_model_safe(trainer, output_dir):
    """安全保存模型，优先尝试 save_model，失败则回退到 torch.save"""
    try:
        trainer.save_model(output_dir)
    except Exception as e:
        print(f"Standard save failed ({e}), using fallback torch.save...")
        os.makedirs(output_dir, exist_ok=True)
        model_to_save = trainer.model.module if hasattr(trainer.model, 'module') else trainer.model
        torch.save(model_to_save.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
        if trainer.tokenizer:
            trainer.tokenizer.save_pretrained(output_dir)

# ---------------------- 6. 主训练函数 ----------------------
def train_rm(innerMoERM):
    out_dir = os.path.join(config.out_dir, "unified")
    os.makedirs(os.path.join(out_dir, "ckpts"), exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path, trust_remote_code=True)

    # --- 加载数据 ---
    # 训练集：从训练文件加载
    train_dataset = MoERMDataset(tokenizer=tokenizer, max_length=512, split="train")
    # 测试集：从测试文件加载，如果不存在则从训练集分割
    val_dataset = MoERMDataset(tokenizer=tokenizer, max_length=512, split="test")
    
    if len(val_dataset) == 0:
        # 如果测试集为空（文件不存在），从训练集中分割一部分作为验证集
        print(f"[Train] Test dataset not found, splitting training data for validation...")
        train_size = int(len(train_dataset) * 0.9)
        val_size = len(train_dataset) - train_size
        # 固定随机种子，避免多卡进程切分不一致导致验证集样本数累加
        generator = torch.Generator().manual_seed(42)
        train_dataset, val_dataset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size], generator=generator
        )
        print(f"[Train] Split: {train_size} train, {val_size} validation")
    
    print(f"[Train] Train samples: {len(train_dataset)}")
    print(f"[Train] Validation samples: {len(val_dataset)}")

    # --- TrainingArguments ---
    training_args = TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_epochs,
        
        # === 自动化评估配置 ===
        eval_strategy="epoch",          # 每个 epoch 结束时评估
        save_strategy="epoch",          # 每个 epoch 结束时保存 checkpoint
        load_best_model_at_end=True,    # 训练结束后加载 Acc 最高的模型
        metric_for_best_model="accuracy",
        greater_is_better=True,
        
        logging_dir=os.path.join(out_dir, "logs"),
        logging_steps=50,
        bf16=True,
        report_to="none",
        # deepspeed="./ds_config.json", # 如果使用 deepspeed 启动，这里会自动生效或需要显式指定json路径
        
        # 训练稳定性配置
        max_grad_norm=1.0,
        warmup_steps=50,
        lr_scheduler_type="cosine",
        save_total_limit=2,             # 最多保留 2 个 checkpoint
        save_safetensors=False,         # 避免共享内存问题
        
        # === 关键配置 ===
        # 设为 False，防止 Trainer 自动移除 inputs 中的 'labels' (因为我们对其进行了自定义修改)
        # 虽然标准 Trainer 也会保留 labels，但显式设置更安全
        remove_unused_columns=False, 
    )

    # --- 使用自定义 Trainer ---
    trainer = MoETrainer(
        model=innerMoERM,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
    )
    
    # 打印参数统计
    total_params = sum(p.numel() for p in innerMoERM.parameters())
    trainable_params = sum(p.numel() for p in innerMoERM.parameters() if p.requires_grad)
    print(f"\n📊 Model Params: Total={total_params:,}, Trainable={trainable_params:,} ({trainable_params/total_params:.2%})")

    # --- 开始训练 ---
    print("\n🚀 Starting training...")
    
    # 初始评估 (可选，为了看 Baseline)
    print("Evaluating baseline...")
    init_metrics = trainer.evaluate()
    print(f"Baseline metrics: {init_metrics}")

    trainer.train()
    
    # --- 训练结束 ---
    print(f"\n✅ Training completed. Best metric found: {trainer.state.best_metric}")
    
    # 保存最佳模型 (load_best_model_at_end=True 保证此时 trainer.model 已经是最佳权重)
    final_output_dir = os.path.join(out_dir, "best_model")
    save_model_safe(trainer, final_output_dir)
    print(f"💾 Best model saved to: {final_output_dir}")
