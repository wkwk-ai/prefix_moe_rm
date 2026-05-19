import innermoe.config as config
import os, json, random
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from transformers import AutoTokenizer, Trainer, TrainingArguments
from itertools import combinations
from tqdm import tqdm

class HardPromptRMDataset(Dataset):
    """Dataset for Hard Prompt RM - 与MoERMDataset类似，但不需要router"""

    def __init__(self, tokenizer, max_length=512, split="train"):
        """
        Args:
            tokenizer: Tokenizer
            max_length: 最大序列长度
            split: "train" 或 "test"
        """
        self.samples = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.split = split

        # 数据路径
        if split == "train":
            # 强制使用原始 1w 条数据文件，保持与 prefix/baseline 相同规模（9k/1k）
            file_path = os.path.join(config.rawdata_dir, "Ernie-rlhf-train.jsonl")
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Train data not found: {file_path}")
        elif split == "test":
            test_pairs_file_path = os.path.join(config.rawdata_dir, "Ernie-rlhf-test-pairs.jsonl")
            test_original_file_path = os.path.join(config.rawdata_dir, "Ernie-rlhf-test.jsonl")
            
            if os.path.exists(test_pairs_file_path):
                file_path = test_pairs_file_path
            elif os.path.exists(test_original_file_path):
                file_path = test_original_file_path
            else:
                # 如果测试集不存在，返回空数据集
                print(f"[HardPromptRMDataset] Warning: Test data not found. Returning empty dataset.")
                self.samples = []
                return
        else:
            raise ValueError(f"split must be 'train' or 'test', got {split}")

        self.user_token = "<extra_0>"
        self.bot_token = "<extra_1>"
        self.end_token = "<extra_2>"
        
        # 类别映射
        cat_to_id = {cat: i for i, cat in enumerate(config.cat_list)}

        skipped_no_label = 0
        skipped_len_mismatch = 0
        skipped_not_two_responses = 0
        skipped_same_ranks = 0

        desc = f"Loading {split} data"
        with open(file_path, "r") as f:
            for line in tqdm(f, desc=desc):
                try:
                    j = json.loads(line)
                except json.JSONDecodeError:
                    continue
                
                if "src" not in j or "response" not in j:
                    continue
                
                if "label" not in j:
                    skipped_no_label += 1
                    continue
                
                label = j["label"]
                if label not in cat_to_id:
                    continue
                category_id = cat_to_id[label]
                
                responses = j["response"]
                rank = j.get("rank", None)
                
                if not isinstance(responses, list) or len(responses) < 2:
                    skipped_not_two_responses += 1
                    continue
                
                if not isinstance(rank, list) or len(rank) != len(responses):
                    skipped_len_mismatch += 1
                    continue
                
                query = j["src"][-1] if isinstance(j["src"], list) else j["src"]
                
                # 只处理每个prompt的2个response（chosen和rejected）
                if len(responses) == 2:
                    input_ids_list = []
                    attention_mask_list = []
                    
                    for resp in responses:
                        text = f"{self.user_token}{query}{self.bot_token}{resp}{self.end_token}"
                        enc = self.tokenizer(
                            text,
                            truncation=True,
                            max_length=self.max_length,
                            return_tensors="pt"
                        )
                        input_ids_list.append(enc["input_ids"].squeeze(0))
                        attention_mask_list.append(enc["attention_mask"].squeeze(0))
                    
                    self.samples.append({
                        "input_ids": input_ids_list,
                        "attention_mask": attention_mask_list,
                        "labels": torch.tensor(rank, dtype=torch.long),
                        "category_id": category_id,
                        "sample_id": len(self.samples)
                    })
        
        print(f"[HardPromptRMDataset] Loaded {len(self.samples)} samples")
        if split == "train":
            print(f"  Skipped: no_label={skipped_no_label}, len_mismatch={skipped_len_mismatch}, not_two_responses={skipped_not_two_responses}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch):
    """Collate function for Hard Prompt RM"""
    all_input_ids = []
    all_attention_mask = []
    all_labels = []
    all_category_ids = []
    all_sample_ids = []
    
    for item in batch:
        num_responses = len(item["input_ids"])
        sample_id = item["sample_id"]
        category_id = item["category_id"]
        
        all_input_ids.extend(item["input_ids"])
        all_attention_mask.extend(item["attention_mask"])
        all_labels.extend(item["labels"])
        all_category_ids.extend([category_id] * num_responses)
        all_sample_ids.extend([sample_id] * num_responses)
    
    # Padding
    input_ids_padded = nn.utils.rnn.pad_sequence(
        all_input_ids, batch_first=True, padding_value=0
    )
    attention_mask_padded = nn.utils.rnn.pad_sequence(
        all_attention_mask, batch_first=True, padding_value=0
    )
    
    # all_labels是list of tensors，需要先stack
    ranks_tensor = torch.stack(all_labels) if isinstance(all_labels[0], torch.Tensor) else torch.tensor(all_labels)
    sample_ids_tensor = torch.tensor(all_sample_ids, dtype=torch.long)
    
    # 将labels和sample_ids合并为 [N, 2] 格式，方便compute_metrics处理
    labels_with_ids = torch.stack([
        ranks_tensor,  # ranks
        sample_ids_tensor  # sample_ids
    ], dim=1)  # [N, 2]
    
    return {
        "input_ids": input_ids_padded,
        "attention_mask": attention_mask_padded,
        "labels": labels_with_ids,  # [N, 2] -> (rank, sample_id)
        "category_ids": torch.tensor(all_category_ids, dtype=torch.long),
        "sample_ids": torch.tensor(all_sample_ids, dtype=torch.long),
    }


def compute_metrics(eval_pred):
    """计算pairwise准确率"""
    predictions = eval_pred.predictions
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    
    predictions = np.array(predictions).reshape(-1)
    labels_full = np.array(eval_pred.label_ids)
    
    # 从collate_fn返回的格式中提取ranks和sample_ids
    # labels_full可能是 [N, 2] 格式：[:, 0]是rank，[:, 1]是sample_id
    # 或者只是 [N] 格式：只有rank
    if labels_full.ndim == 2 and labels_full.shape[1] >= 2:
        ranks = labels_full[:, 0]
        sample_ids = labels_full[:, 1] if labels_full.shape[1] > 1 else np.zeros_like(ranks)
    else:
        ranks = labels_full if labels_full.ndim == 1 else labels_full.flatten()
        sample_ids = np.zeros_like(ranks)
    
    # 使用DataFrame方便分组计算
    df = pd.DataFrame({
        'score': predictions,
        'rank': ranks,
        'sample_id': sample_ids
    })
    
    total_pairs = 0
    correct_pairs = 0
    
    # 按sample_id分组计算Pairwise Accuracy
    grouped = df.groupby('sample_id')
    for _, group in grouped:
        scores = group['score'].values
        rs = group['rank'].values
        n = len(scores)
        
        # 生成所有可能的pair
        for i in range(n):
            for j in range(i + 1, n):
                if rs[i] != rs[j]:
                    total_pairs += 1
                    # rank越小越好，所以score应该更大
                    if (rs[i] < rs[j] and scores[i] > scores[j]) or \
                       (rs[i] > rs[j] and scores[i] < scores[j]):
                        correct_pairs += 1
    
    acc = correct_pairs / total_pairs if total_pairs > 0 else 0.0
    return {"accuracy": float(acc)}


class HardPromptRMTrainer(Trainer):
    """自定义Trainer，按照 train_module_prefix.py 的逻辑实现"""
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        重写 compute_loss 以处理 labels。
        直接调用模型 forward，避免 Trainer 父类的 in-place 操作问题。
        （按照 train_module_prefix.py 的逻辑）
        """
        # 准备模型输入
        model_inputs = {
            "input_ids": inputs.get("input_ids"),
            "attention_mask": inputs.get("attention_mask"),
        }
        
        # 添加 category_ids（如果存在）
        if "category_ids" in inputs:
            model_inputs["category_ids"] = inputs["category_ids"]
        
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


def compute_pairwise_loss_vectorized(rewards, ranks, sample_ids):
    """向量化的pairwise loss计算"""
    import torch.nn.functional as F
    
    # 广播构造比较矩阵
    id_mask = sample_ids.unsqueeze(1) == sample_ids.unsqueeze(0)  # [N, N]
    id_mask.fill_diagonal_(False)
    
    # 确定优劣关系
    rank_diff = ranks.unsqueeze(1) - ranks.unsqueeze(0)  # [N, N]
    valid_pair_mask = id_mask & (rank_diff < 0)
    
    if not valid_pair_mask.any():
        return rewards.sum() * 0.0
    
    # 计算Reward差异
    r_diff = rewards.unsqueeze(1) - rewards.unsqueeze(0)  # r[i] - r[j]
    valid_diffs = r_diff[valid_pair_mask]
    
    # LogSigmoid Loss
    loss = -F.logsigmoid(valid_diffs).mean()
    
    return loss


def train_hardprompt_rm(hardprompt_rm):
    """
    训练Hard Prompt RM（使用固定的硬prompt，但backbone和value head可训练）
    """
    out_dir = os.path.join(config.out_dir, "hardprompt")
    os.makedirs(out_dir, exist_ok=True)
    
    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path, trust_remote_code=True)
    
    # 加载数据：显式使用 train / test 文件
    train_dataset = HardPromptRMDataset(tokenizer=tokenizer, max_length=512, split="train")
    val_dataset = HardPromptRMDataset(tokenizer=tokenizer, max_length=512, split="test")

    print(f"📊 Dataset split: {len(train_dataset)} train, {len(val_dataset)} val (from explicit train/test files)")
    
    # 打印参数信息
    param_info = hardprompt_rm.get_trainable_parameters()
    print(f"🔧 Trainable parameters: {param_info['trainable_params']:,} / {param_info['all_params']:,} ({param_info['trainable_percentage']:.2f}%)")
    print(f"📝 NOTE: Using fixed hard prompts (not trainable), but backbone and value head are trainable")
    
    # 训练参数
    training_args = TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        logging_dir=os.path.join(out_dir, "logs"),
        logging_steps=50,
        bf16=True,
        report_to="none",
        remove_unused_columns=False,
        save_safetensors=False,
        max_grad_norm=1.0,
        warmup_steps=50,
        lr_scheduler_type="cosine",
        save_total_limit=2,
    )
    
    # 创建Trainer
    trainer = HardPromptRMTrainer(
        model=hardprompt_rm,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )
    
    print("Evaluating baseline before training...")
    init_metrics = trainer.evaluate()
    print(f"Baseline eval metrics: {init_metrics}")

    print("🚀 Starting Hard Prompt RM training...")
    trainer.train()
    
    # 保存最佳模型
    best_model_path = os.path.join(out_dir, "best_model")
    os.makedirs(best_model_path, exist_ok=True)
    
    try:
        trainer.save_model(best_model_path)
        print(f"✅ Best model saved to {best_model_path}")
    except Exception as e:
        print(f"⚠️  Error saving with trainer.save_model: {e}")
        # 手动保存
        model_to_save = trainer.model.module if hasattr(trainer.model, 'module') else trainer.model
        torch.save(model_to_save.state_dict(), os.path.join(best_model_path, "pytorch_model.bin"))
        tokenizer.save_pretrained(best_model_path)
        print(f"✅ Best model saved manually to {best_model_path}")
    
    print("✅ Hard Prompt RM training complete!")

