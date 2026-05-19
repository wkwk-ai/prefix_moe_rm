import innermoe.config as config
import os, json, random
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from transformers import AutoTokenizer, Trainer, TrainingArguments

class BaselineRMDataset(Dataset):
    """Lazy-tokenized pairwise dataset for the baseline RM - 按照 train_module_prefix.py 的逻辑"""

    def __init__(self, data_path):
        self.samples = []
        self.user_token = "<extra_0>"
        self.bot_token = "<extra_1>"
        self.end_token = "<extra_2>"

        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")

        print(f"[BaselineRMDataset] Loading data from {data_path}...")
        skipped_not_two_responses = 0
        skipped_len_mismatch = 0
        skipped_same_ranks = 0
        skipped_no_rank = 0

        with open(data_path, "r") as f:
            for line in f:
                try:
                    j = json.loads(line)
                except json.JSONDecodeError:
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
        print(f"[BaselineRMDataset] Loaded {len(self.samples)} valid samples.")
        if skipped_not_two_responses > 0:
            print(f"[BaselineRMDataset] Skipped {skipped_not_two_responses} samples with != 2 responses.")
        if skipped_len_mismatch > 0:
            print(f"[BaselineRMDataset] Skipped {skipped_len_mismatch} samples with rank/response length mismatch.")
        if skipped_same_ranks > 0:
            print(f"[BaselineRMDataset] Skipped {skipped_same_ranks} samples with same ranks (invalid for pairwise loss).")
        if skipped_no_rank > 0:
            print(f"[BaselineRMDataset] Skipped {skipped_no_rank} samples with no rank.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # 传入 idx 作为 sample_id，确保全局或局部唯一性足以区分样本
        sample = self.samples[idx]
        query = sample["query"]
        responses = sample["responses"]
        rank = sample["rank"]

        return {
            "query": query,
            "responses": responses,
            "rank": rank,
            "sample_id": idx  # 关键：传递样本ID
        }


def train_baseline_rm(baseline_rm):
    """
    训练Baseline Reward Model
    """
    out_dir = os.path.join(config.out_dir, "baseline")
    os.makedirs(out_dir, exist_ok=True)
    
    lr = config.learning_rate
    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path, trust_remote_code=True)

    # 使用独立的 train / test 文件
    train_data_path = os.path.join(config.rawdata_dir, "Ernie-rlhf-train.jsonl")
    val_data_path = os.path.join(config.rawdata_dir, "Ernie-rlhf-test.jsonl")
    train_dataset = BaselineRMDataset(train_data_path)
    val_dataset = BaselineRMDataset(val_data_path)

    print(f"📊 Dataset split: {len(train_dataset)} train, {len(val_dataset)} val (from explicit train/test files)")
    
    # 打印可训练参数
    param_info = baseline_rm.get_trainable_parameters()
    print(f"🔧 Trainable parameters: {param_info['trainable_params']:,} / {param_info['all_params']:,} ({param_info['trainable_percentage']:.2f}%)")
    
    # 训练参数（单GPU，不使用DeepSpeed）
    training_args = TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        learning_rate=lr,
        num_train_epochs=config.num_epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_dir=os.path.join(out_dir, "logs"),
        logging_steps=50,
        bf16=True,
        report_to="none",
        eval_accumulation_steps=1,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        dataloader_num_workers=0,
        save_total_limit=2,
        dataloader_drop_last=True,
        remove_unused_columns=False,  # 关键：防止Trainer删除query/chosen/rejected字段
        save_safetensors=False,       # 解决共享权重导致的RuntimeError
    )
    
    def collate_fn(batch):
        """
        将 batch 中的 rank 和 sample_id 拼接到 labels 中。
        Labels Shape: [Total_Responses, 2] -> (Rank, SampleID)
        （按照 train_module_prefix.py 的逻辑）
        """
        all_responses_input_ids = []
        all_responses_attention_mask = []
        all_labels = []
        
        user_token = "<extra_0>"
        bot_token = "<extra_1>"
        end_token = "<extra_2>"
        
        for item in batch:
            num_responses = len(item["responses"])
            sample_id = item["sample_id"]
            query = item["query"]
            
            # Tokenize每个response
            for resp in item["responses"]:
                text = f"{user_token}{query}{bot_token}{resp}{end_token}"
                enc = tokenizer(
                    text,
                    max_length=512,
                    padding=False,
                    truncation=True,
                    return_tensors="pt"
                )
                all_responses_input_ids.append(enc["input_ids"].squeeze(0))
                all_responses_attention_mask.append(enc["attention_mask"].squeeze(0))
            
            # 构造 Rank
            ranks = torch.tensor(item["rank"], dtype=torch.long)
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
    
    def compute_metrics(eval_pred):
        """计算pairwise准确率（按照 train_module_prefix.py 的逻辑）"""
        import numpy as np
        import pandas as pd
        
        predictions = eval_pred.predictions
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        predictions = np.array(predictions).reshape(-1)
        
        labels_full = np.array(eval_pred.label_ids)
        
        # 从collate_fn返回的格式中提取ranks和sample_ids
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
    
    # 创建自定义Trainer（按照 train_module_prefix.py 的逻辑）
    class BaselineRMTrainer(Trainer):
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
    
    trainer = BaselineRMTrainer(
        model=baseline_rm,
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

    print("🚀 Starting Baseline RM training...")
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
    
    print("✅ Baseline RM training complete!")
