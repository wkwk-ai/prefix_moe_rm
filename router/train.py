"""
Training script for the Router model.
This script trains a router to classify queries into different categories.
"""
import os
import json
import random
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from router.modeling import Router
import router.config as config


# ---------------------------
# Dataset：基于你的 MoERMDataset，但每个 response 单独作为样本
# ---------------------------
class RouterDataset(Dataset):
    def __init__(self, tokenizer):
        """
        从 Ernie-rlhf-train.jsonl 中读取数据，
        对每条样本 j：
            - 保证 src, response, label 存在
            - responses >= 2 且 rank 长度匹配
            - label 在 cat_list 内
        然后把同一 query 下的每条 response 展开为独立的训练样本。

        展开后的样本格式：
            {
                "input_ids": Tensor[L]
                "attention_mask": Tensor[L]
                "labels": Tensor(1)  # label 对应 cat_list
            }
        """
        self.samples = []
        self.tokenizer = tokenizer

        self.user_token = "<extra_0>"
        self.bot_token = "<extra_1>"
        self.end_token = "<extra_2>"

        file_path = os.path.join(config.rawdata_dir, "Ernie-rlhf-train.jsonl")
        if not os.path.exists(file_path):
            raise FileNotFoundError(file_path)

        skipped = {
            "no_src_or_resp": 0,
            "no_label": 0,
            "label_invalid": 0,
            "few_responses": 0,
            "len_mismatch": 0,
        }

        with open(file_path, "r") as f:
            for i, line in enumerate(tqdm(f, desc="Loading router data")):

                j = json.loads(line)

                # ========== 基本检查 ==========
                if "src" not in j or "response" not in j:
                    skipped["no_src_or_resp"] += 1
                    continue

                if "label" not in j or j["label"] is None:
                    skipped["no_label"] += 1
                    continue

                label = j["label"]
                if label not in config.cat_list:
                    skipped["label_invalid"] += 1
                    continue
                label_id = config.cat_list.index(label)

                responses = j["response"]
                rank = j.get("rank", None)

                if not isinstance(responses, list) or len(responses) < 2:
                    skipped["few_responses"] += 1
                    continue

                if (not isinstance(rank, list)) or len(rank) != len(responses):
                    skipped["len_mismatch"] += 1
                    continue

                # ========== 每个 query 只生成一条样本（不按 response 展开）==========
                query = j["src"][-1]

                text = f"{self.user_token}{query}{self.end_token}"

                enc = tokenizer(
                    text,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                )

                self.samples.append({
                    "input_ids": enc["input_ids"].squeeze(0),
                    "attention_mask": enc["attention_mask"].squeeze(0),
                    "labels": torch.tensor(label_id, dtype=torch.long),
                })

        random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ---------------------------
# collate
# ---------------------------
def collate_fn(batch):
    """
    batch: list of dict
        {
            "input_ids": Tensor[L],
            "attention_mask": Tensor[L],
            "labels": Tensor(1)
        }
    """
    input_ids = [item["input_ids"] for item in batch]
    attention_mask = [item["attention_mask"] for item in batch]
    labels = [item["labels"] for item in batch]

    # padding
    input_ids = nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=0
    )
    attention_mask = nn.utils.rnn.pad_sequence(
        attention_mask, batch_first=True, padding_value=0
    )
    labels = torch.stack(labels)   # [B]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


# ---------------------------
# metrics
# ---------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, preds)}


# ---------------------------
# Trainer subclass: 禁用 safetensors 自动保存（避免 weight tying 问题）
# ---------------------------
class NoSafeTensorTrainer(Trainer):
    def _save(self, output_dir=None, state_dict=None):
        # model 必须实现 save_pretrained
        self.model.save_pretrained(output_dir)
        # 同时保存一份 pytorch_model.bin，让 Trainer 的 load_best_model_at_end 能找到
        import shutil
        router_bin = os.path.join(output_dir, "router.bin")
        pytorch_bin = os.path.join(output_dir, "pytorch_model.bin")
        if os.path.exists(router_bin) and not os.path.exists(pytorch_bin):
            shutil.copy2(router_bin, pytorch_bin)


# ---------------------------
# train entry
# ---------------------------
def train_router():
    """
    Train the router model.
    Uses configuration from router.config.
    """
    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path, trust_remote_code=True)

    dataset = RouterDataset(tokenizer)

    # simple split
    train_size = int(len(dataset) * 0.9)
    eval_size = len(dataset) - train_size
    train_dataset, eval_dataset = torch.utils.data.random_split(dataset, [train_size, eval_size])

    model = Router(config.model_name_or_path, config.cat_list)

    training_args = TrainingArguments(
        output_dir=config.out_dir,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        learning_rate=config.lr,
        num_train_epochs=config.num_epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_accuracy",
        greater_is_better=True,
        bf16=True,
        logging_steps=50,
        report_to="none",
        save_total_limit=2,
        save_safetensors=False,
    )

    trainer = NoSafeTensorTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # 保存最终 router
    model.save_pretrained(config.out_dir)


if __name__ == "__main__":
    os.makedirs(config.out_dir, exist_ok=True)
    train_router()
