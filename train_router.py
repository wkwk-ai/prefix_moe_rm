# train_router.py
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
    AutoConfig,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
)

# ---------------------------
# Router 模型（简单封装）
# ---------------------------
class RouterForClassification(nn.Module):
    def __init__(self, model_name_or_path, cat_list):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.cat_list = cat_list

        base_config = AutoConfig.from_pretrained(model_name_or_path)
        # 使用 LM backbone 作为 encoder，提取 hidden state
        self.backbone = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            config=base_config,
            torch_dtype=torch.bfloat16,
        )

        hidden = base_config.hidden_size
        self.value_head = nn.Linear(hidden, len(cat_list))
        # 确保 value_head 使用 bfloat16
        self.value_head = self.value_head.to(dtype=torch.bfloat16)

    def forward(self, input_ids, attention_mask=None, labels=None):
        # backbone 返回含 hidden_states（我们只需要最后一层）
        outputs = self.backbone(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        last_hidden = outputs.hidden_states[-1]  # [B, T, H]

        # 取最后一个非 pad token 的 hidden（按 attention_mask）
        last_idx = attention_mask.sum(dim=1) - 1
        pooled = last_hidden[torch.arange(last_hidden.size(0)), last_idx]  # [B, H]

        logits = self.value_head(pooled)  # [B, C]

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)

        return {"loss": loss, "logits": logits}

    # 保存/加载（自定义格式，Trainer 会调用）
    def save_pretrained(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(save_dir, "router.bin"))
        json.dump(
            {"model_name_or_path": self.model_name_or_path, "cat_list": self.cat_list},
            open(os.path.join(save_dir, "router_config.json"), "w"),
            ensure_ascii=False,
            indent=2,
        )

    @classmethod
    def from_pretrained(cls, save_dir):
        cfg = json.load(open(os.path.join(save_dir, "router_config.json")))
        model = cls(cfg["model_name_or_path"], cfg["cat_list"])
        sd = torch.load(os.path.join(save_dir, "router.bin"), map_location="cpu")
        model.load_state_dict(sd)
        return model


# ---------------------------
# Dataset：基于你的 MoERMDataset，但每个 response 单独作为样本
# ---------------------------
class RouterDataset(Dataset):
    def __init__(self, config, tokenizer):
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

                # ========== 展开 response ==========
                query = j["src"][-1]

                for resp in responses:
                    text = (
                        f"{self.user_token}{query}"
                        f"{self.bot_token}{resp}"
                        f"{self.end_token}"
                    )

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
        print(f"[RouterDataset] Loaded {len(self.samples)} samples")
        print("Skipped:", skipped)

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
        print(f"[RouterTrainer] Custom save to {output_dir}")
        # model 必须实现 save_pretrained
        self.model.save_pretrained(output_dir)


# ---------------------------
# train entry
# ---------------------------
def train_router(config):
    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path, trust_remote_code=True)

    dataset = RouterDataset(config, tokenizer)

    # simple split
    train_size = int(len(dataset) * 0.9)
    eval_size = len(dataset) - train_size
    train_dataset, eval_dataset = torch.utils.data.random_split(dataset, [train_size, eval_size])

    model = RouterForClassification(config.model_name_or_path, config.cat_list)
    
    # DeepSpeed 会自动管理设备，不需要手动 .cuda()
    # model = model.cuda()

    training_args = TrainingArguments(
        output_dir=config.out_dir,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        learning_rate=config.lr,
        num_train_epochs=config.epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_accuracy",
        greater_is_better=True,
        bf16=True,
        logging_steps=50,
        report_to="none",
        deepspeed="./ds_config_router.json",  # 启用 DeepSpeed
        dataloader_num_workers=0,
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
    print("Saved router to:", config.out_dir)


# ---------------------------
# Config
# ---------------------------
class Config:
    model_name_or_path = "/mnt/data/model/Qwen2.5-1.5B-Instruct"
    rawdata_dir = "./prepare_data/Ernie-rlhf"
    out_dir = "./results/router"
    cat_list = ['L1_basic', 'L2_application', 'L3_reasoning']
    batch_size = 8
    lr = 1e-5
    epochs = 3

if __name__ == "__main__":
    os.makedirs(Config.out_dir, exist_ok=True)
    train_router(Config())
