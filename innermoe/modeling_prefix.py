import os

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import torch.nn as nn
import torch
import torch.nn.functional as F
import router.config as router_config


# --------------------------------------------------------
# Router: 路由模块 (保持不变，确保 dtype 兼容)
# --------------------------------------------------------
class Router(nn.Module):
    def __init__(self, ckpt_path=None):
        super().__init__()

        # 你的基础模型路径
        base_model_name = "/mnt/data/model/Qwen2.5-1.5B-Instruct"
        cat_list = router_config.cat_list

        base_config = AutoConfig.from_pretrained(base_model_name)

        self.backbone = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            config=base_config,
            torch_dtype=torch.bfloat16,
        )

        hidden = base_config.hidden_size
        self.value_head = nn.Linear(hidden, len(cat_list), dtype=torch.bfloat16)

        if ckpt_path is not None:
            print(f"🔄 Loading Router weights from: {ckpt_path}/router.bin")
            state_dict = torch.load(f"{ckpt_path}/router.bin", map_location="cpu")
            self.load_state_dict(state_dict, strict=True)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )

        hidden_states = outputs.hidden_states[-1]
        batch_size = input_ids.size(0)

        if attention_mask is None:
            last_indices = torch.tensor(
                [hidden_states.size(1) - 1] * batch_size,
                device=input_ids.device
            )
        else:
            last_indices = attention_mask.sum(dim=1) - 1

        last_hidden_states = hidden_states[torch.arange(batch_size), last_indices]
        logits = self.value_head(last_hidden_states)

        return {"logits": logits}


# --------------------------------------------------------
# InnerMoERM: 核心模型
# --------------------------------------------------------
class InnerMoERM(nn.Module):
    def __init__(self, config, prefix_prompts, router_ckpt_path=None, ckpt_path=None):
        super().__init__()

        self.config = AutoConfig.from_pretrained(config.model_name_or_path)

        # 1. 主模型 (Backbone)
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name_or_path,
            config=self.config,
            torch_dtype=torch.bfloat16,
        )

        # 2. Score Head (输出 scalar reward)
        self.value_head = nn.Linear(self.config.hidden_size, 1, dtype=torch.bfloat16)
        self.prelu = nn.PReLU().to(dtype=torch.bfloat16)

        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name_or_path,
            trust_remote_code=True
        )

        # 3. 构造 Prefix Embeddings
        prefix_embeds_list = []
        prefix_lens = []

        # 预计算 Prompt Embeddings
        for cat in config.cat_list:
            prefix_text = prefix_prompts.get(cat, f"请你对{cat}任务的回答进行合理的评分。")
            tok = self.tokenizer(prefix_text, return_tensors="pt", truncation=True)

            with torch.no_grad():
                embeds = self.model.get_input_embeddings()(tok["input_ids"]).squeeze(0)
                prefix_embeds_list.append(embeds)
                prefix_lens.append(embeds.size(0))

        self.max_prefix_len = max(prefix_lens)
        self.num_prefix = len(prefix_embeds_list)
        D = prefix_embeds_list[0].size(-1)

        # Padding & Stacking
        padded_prefixes = []
        prefix_masks = []

        for emb in prefix_embeds_list:
            L = emb.size(0)
            pad_len = self.max_prefix_len - L
            padded_emb = F.pad(emb, (0, 0, 0, pad_len), value=0.0)
            padded_prefixes.append(padded_emb)

            mask = torch.zeros(self.max_prefix_len)
            mask[:L] = 1.0
            prefix_masks.append(mask)

        # Parameters
        self.prefix_embeddings = nn.Parameter(torch.stack(padded_prefixes, dim=0))  # [C, max_L, D]

        # Register buffer for mask (non-trainable)
        self.register_buffer("prefix_mask", torch.stack(prefix_masks, dim=0))  # [C, max_L]

        # 4. Router
        self.router = Router(ckpt_path=router_ckpt_path)
        self.router.to(dtype=torch.bfloat16)

        for param in self.router.parameters():
            param.requires_grad = False

        # 5. Backbone 全量训练
        # for param in self.model.parameters():
        #     param.requires_grad = False

        # 6. Temperature scaling for router softmax (越小分布越尖锐)
        self.router_temperature = 0.1

        # 7. Diversity loss 系数 (防止 prefix embeddings 坍缩)
        self.diversity_loss_weight = 0.1

        # 8. Optional: load from existing RM checkpoint for fine-tuning
        if ckpt_path is not None:
            print(f"Loading InnerMoERM weights from: {ckpt_path}")
            ckpt_file = os.path.join(ckpt_path, "pytorch_model.bin") if os.path.isdir(ckpt_path) else ckpt_path
            state_dict = torch.load(ckpt_file, map_location="cpu")
            missing, unexpected = self.load_state_dict(state_dict, strict=False)
            if missing:
                print(f"  Missing keys: {missing}")
            if unexpected:
                print(f"  Unexpected keys: {unexpected}")
            print(f"  Checkpoint loaded successfully.")

    def get_router_probs(self, input_ids, attention_mask):
        out = self.router(input_ids=input_ids, attention_mask=attention_mask)
        logits = out["logits"]
        return torch.softmax(logits / self.router_temperature, dim=-1)

    # ======================================================
    # Prefix Diversity Loss: 惩罚 expert prefix 之间的余弦相似度
    # ======================================================
    def compute_diversity_loss(self):
        # prefix_embeddings: [C, max_L, D], 按有效长度 mean-pool 得到 [C, D]
        mask = self.prefix_mask.unsqueeze(-1)  # [C, max_L, 1]
        pooled = (self.prefix_embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)  # [C, D]

        # 归一化
        pooled_norm = F.normalize(pooled, dim=-1)  # [C, D]

        # 两两余弦相似度
        sim_matrix = torch.mm(pooled_norm, pooled_norm.t())  # [C, C]

        # 取上三角（排除对角线），求均值
        C = sim_matrix.size(0)
        mask_triu = torch.triu(torch.ones(C, C, device=sim_matrix.device), diagonal=1).bool()
        diversity_loss = sim_matrix[mask_triu].mean()

        return diversity_loss

    # ======================================================
    # 向量化 Pairwise Loss 计算 (高效版)
    # ======================================================
    def compute_vectorized_loss(self, rewards, ranks, sample_ids):
        id_mask = sample_ids.unsqueeze(1) == sample_ids.unsqueeze(0)
        id_mask.fill_diagonal_(False)

        rank_diff = ranks.unsqueeze(1) - ranks.unsqueeze(0)
        valid_pair_mask = id_mask & (rank_diff < 0)

        if not valid_pair_mask.any():
            return rewards.sum() * 0.0

        r_diff = rewards.unsqueeze(1) - rewards.unsqueeze(0)
        valid_diffs = r_diff[valid_pair_mask]

        loss = -F.logsigmoid(valid_diffs).mean()
        return loss

    # ======================================================
    # Forward
    # ======================================================
    def forward(self, input_ids, attention_mask, labels=None, sample_boundaries=None, **kwargs):
        device = input_ids.device
        num_responses = input_ids.size(0)

        # --- 1. Router ---
        with torch.no_grad():
            cat_probs = self.get_router_probs(input_ids, attention_mask)

        # --- 2. MoE Prefix Fusion ---
        token_embeds = self.model.get_input_embeddings()(input_ids)

        C, max_L, H = self.prefix_embeddings.shape
        expanded_prefix = self.prefix_embeddings.unsqueeze(0).expand(num_responses, -1, -1, -1)
        expanded_mask = self.prefix_mask.unsqueeze(0).expand(num_responses, -1, -1)

        weights = cat_probs.view(num_responses, C, 1, 1)
        fused_prefixes = torch.sum(weights * expanded_prefix, dim=1)

        fused_masks = torch.sum(weights.squeeze(-1) * expanded_mask, dim=1)
        fused_masks = (fused_masks > 0.1).to(dtype=attention_mask.dtype)

        inputs_embeds = torch.cat([fused_prefixes, token_embeds], dim=1)
        attention_mask_with_prefix = torch.cat([fused_masks, attention_mask], dim=1)

        # --- 3. Backbone ---
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask_with_prefix,
            output_hidden_states=True,
            return_dict=True
        )

        last_hidden = outputs.hidden_states[-1]
        seq_lengths = attention_mask_with_prefix.sum(dim=1) - 1
        final_hidden = last_hidden[torch.arange(num_responses, device=device), seq_lengths]

        # --- 4. Reward Head ---
        h = self.prelu(final_hidden)
        h = self.value_head(h)
        rewards = h.squeeze(-1)

        # --- 5. Loss ---
        loss = None
        if labels is not None:
            if labels.dim() == 2 and labels.size(1) == 2:
                ranks = labels[:, 0]
                sample_ids = labels[:, 1]
                loss = self.compute_vectorized_loss(rewards, ranks, sample_ids)

            elif labels.dim() == 1:
                ranks = labels
                sample_ids = torch.zeros_like(ranks)
                loss = self.compute_vectorized_loss(rewards, ranks, sample_ids)

            else:
                if sample_boundaries is not None:
                    pass

            # 加上 diversity loss，防止 prefix embeddings 坍缩
            if loss is not None:
                diversity_loss = self.compute_diversity_loss()
                loss = loss + self.diversity_loss_weight * diversity_loss

        return {"loss": loss, "logits": rewards}
