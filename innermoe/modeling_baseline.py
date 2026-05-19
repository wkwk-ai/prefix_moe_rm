from transformers import AutoConfig, AutoModelForCausalLM
import torch.nn as nn
import torch
import torch.nn.functional as F

class BaselineRM(nn.Module):
    """
    Baseline Reward Model - 不使用Router和Prefix
    只有一个共享的backbone + 单一的value head
    用于对比实验，验证Router和Prefix的作用
    """
    def __init__(self, config):
        super().__init__()
        
        self.config_obj = AutoConfig.from_pretrained(config.model_name_or_path, trust_remote_code=True)
        
        # 语言模型backbone（冻结）
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name_or_path,
            config=self.config_obj,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        
        # Backbone 全量训练
        # for param in self.model.parameters():
        #     param.requires_grad = False
        
        # 单一的reward head（可训练）
        self.value_head = nn.Linear(self.config_obj.hidden_size, 1, dtype=torch.bfloat16)
        self.prelu = nn.PReLU()
        
        print("🔧 [BaselineRM] Initialized:")
        print(f"   - Backbone: {config.model_name_or_path} (FROZEN)")
        print(f"   - Value head: Linear({self.config_obj.hidden_size}, 1) (TRAINABLE)")
        print(f"   - No Router, No Prefix embeddings")

    def forward(self, input_ids, attention_mask, labels=None, **kwargs):
        """
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            labels: [batch_size, 2] -> (Rank, SampleID) 或 [batch_size] -> Rank
        
        Returns:
            dict with keys:
                - logits: [batch_size] - reward分数
                - loss: scalar (如果提供labels)
        （按照 train_module_prefix.py 的逻辑）
        """
        batch_size = input_ids.size(0)
        device = input_ids.device
        
        # 通过backbone获取隐藏状态
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        
        # 获取最后一层隐藏状态
        hidden_states = outputs.hidden_states[-1]  # [batch_size, seq_len, hidden_size]
        
        # 获取每个序列最后一个有效token的隐藏状态
        seq_lengths = (attention_mask.sum(dim=1) - 1).long()  # [batch_size]
        last_hidden_states = hidden_states[torch.arange(batch_size, device=device), seq_lengths]
        
        # 通过value head得到reward
        logits = self.value_head(self.prelu(last_hidden_states))  # [batch_size, 1]
        logits = logits.squeeze(-1)  # [batch_size]
        
        # --- Loss Calculation (按照 train_module_prefix.py 的逻辑) ---
        loss = None
        if labels is not None:
            # 判断 labels 格式
            if labels.dim() == 2 and labels.size(1) == 2:
                # 格式: [Rank, SampleID] (由 Collate Function 生成)
                ranks = labels[:, 0]
                sample_ids = labels[:, 1]
                loss = self.compute_vectorized_loss(logits, ranks, sample_ids)
            
            elif labels.dim() == 1:
                # 格式: [Rank] 
                ranks = labels
                sample_ids = torch.zeros_like(ranks)
                loss = self.compute_vectorized_loss(logits, ranks, sample_ids)
        
        return {"loss": loss, "logits": logits}

    def compute_vectorized_loss(self, rewards, ranks, sample_ids):
        """
        计算pairwise ranking loss（向量化实现，与 modeling_prefix.py 保持一致）
        
        Args:
            rewards: [N] - 每个response的reward分数
            ranks:   [N] - 每个response的rank（数值越小越好）
            sample_ids: [N] - 每个response所属的样本ID（用于区分不同prompt）
        """
        # 1. 广播构造比较矩阵 [N, N]
        id_mask = sample_ids.unsqueeze(1) == sample_ids.unsqueeze(0) # [N, N]
        id_mask.fill_diagonal_(False)
        
        # 2. 确定优劣关系
        rank_diff = ranks.unsqueeze(1) - ranks.unsqueeze(0) # [N, N]
        valid_pair_mask = id_mask & (rank_diff < 0)
        
        if not valid_pair_mask.any():
            return rewards.sum() * 0.0
        
        # 3. 计算 Reward 差异
        r_diff = rewards.unsqueeze(1) - rewards.unsqueeze(0) # r[i] - r[j]
        valid_diffs = r_diff[valid_pair_mask]
        
        # LogSigmoid Loss
        loss = -F.logsigmoid(valid_diffs).mean()
        
        return loss

    def gradient_checkpointing_enable(self):
        """启用梯度检查点（如果需要节省显存）"""
        self.model.gradient_checkpointing_enable()

    def get_trainable_parameters(self):
        """获取可训练参数的统计信息"""
        trainable_params = 0
        all_params = 0
        for name, param in self.named_parameters():
            all_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        
        return {
            "trainable_params": trainable_params,
            "all_params": all_params,
            "trainable_percentage": 100 * trainable_params / all_params
        }
