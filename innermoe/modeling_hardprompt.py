from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import torch.nn as nn
import torch
import torch.nn.functional as F
from innermoe.modeling_prefix import Router


class HardPromptRM(nn.Module):
    """
    Hard Prompt Reward Model - 对比实验
    先用 Router 对输入做分类（argmax 选类别），再拼接对应硬 prompt，训练 backbone + value head。
    """
    def __init__(self, config, hard_prompts=None, router_ckpt_path=None):
        super().__init__()
        
        self.config_obj = AutoConfig.from_pretrained(config.model_name_or_path, trust_remote_code=True)
        
        # Backbone（可训练）
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name_or_path,
            config=self.config_obj,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        
        # Value head（可训练）
        self.value_head = nn.Linear(self.config_obj.hidden_size, 1, dtype=torch.bfloat16)
        self.prelu = nn.PReLU()

        # Backbone 全量训练
        # for param in self.model.parameters():
        #     param.requires_grad = False
        
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path, trust_remote_code=True)
        
        # 硬 prompt 文本（固定，不训练）
        if hard_prompts is None:
            hard_prompts = {
                "L1_basic": "You are a medical knowledge assessment expert. Please rate the answers to basic medical knowledge based on accuracy, completeness, conceptual understanding, and memory accuracy. Focus on factual accuracy and conceptual clarity.",
                "L2_application": "You are a clinical knowledge application assessment expert. Please rate clinical case analyses based on diagnostic accuracy, information integration ability, depth of analysis, clinical relevance, and logic. Focus on the ability to apply knowledge to actual cases.",
                "L3_reasoning": "You are an advanced clinical reasoning assessment expert. Please rate complex clinical reasoning based on reasoning depth, rationality of treatment plans, handling of uncertainty, multi-step planning, decision-making ability, and innovation. Focus on multi-step reasoning and treatment planning capabilities."
            }
        
        self.hard_prompts = hard_prompts
        self.cat_list = list(hard_prompts.keys()) if isinstance(hard_prompts, dict) else config.cat_list
        
        # 预计算 prompt embeddings（固定，不训练）
        prefix_embeds_list = []
        prefix_lens = []
        with torch.no_grad():
            for cat in self.cat_list:
                prompt_text = hard_prompts.get(cat, hard_prompts.get(list(hard_prompts.keys())[0]))
                tok = self.tokenizer(prompt_text, return_tensors="pt", truncation=True)
                embeds = self.model.get_input_embeddings()(tok["input_ids"]).squeeze(0)  # [L, D]
                prefix_embeds_list.append(embeds)
                prefix_lens.append(embeds.size(0))
        
        self.max_prefix_len = max(prefix_lens) if prefix_lens else 0
        
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
        
        self.register_buffer("prefix_embeddings", torch.stack(padded_prefixes, dim=0))  # [C, max_L, D]
        self.register_buffer("prefix_mask", torch.stack(prefix_masks, dim=0))          # [C, max_L]

        # Router（仅推理，hard argmax 选择类别）
        self.router = Router(ckpt_path=router_ckpt_path)
        self.router.to(dtype=torch.bfloat16)
        for param in self.router.parameters():
            param.requires_grad = False
        
        print("🔧 [HardPromptRM] Initialized:")
        print(f"   - Backbone: {config.model_name_or_path} (FROZEN)")
        print(f"   - Value head: Linear({self.config_obj.hidden_size}, 1) (TRAINABLE)")
        print(f"   - Hard prompts: {len(self.hard_prompts)} prompts (FIXED, NOT TRAINABLE)")
        print(f"   - Router: argmax selection (FROZEN)")
    
    def forward(self, input_ids, attention_mask, labels=None, category_ids=None, **kwargs):
        batch_size = input_ids.size(0)
        device = input_ids.device
        
        # Router 选择类别（如未提供 category_ids）
        if category_ids is None:
            with torch.no_grad():
                router_out = self.router(input_ids=input_ids, attention_mask=attention_mask)
                category_ids = torch.argmax(router_out["logits"], dim=-1)  # [N]
        
        # Token embeddings
        token_embeds = self.model.get_input_embeddings()(input_ids)  # [N, L, H]
        
        # 选择硬 prompt（hard）
        selected_prefixes = self.prefix_embeddings[category_ids]  # [N, max_L, H]
        selected_masks = self.prefix_mask[category_ids].to(dtype=attention_mask.dtype)  # [N, max_L]
        
        # 拼接
        inputs_embeds = torch.cat([selected_prefixes, token_embeds], dim=1)  # [N, P+L, H]
        attention_mask_with_prefix = torch.cat([selected_masks, attention_mask], dim=1)  # [N, P+L]
        
        # Backbone forward
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask_with_prefix,
            output_hidden_states=True,
            return_dict=True
        )
        
        hidden_states = outputs.hidden_states[-1]  # [N, P+L, H]
        seq_lengths = attention_mask_with_prefix.sum(dim=1) - 1
        last_hidden_states = hidden_states[torch.arange(batch_size, device=device), seq_lengths]
        
        h = self.prelu(last_hidden_states)
        logits = self.value_head(h).squeeze(-1)  # [N]
        
        loss = None
        if labels is not None:
            if labels.dim() == 2 and labels.size(1) == 2:
                ranks = labels[:, 0]
                sample_ids = labels[:, 1]
                loss = self.compute_vectorized_loss(logits, ranks, sample_ids)
            elif labels.dim() == 1:
                ranks = labels
                sample_ids = torch.zeros_like(ranks)
                loss = self.compute_vectorized_loss(logits, ranks, sample_ids)
        
        return {"loss": loss, "logits": logits}
    
    def compute_vectorized_loss(self, rewards, ranks, sample_ids):
        id_mask = sample_ids.unsqueeze(1) == sample_ids.unsqueeze(0)  # [N, N]
        id_mask.fill_diagonal_(False)
        rank_diff = ranks.unsqueeze(1) - ranks.unsqueeze(0)
        valid_pair_mask = id_mask & (rank_diff < 0)
        if not valid_pair_mask.any():
            return rewards.sum() * 0.0
        r_diff = rewards.unsqueeze(1) - rewards.unsqueeze(0)
        valid_diffs = r_diff[valid_pair_mask]
        loss = -F.logsigmoid(valid_diffs).mean()
        return loss
    
    def get_trainable_parameters(self):
        trainable_params = 0
        all_params = 0
        for _, param in self.named_parameters():
            all_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        return {
            "trainable_params": trainable_params,
            "all_params": all_params,
            "trainable_percentage": 100 * trainable_params / all_params if all_params > 0 else 0.0
        }
