import torch.nn as nn
import torch
from typing import Optional
import innermoe.config as config

prefix_prompts = {
    "roleplay": "你是一名角色扮演评估专家，请根据人设和情感带入、对话感、共情能力、关系特点体现、个性化特征体现、内容丰富性对回答进行评分。",
    "chat": "你是一名闲聊评估专家，请根据对话感、主动性、情感表达、共情能力、内容丰富性对回答进行评分。",
    "subj_qa": "你是一名主观知识问答评估专家，请根据说服力、逻辑性、观点丰富度、知识面广度、问题针对性对回答进行评分。",
    "obj_qa": "你是一名客观知识问答评估专家，请根据正确性、客观程度、推理能力、逻辑性、知识面深度、问题针对性对回答进行评分。",
    "text": "你是一名文本创作评估专家，请根据意图符合程度、表达能力、可读性、内容丰富性、逻辑性对回答进行评分。"
}


from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import torch.nn as nn
import torch
import innermoe.config as config


class InnerMoERM(nn.Module):
    
    def __init__(self, categories):
        super().__init__()
        self.base_config = AutoConfig.from_pretrained(config.model_name_or_path)
        self.base_model = AutoModelForCausalLM.from_pretrained(self.base_config)
        self.value_head = nn.Linear(self.base_model.config.hidden_size, 1)

        self.prelu = nn.PReLU()
        self.sigmoid = nn.Sigmoid()
        self.device = config.device

        # 初始化 prefix embedding
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path, trust_remote_code=True)
        prefix_list = []
        for cat in categories:
            prefix_text = prefix_prompts.get(cat, f"请你对{cat}任务的回答进行合理的评分。")
            tok = self.tokenizer(prefix_text, return_tensors="pt").to(self.device)
            with torch.no_grad():
                embeds = self.base_model.embed_tokens(tok["input_ids"])  # [1, L, hidden]
                prefix_vec = embeds.mean(dim=1)  # 平均池化
            prefix_list.append(prefix_vec.squeeze(0))

        prefix_init = torch.stack(prefix_list, dim=0)  # [num_categories, hidden]
        self.prefix_embeddings = nn.Parameter(prefix_init)  # 可训练参数

    def forward(self,
                input_ids: torch.LongTensor, 
                attention_mask: torch.Tensor,
                cat_probs: torch.Tensor) -> torch.Tensor:
        """
        Forward with prefix-MoE:
        - cat_probs: [batch, num_categories]
        """
        # weighted prefix
        weighted_prefix = torch.matmul(cat_probs, self.prefix_embeddings)  # [batch, hidden]
        weighted_prefix = weighted_prefix.unsqueeze(1)  # [batch, 1, hidden]

        # embed tokens
        inputs_embeds = self.base_model.embed_tokens(input_ids)

        # prepend prefix
        inputs_embeds = torch.cat([weighted_prefix, inputs_embeds], dim=1)

        # adjust attention mask
        prefix_mask = torch.ones((attention_mask.size(0), 1), dtype=attention_mask.dtype, device=attention_mask.device)
        attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)

        # base model
        outputs = self.base_model(inputs_embeds=inputs_embeds,
                                  attention_mask=attention_mask,
                                  output_hidden_states=True)
        last_hidden_states = outputs.hidden_states[-1]

        # get final hidden
        last_index = attention_mask.cumsum(dim=1).argmax(dim=1)
        final_hidden = last_hidden_states.gather(
            1, last_index.view(-1, 1, 1).expand(-1, 1, last_hidden_states.size(-1))
        ).squeeze(1)

        # reward prediction
        value = self.value_head(final_hidden).squeeze(-1)
        return value

    def get_reward(self, context, tokenizer, cat_probs: torch.Tensor):
        self.eval()
        with torch.no_grad():
            user = '<extra_0>'
            bot = '<extra_1>'
            end = '<extra_2>'
            prompt = f"{user}{context['src'][-1]}{bot}{context['tgt'][-1]}{end}"
            tok = tokenizer(prompt, return_tensors="pt").to(config.device)
            reward = self.forward(
                tok['input_ids'],
                tok['attention_mask'],
                cat_probs=cat_probs.unsqueeze(0).to(config.device)
            )
        return reward.item()

