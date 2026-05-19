from typing import Any

import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from ..types import SamplerBase, SamplerResponse, MessageList

DEFAULT_LLAMA_SYSTEM_MESSAGE = "You are a helpful assistant."

class LocalLlamaSampler(SamplerBase):
    def __init__(self, model_path, device="cuda", system_message: str | None = None,):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).to(device)
        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=device,
        )
        self.system_message = system_message 

    def _pack_message(self, role: str, content: Any):
        return {"role": str(role), "content": content}

    def _build_prompt(self, message_list: MessageList):
        # 只允许 system/user/assistant 三种角色
        allowed_roles = {"system", "user", "assistant"}
        for msg in message_list:
            if msg["role"] not in allowed_roles:
                raise ValueError(f"Llama sampler only supports system, user, assistant roles, got {msg['role']}")

        # 自动补充 system message（如果没有）
        if not message_list or message_list[0]["role"] != "system":
            message_list = [{"role": "system", "content": self.system_message}] + message_list

        return self.tokenizer.apply_chat_template(
            message_list, tokenize=False, add_generation_prompt=True
        )

    def __call__(self, message_list: MessageList) -> SamplerResponse:
        prompt = self._build_prompt(message_list)
        outputs = self.pipeline(
            prompt,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )
        response_text = outputs[0]["generated_text"][len(prompt):]
        return SamplerResponse(
            response_text=response_text,
            response_metadata={},
            actual_queried_message_list=message_list,
        )
