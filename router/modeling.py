import os
import json
import torch.nn as nn
import torch
from typing import Optional, Dict
import router.config as config
from transformers import AutoConfig, AutoModelForCausalLM


class Router(nn.Module):
    """
    Router model for classification tasks.
    Uses a language model backbone with a classification head.
    """
    def __init__(self, model_name_or_path=None, cat_list=None, ckpt_path=None):
        super().__init__()
        self.model_name_or_path = model_name_or_path or config.model_name_or_path
        self.cat_list = cat_list or config.cat_list
        
        base_config = AutoConfig.from_pretrained(self.model_name_or_path)
        # Use LM backbone as encoder to extract hidden states
        self.backbone = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            config=base_config,
            torch_dtype=torch.bfloat16,
        )
        
        hidden = base_config.hidden_size
        self.value_head = nn.Linear(hidden, len(self.cat_list), dtype=torch.bfloat16)
        
        # Load weights if checkpoint path is provided
        if ckpt_path is not None:
            state_dict = torch.load(f"{ckpt_path}/router.bin", map_location="cpu")
            self.load_state_dict(state_dict, strict=True)

    def forward(self, input_ids, attention_mask=None, labels=None):
        """
        Forward pass through the router model.
        
        Args:
            input_ids: Tokenized input [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            labels: Optional labels for training [batch_size]
        
        Returns:
            dict with "loss" and "logits" keys
        """
        # Backbone returns hidden_states (we only need the last layer)
        outputs = self.backbone(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        
        last_hidden = outputs.hidden_states[-1]  # [B, T, H]
        
        # Get the last non-pad token's hidden state (according to attention_mask)
        last_idx = attention_mask.sum(dim=1) - 1
        pooled = last_hidden[torch.arange(last_hidden.size(0)), last_idx]  # [B, H]
        
        logits = self.value_head(pooled)  # [B, C]
        
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
        
        return {"loss": loss, "logits": logits}
    
    def get_probabilities(self, context, tokenizer):
        """
        Get probability distribution over categories for a given context.
        
        Args:
            context: dict with 'src' key containing list of queries
            tokenizer: Tokenizer to encode the context
        
        Returns:
            torch.Tensor: Probability distribution [num_categories]
        """
        user_token = "<extra_0>"
        bot_token = "<extra_1>"
        end_token = "<extra_2>"
        
        self.eval()
        with torch.no_grad():
            # Construct input text from context
            query = context['src'][-1] if isinstance(context['src'], list) else context['src']
            text = f"{user_token}{query}{end_token}"
            
            tokenized = tokenizer(
                text,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            
            # Move to same device as model
            device = next(self.parameters()).device
            input_ids = tokenized["input_ids"].to(device)
            attention_mask = tokenized["attention_mask"].to(device)
            
            output = self(input_ids=input_ids, attention_mask=attention_mask)
            logits = output["logits"]  # [1, num_categories]
            probs = torch.softmax(logits, dim=-1).squeeze(0)  # [num_categories]
        
        return probs
    
    def route(self, context, tokenizer):
        """
        Route to a single category (returns category index).
        
        Args:
            context: dict with 'src' key containing list of queries
            tokenizer: Tokenizer to encode the context
        
        Returns:
            int: Predicted category index
        """
        self.eval()
        with torch.no_grad():
            probs = self.get_probabilities(context, tokenizer)
            predicted = torch.argmax(probs, dim=-1)
        
        return predicted.item()
    
    def save_pretrained(self, save_dir):
        """
        Save the router model to disk.
        
        Args:
            save_dir: Directory to save the model
        """
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
        """
        Load the router model from disk.
        
        Args:
            save_dir: Directory containing the saved model
        
        Returns:
            Router: Loaded router model
        """
        cfg = json.load(open(os.path.join(save_dir, "router_config.json")))
        model = cls(cfg["model_name_or_path"], cfg["cat_list"])
        sd = torch.load(os.path.join(save_dir, "router.bin"), map_location="cpu")
        model.load_state_dict(sd)
        return model
