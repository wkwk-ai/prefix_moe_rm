import os
# 强制使用单卡，避免DataParallel在多卡环境下的peer mapping错误
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.nn as nn
from transformers import AutoTokenizer
import json
from tqdm import tqdm
import base_config as config
from innermoe.modeling_baseline import BaselineRM
from innermoe.train_module_baseline import train_baseline_rm


class BaselineRewardModel(nn.Module):
    """
    Baseline Reward Model - 用于对比实验
    不使用Router和Prefix，只有单一的reward head
    """
    def __init__(self):
        super(BaselineRewardModel, self).__init__()

    def train(self):
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path, trust_remote_code=True)
        
        print('-' * 15 + 'begin training Baseline RM' + '-' * 15)
        self.baseline_rm = BaselineRM(config)
        
        train_baseline_rm(baseline_rm=self.baseline_rm)
        
        print('-' * 15 + 'finish training' + '-' * 15)

    def test(self):
        print('-' * 15 + 'begin testing Baseline RM' + '-' * 15)
        
        # 加载测试数据
        with open(os.path.join(config.rawdata_dir, 'Ernie-rlhf-test.jsonl'), 'r') as f:
            lines = [json.loads(line) for line in f.readlines()]
        
        results = {}
        results['all'] = {'acc': 0, 'tot': 0}
        
        print(f"Testing on {len(lines)} samples...")
        for line in tqdm(lines):
            if line['label'] not in results.keys():
                results[line['label']] = {'acc': 0, 'tot': 0}
            
            # 计算每个response的reward
            rewards = []
            for response in line['response']:
                context = {'src': line['src'], 'tgt': line['tgt'] + [response]}
                reward = self.forward(context)
                rewards.append(reward)
            
            # Pairwise比较
            for i in range(len(line['rank'])):
                for j in range(i + 1, len(line['rank'])):
                    rank_i = line['rank'][i]
                    rank_j = line['rank'][j]
                    
                    # 跳过相同rank的pair
                    if rank_i == rank_j:
                        continue
                    
                    # rank值越小越好（1比2好）
                    # 如果rank_i < rank_j，那么reward_i应该 > reward_j
                    correct = (rank_i < rank_j and rewards[i] > rewards[j]) or \
                              (rank_i > rank_j and rewards[i] < rewards[j])
                    
                    if correct:
                        results[line['label']]['acc'] += 1
                        results['all']['acc'] += 1
                    
                    results[line['label']]['tot'] += 1
                    results['all']['tot'] += 1
        
        # 计算准确率
        for key in results.keys():
            results[key]['acc_rate'] = results[key]['acc'] / results[key]['tot'] if results[key]['tot'] > 0 else 0.0
        
        print('-' * 15 + 'finish testing' + '-' * 15)
        print("📊 Baseline RM Test Results:")
        print(json.dumps(results, indent=2))
        
        # 保存结果
        result_file = os.path.join(config.out_dir, "baseline", "test_results.json")
        with open(result_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"✅ Results saved to {result_file}")

    def forward(self, context):
        """
        context: {'src': [queries], 'tgt': [history + response]}
        """
        # 构造输入文本
        user_token = "<extra_0>"
        bot_token = "<extra_1>"
        end_token = "<extra_2>"
        
        # 构造query部分
        # 只使用最后一个query，与训练保持一致
        query = context['src'][-1] if len(context['src']) > 0 else ""
        src = f"{user_token}{query}"
        
        # 构造response部分
        response = context['tgt'][-1] if 'tgt' in context else ""
        text = f"{src}{bot_token}{response}{end_token}"
        
        # Tokenize
        enc = self.tokenizer(
            text,
            max_length=512,
            truncation=True,
            return_tensors="pt"
        )
        
        # 移动到GPU
        device = next(self.baseline_rm.parameters()).device
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        
        # 计算reward
        self.baseline_rm.eval()
        with torch.no_grad():
            outputs = self.baseline_rm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=None
            )
            reward = outputs["logits"]  # scalar
        
        return reward.item()


if __name__ == '__main__':
    model = BaselineRewardModel()
    model.train()
    model.test()
