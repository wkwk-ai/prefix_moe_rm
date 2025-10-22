import torch.nn as nn
from transformers import AutoTokenizer
import os
import json
from tqdm import tqdm
import base_config as config
from router.modeling import Router
from router.train import train_router
from prepare_data.main import prepare_data
from innermoe.modeling_prefix import InnerMoERM
from innermoe.train_pipe_prefix import train_pipe


class DMoERM(nn.Module):

    def __init__(self):
        super(DMoERM, self).__init__()

    def train(self):

        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path, trust_remote_code=True)

        # 初始化 Router
        self.router = Router()
        self.router.to(config.device)
        # 训练 Router
        print('-' * 15 + 'begin training router' + '-' * 15)
        train_router(self.router)

        # 初始化并训练唯一一个 InnerMoERM
        print('-' * 15 + 'begin training unified InnerMoERM' + '-' * 15)
        self.innerMoERM = InnerMoERM().to(config.device)
        train_pipe(self.innerMoERM, router=self.router, tokenizer=self.tokenizer)  
        # 这里的 train_pipe 需要稍微修改，让它在训练时用 router.probabilities(context) 的输出作为额外输入

        print('-' * 15 + 'finish training' + '-' * 15)

    def test(self):

        print('-' * 15 + 'begin testing' + '-' * 15)

        with open(os.path.join(config.data_dir, 'Ernie-rlhf-test.jsonl'), 'r') as f:
            lines = [json.loads(line) for line in f.readlines()]
        
        results = {}
        results['all'] = {'acc': 0, 'tot': 0}

        # test the consistency with human annotation
        for line in tqdm(lines[:20]):

            if line['label'] not in results.keys():
                results[line['label']] = {'acc': 0, 'tot': 0}
            
            rewards = []
            for response in line['response']:
                context = {'src': line['src'], 'tgt': line['tgt'] + [response]}
                reward = self.forward(context)
                rewards.append(reward)

            for i, rank1 in enumerate(line['rank']):
                for j, rank2 in enumerate(line['rank'][i + 1:]):
                    if rank1 > rank2 and rewards[i] > rewards[j] \
                        or rank1 < rank2 and rewards[i] < rewards[j]:
                        results[line['label']]['acc'] += 1
                        results['all']['acc'] += 1
                    if rank1 != rank2:
                        results[line['label']]['tot'] += 1
                        results['all']['tot'] += 1

        for key in results.keys():
            results[key]['acc_rate'] = results[key]['acc'] / results[key]['tot']
        
        print('-' * 15 + 'finish testing' + '-' * 15)
        print(results)

    def forward(self, context):

        # router 给出各类别的概率分布
        cat_probs = self.router.get_probabilities(context, self.tokenizer)  
        # shape: [num_categories]

        # InnerMoERM 接收输入时额外带上 cat_probs
        final_reward = self.innerMoERM.get_reward(context, self.tokenizer, cat_probs=cat_probs)
        
        return final_reward


if __name__ == '__main__':
    model = DMoERM()
    model.train()
    model.test()
