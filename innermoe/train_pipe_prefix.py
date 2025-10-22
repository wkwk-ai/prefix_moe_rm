import torch
import os
import argparse
import innermoe.config as config
from innermoe.modeling_prefix import InnerMoERM
from innermoe.load_data import load_all_data 
from innermoe.train_module_prefix import train_rm


def unified_train(innerMoERM, data, router, tokenizer):
    """
    Train the unified InnerMoERM in a given training phase
    """
    lr = config.learning_rate

    out_dir = os.path.join(config.out_dir, "unified")
    os.makedirs(os.path.join(out_dir, 'ckpts'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'pictures'), exist_ok=True)

    ckpt_path = os.path.join(out_dir, 'ckpts', '.pth')
    if os.path.exists(ckpt_path):
        innerMoERM.load_state_dict(torch.load(ckpt_path, map_location=config.device))
        return

    acc = train_rm(innerMoERM, data, lr, out_dir, router=router, tokenizer=tokenizer)

    innerMoERM.load_state_dict(torch.load(ckpt_path, map_location=config.device))
    innerMoERM = innerMoERM.to(config.device)
    print(f'finish training ! best acc: {acc}')


def train_pipe(innerMoERM, router, tokenizer):
    """
    Train the unified InnerMoERM in three training phases sequentially,
    using router probabilities instead of separate models per category.
    """
    out_dir = os.path.join(config.out_dir, "unified")
    os.makedirs(out_dir, exist_ok=True)

    data = load_all_data(config.phasedata_dir)

    # Phase 1
    unified_train(innerMoERM, data, router, tokenizer)



if __name__ == '__main__':
    """ The unified InnerMoERM can be trained and used independently """
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    from router.modeling import Router
    from transformers import AutoTokenizer
    import base_config as base_conf

    tokenizer = AutoTokenizer.from_pretrained(base_conf.model_name_or_path, trust_remote_code=True)
    router = Router().to(config.device)

    innerMoERM = InnerMoERM().to(config.device)
    train_pipe(innerMoERM, router, tokenizer)
