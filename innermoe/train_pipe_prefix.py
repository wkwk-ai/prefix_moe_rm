import torch
import os
import argparse
import innermoe.config as config
from innermoe.modeling_prefix import InnerMoERM
from innermoe.load_data import load_all_data 
from innermoe.train_module_prefix import train_rm


def phase_train(innerMoERM, data, phase, mode, router, tokenizer):
    """
    Train the unified InnerMoERM in a given training phase
    """
    print('-' * 10 + f'begin training {mode}' + '-' * 10)
    lr = config.phase_lrs[phase - 1]

    out_dir = os.path.join(config.out_dir, "unified")
    os.makedirs(os.path.join(out_dir, 'ckpts'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'pictures'), exist_ok=True)

    ckpt_path = os.path.join(out_dir, 'ckpts', mode + '.pth')
    if os.path.exists(ckpt_path):
        print(f'already trained {mode}, passed!')
        innerMoERM.load_state_dict(torch.load(ckpt_path, map_location=config.device))
        return

    acc = train_rm(innerMoERM, data, lr, mode, out_dir, router=router, tokenizer=tokenizer)

    innerMoERM.load_state_dict(torch.load(ckpt_path, map_location=config.device))
    innerMoERM = innerMoERM.to(config.device)
    print(f'finish training {mode}! best acc: {acc}')


def train_pipe(innerMoERM, router, tokenizer):
    """
    Train the unified InnerMoERM in three training phases sequentially,
    using router probabilities instead of separate models per category.
    """
    out_dir = os.path.join(config.out_dir, "unified")
    os.makedirs(out_dir, exist_ok=True)

    phase1_data, phase2_data, phase3_data = load_all_data(config.phasedata_dir)

    # Phase 1
    phase_train(innerMoERM, phase1_data, 1, 'phase1', router, tokenizer)

    # Phase 2
    innerMoERM.change_phase(2)
    for key, value in phase2_data.items():
        innerMoERM.change_cap(key)
        phase_train(innerMoERM, value, 2, f"phase2_{key}", router, tokenizer)

    # Phase 3
    innerMoERM.change_phase(3)
    phase_train(innerMoERM, phase3_data, 3, 'phase3', router, tokenizer)


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
