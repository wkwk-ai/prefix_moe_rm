import torch
import os

model_name_or_path = "/mnt/data/model/Qwen2.5-1.5B-Instruct"

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

rawdata_dir = './Ernie-rlhf'
os.makedirs(rawdata_dir, exist_ok=True)

out_dir = './results'
os.makedirs(out_dir, exist_ok=True)

phasedata_dir = os.path.join(out_dir, 'phasedata')
os.makedirs(phasedata_dir, exist_ok=True)
