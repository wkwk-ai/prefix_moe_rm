import torch
import os

model_name_or_path = "/mnt/data/model/Qwen2.5-1.5B-Instruct"

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = None

rawdata_dir = './prepare_data/Ernie-rlhf'
os.makedirs(rawdata_dir, exist_ok=True)

makedata_dir = 'Ernie-rlhf'
os.makedirs(makedata_dir, exist_ok=True)

out_dir = './results'
os.makedirs(out_dir, exist_ok=True)

phasedata_dir = os.path.join('./prepare_data/results', 'phasedata')
os.makedirs(phasedata_dir, exist_ok=True)


# ============ innermodel.config ============
categories = ['L1_recall', 'L2_analysis', 'L3_decision', 'L4_synthesis']
cat_list = categories


