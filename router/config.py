import os
from base_config import *

cat_list = ['L1_recall', 'L2_analysis', 'L3_decision', 'L4_synthesis']

out_dir = os.path.join(out_dir, 'router')
os.makedirs(out_dir, exist_ok=True)

val_rate = 0.2
eval_samples = 500
batch_size = 8
num_epochs = 3
steps_per_eval = 1000

lr = 1e-5
