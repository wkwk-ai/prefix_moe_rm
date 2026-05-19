from peft import LoraConfig
import os
from base_config import *

out_dir = os.path.join(out_dir, 'innermoe')
os.makedirs(out_dir, exist_ok=True)

# ============ 新增: 类别定义 ============
categories = ['L1_recall', 'L2_analysis', 'L3_decision', 'L4_synthesis']
cat_list = categories

#eval_samples = 500
#max_no_adding_times = 20
#steps_per_eval = 100

learning_rate = 5e-5  # 稍微降低学习率，更稳定
#phase1_data_rate = 0.6
val_rate = 0.2
batch_size = 1
num_epochs = 3  # 3个epoch
