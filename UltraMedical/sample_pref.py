import json
import random
from collections import defaultdict, Counter

with open("UltraMedical_Pref/datasets--TsinghuaC3I--UltraMedical-Preference/snapshots/761eb7935310ba662a96d93c5af342e5269d5759/data/train.json", "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"原始数据条数: {len(data)}")

# 按label_type分组数据
data_by_type = defaultdict(list)
for item in data:
    label_type = item.get("label_type", "unknown")
    data_by_type[label_type].append(item)

# 显示原始分布
print("\n原始数据label_type分布:")
type_counts = Counter([item.get("label_type", "unknown") for item in data])
for label_type, count in type_counts.most_common():
    print(f"  {label_type}: {count} ({count/len(data)*100:.2f}%)")

# 分层采样：按原始比例采样
total_samples = 10000
sampled_data = []

for label_type, items in data_by_type.items():
    # 计算该类别的采样数量（按原始比例）
    proportion = len(items) / len(data)
    sample_size = int(total_samples * proportion)
    
    # 如果该类别的数据量少于采样数量，则全部采样
    sample_size = min(sample_size, len(items))
    
    sampled = random.sample(items, sample_size)
    sampled_data.extend(sampled)
    print(f"\n{label_type}: 采样 {sample_size} 条 (原始 {len(items)} 条, 比例 {proportion*100:.2f}%)")

# 如果总数不足10000，从剩余数据中随机补充
if len(sampled_data) < total_samples:
    remaining = [item for item in data if item not in sampled_data]
    needed = total_samples - len(sampled_data)
    if len(remaining) >= needed:
        additional = random.sample(remaining, needed)
        sampled_data.extend(additional)
        print(f"\n补充采样: {needed} 条")
    else:
        print(f"\n警告: 数据不足，实际采样 {len(sampled_data)} 条")

# 随机打乱
random.shuffle(sampled_data)

# 显示采样后的分布
print("\n采样后数据label_type分布:")
sampled_type_counts = Counter([item.get("label_type", "unknown") for item in sampled_data])
for label_type, count in sampled_type_counts.most_common():
    print(f"  {label_type}: {count} ({count/len(sampled_data)*100:.2f}%)")

with open("UltraMedical_Pref_sampled_10000.json", "w", encoding="utf-8") as f:
    json.dump(sampled_data, f, ensure_ascii=False, indent=2)

print(f"\n已保存 {len(sampled_data)} 条分层采样样本到 UltraMedical_Pref_sampled_10000.json")