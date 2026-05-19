import json
import random

with open("UltraMedical.json", "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"原始数据条数: {len(data)}")

sampled_data = random.sample(data, 1000)

with open("UltraMedical_sampled_1000.json", "w", encoding="utf-8") as f:
    json.dump(sampled_data, f, ensure_ascii=False, indent=2)

print("已保存 1000 条随机样本到 UltraMedical_sampled_1000.json")
