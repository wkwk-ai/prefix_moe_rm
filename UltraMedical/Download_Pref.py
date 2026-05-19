import os
import pandas as pd
from huggingface_hub import snapshot_download
import pyarrow

# 获取当前 py 文件所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 1. 下载 UltraMedical-Preference 数据集到 py 文件所在目录下的 UltraMedical_Pref 文件夹
local_dir = os.path.join(current_dir, "UltraMedical_Pref")

snapshot_download(
    repo_id="TsinghuaC3I/UltraMedical-Preference",  # ✅ 改这里
    repo_type="dataset",
    cache_dir=local_dir,
    local_dir_use_symlinks=False,
    resume_download=True
)

print("✅ 数据集已下载到:", local_dir)

# 2. 找到 parquet 文件路径
parquet_file = None
for root, dirs, files in os.walk(local_dir):
    for f in files:
        if f.endswith(".parquet"):
            parquet_file = os.path.join(root, f)
            break

if parquet_file is None:
    raise FileNotFoundError("没有找到 parquet 文件，请检查下载目录")

print("📂 找到 parquet 文件:", parquet_file)

# 3. 读取 parquet 文件
df = pd.read_parquet(parquet_file, engine="pyarrow")

# 4. 保存为 CSV 和 JSON 到当前目录
csv_path = os.path.join(current_dir, "UltraMedical_Pref.csv")
json_path = os.path.join(current_dir, "UltraMedical_Pref.json")

df.to_csv(csv_path, index=False, encoding="utf-8-sig")
df.to_json(json_path, orient="records", force_ascii=False)

print("✅ 已保存为:", csv_path, "和", json_path)
print(df.head())
