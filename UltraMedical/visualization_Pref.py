import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

# 读取数据
with open("UltraMedical_Pref_sampled_1000_classified.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# 转成 DataFrame
df = pd.DataFrame(data)

# 提取认知水平 (L1 / L2 / L3)
df["cognitive_level"] = df["cognitive_level"].apply(lambda x: re.search(r"L\d", x).group(0) if pd.notna(x) else None)

# 统计分布
counts = df["cognitive_level"].value_counts().sort_index()
print(counts)

sns.set_theme(style="whitegrid")

# 柱状图
plt.figure(figsize=(6,4))
sns.barplot(x=counts.index, y=counts.values, palette="Set2")
plt.xlabel("Cognitive Level")
plt.ylabel("Number of Questions")
plt.title("Distribution of Medical Questions by Cognitive Level")
plt.show()

# 饼图
plt.figure(figsize=(6,6))
plt.pie(counts.values, labels=counts.index, autopct='%1.1f%%', colors=sns.color_palette("Set2"))
plt.title("Proportion of Questions by Cognitive Level")
plt.show()
