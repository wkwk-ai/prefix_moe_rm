import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 读取分类结果
with open("UltraMedical_sampled_1000_classified.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# 提取 cognitive_level
df = pd.DataFrame(data)
df['cognitive_level'] = df['cognitive_level'].str.extract(r'\\boxed\{(L\d)\}')  # 提取 L1/L2/L3

counts = df['cognitive_level'].value_counts()
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