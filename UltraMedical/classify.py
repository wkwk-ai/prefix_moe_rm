import json
import os
from openai import OpenAI
import time
from tqdm import tqdm

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

client = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
    base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
)

# 输入输出文件
input_file = "UltraMedical_sampled_1000.json"
output_file = "UltraMedical_sampled_1000_classified.json"

# Prompt 模板
PROMPT_TEMPLATE = """
You are a cognitive-level classifier for medical questions. 
Based only on the stem and options, without introducing external assumptions, classify the question into one of three levels, according to a multi-cognitive-level evaluation framework in the medical domain inspired by Bloom’s Taxonomy:

• L1 — Preliminary Knowledge Grasp:
  - Mastery of basic medical knowledge, understanding of concepts and principles.
  - Answers can be directly found in textbooks or lecture notes.
  - Example: "Which hormone stimulates X?"

• L2 — Comprehensive Knowledge Application:
  - Application of basic knowledge to short clinical cases.
  - Requires integrating information to analyze or make a diagnosis/treatment choice.
  - Does not require multi-step clinical planning.
  - Example: Analyze lab data, simple case diagnosis.

• L3 — Advanced Clinical Reasoning & Planning:
  - Requires multi-step clinical reasoning and planning.
  - Involves designing or evaluating treatment plans, or making complex decisions under uncertainty.
  - Example: Design a complete treatment plan or manage uncertainty.

Decision rules:
- If a single fact suffices → L1
- If multiple knowledge points must be integrated → L2
- If multi-step reasoning or strategy is required → L3
- For mixed cases, assign the highest required level (L3 > L2 > L1).

Output format:
- First line: output only the label inside a LaTeX box: \\boxed{{L1|L2|L3}}
- Second line (optional): ≤25 words, short reason in English.

Example input 1:
"Which hormones are secreted by the adrenal medulla?"
Example output 1:
\\boxed{{L1}}

Example input 2:
"A 25-year-old male presents with acute fever, cough with rust-colored sputum, 
and lung auscultation reveals crackles. What is the most likely diagnosis?"
Example output 2:
\\boxed{{L2}}

Example input 3:
"A 58-year-old male with a long history of smoking presents with chronic cough, sputum, 
and weight loss. Lab: sputum cytology shows cancer cells. Chest CT reveals a right upper lobe mass. 
Please design an initial treatment plan and explain your diagnostic reasoning."
Example output 3:
\\boxed{{L3}}

Medical question:
{question_text}
"""

# 读取数据
with open(input_file, "r", encoding="utf-8") as f:
    dataset = json.load(f)

results = []

for idx, item in enumerate(tqdm(dataset, desc="Classifying questions")):
    question_text = item["conversations"][0]["value"]

    prompt = PROMPT_TEMPLATE.format(question_text=question_text)

    # 自动重试机制，防止 502 或其他错误中断
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a cognitive-level classifier for medical questions."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0
            )
            label = response.choices[0].message.content.strip()
            break
        except Exception as e:
            print(f"Error on item {idx}, attempt {attempt+1}: {e}")
            time.sleep(2)
            label = "ERROR"

    item["cognitive_level"] = label
    results.append(item)

# 保存结果
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"分类完成，结果已保存到 {output_file}")


