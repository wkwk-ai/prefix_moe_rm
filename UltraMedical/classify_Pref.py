import json
from openai import OpenAI
import time
import re
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# API配置
API_KEY = os.environ["OPENAI_API_KEY"]

def create_client():
    """为每个线程创建独立的client实例，使用环境变量中的代理设置"""
    return OpenAI(
        api_key=API_KEY,
        base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    )

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 输入输出文件
input_file = "UltraMedical_Pref_sampled_10000.json"
output_file = "UltraMedical_Pref_sampled_10000_classified.json"

# Prompt 模板
PROMPT_TEMPLATE = """
You are a cognitive-level classifier for medical questions.
Based only on the question stem and options (if any), without introducing external assumptions, classify the question into one of four levels. This framework is grounded in Anderson & Krathwohl’s (2001) revision of Bloom’s Taxonomy, adapted for the medical domain:

• L1 — Medical Knowledge Recall (Remember / Understand):
  - Direct retrieval of factual medical knowledge: definitions, mechanisms, classifications, anatomical structures.
  - Single-concept understanding; answers can be found directly in standard textbooks.
  - Example: "Which hormone stimulates X?" / "What are the symptoms of disease Y?"

• L2 — Clinical Analysis & Application (Apply / Analyze):
  - Application of medical knowledge to clinical scenarios (e.g., patient vignettes).
  - Requires integrating multiple clinical findings (symptoms, labs, imaging) to reach a diagnosis or select a treatment.
  - Single-step or short-chain reasoning; does NOT require designing a full management plan.
  - Example: "A 45-year-old presents with X, Y, Z. What is the most likely diagnosis?"

• L3 — Clinical Reasoning & Decision-Making (Evaluate):
  - Multi-step clinical reasoning with explicit planning or decision-making.
  - Involves designing treatment plans, evaluating therapeutic options, managing comorbidities, or making decisions under uncertainty.
  - Focuses on real patient management and clinical judgment.
  - Example: "Design an initial treatment plan for a patient with stage III lung cancer and COPD."

• L4 — Biomedical Synthesis & Innovation (Create):
  - Cross-domain knowledge synthesis, research protocol design, literature review, or critical evaluation.
  - Ethical, policy, or public health analysis in healthcare contexts.
  - Generation of novel hypotheses, methodological frameworks, or comprehensive analytical reports.
  - Example: "Design a research protocol to evaluate the effect of X on Y." / "Discuss the ethical implications of gene editing in human embryos."

Decision rules:
- If a single fact or concept suffices → L1
- If clinical data must be integrated for diagnosis or short-chain application → L2
- If multi-step clinical reasoning, treatment planning, or patient management is required → L3
- If cross-domain synthesis, research design, ethical analysis, or creative generation is required → L4
- For mixed cases, assign the highest required level (L4 > L3 > L2 > L1).

Output format:
- First line: output only the label inside a LaTeX box: \\boxed{{L1|L2|L3|L4}}
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
and weight loss. Chest CT reveals a right upper lobe mass.
Please design an initial treatment plan and explain your diagnostic reasoning."
Example output 3:
\\boxed{{L3}}

Example input 4:
"Design a research protocol that evaluates the correlation between antibiotic exposure
in the neonatal period and the later development of autoimmune conditions."
Example output 4:
\\boxed{{L4}}

Medical question:
{question_text}
"""

# 读取数据
with open(input_file, "r", encoding="utf-8") as f:
    dataset = json.load(f)

# 线程安全的锁和结果列表
results_lock = Lock()
results = [None] * len(dataset)

def classify_item(args):
    """分类单个item的函数"""
    idx, item = args
    question_text = item["prompt"]
    prompt = PROMPT_TEMPLATE.format(question_text=question_text)

    # 为每个线程创建独立的client
    client = create_client()

    # 自动重试机制，防止 502 或其他错误中断
    label = "ERROR"
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
            if attempt < 2:  # 不是最后一次尝试
                time.sleep(2 ** attempt)  # 指数退避
            else:
                print(f"Error on item {idx}, attempt {attempt+1}: {e}")

    # 提取认知级别标签（从\boxed{L1}等格式中提取）
    label_match = re.search(r'\\boxed\{(L[1-4])\}', label)
    if label_match:
        level = label_match.group(1)
        # 转换为训练格式
        label_map = {"L1": "L1_recall", "L2": "L2_analysis", "L3": "L3_decision", "L4": "L4_synthesis"}
        item["label"] = label_map.get(level, "L1_recall")
    else:
        # 如果无法提取，尝试直接匹配
        if "L4" in label.upper():
            item["label"] = "L4_synthesis"
        elif "L3" in label.upper():
            item["label"] = "L3_decision"
        elif "L2" in label.upper():
            item["label"] = "L2_analysis"
        else:
            item["label"] = "L1_recall"  # 默认值
    
    item["cognitive_level"] = label
    return idx, item

# 并行处理
max_workers = 50  # 并发线程数，可以根据API限制调整
proxy_info = os.environ.get("HTTP_PROXY") or os.environ.get("http_proxy") or "未设置"
print(f"开始并行标注，使用 {max_workers} 个线程，HTTP代理: {proxy_info}")

with ThreadPoolExecutor(max_workers=max_workers) as executor:
    # 提交所有任务
    futures = {executor.submit(classify_item, (idx, item)): idx for idx, item in enumerate(dataset)}
    
    # 使用tqdm显示进度
    with tqdm(total=len(dataset), desc="Classifying questions") as pbar:
        for future in as_completed(futures):
            try:
                idx, item = future.result()
                results[idx] = item
                pbar.update(1)
            except Exception as e:
                original_idx = futures[future]
                print(f"Error processing item {original_idx}: {e}")
                pbar.update(1)

# 过滤掉None值（如果有错误的话）
results = [r for r in results if r is not None]

# 保存结果
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"分类完成，结果已保存到 {output_file}")