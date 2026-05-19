"""
使用 Claude 对医疗问题进行 4 级认知层级分类。
基于 Anderson & Krathwohl (2001) 修订版 Bloom's Taxonomy。
"""
import json
from openai import OpenAI
import time
import re
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter

# API 配置
API_KEY = os.environ["OPENAI_API_KEY"]
BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.environ.get("CLASSIFIER_MODEL", "claude-opus-4-5-20251101")

# 输入输出文件
input_file = "UltraMedical_Pref_sampled_10000.json"
output_file = "UltraMedical_Pref_sampled_10000_classified.json"

# Prompt 模板
PROMPT_TEMPLATE = """You are a cognitive-level classifier for medical questions.
Based only on the question stem and options (if any), without introducing external assumptions, classify the question into one of four levels. This framework is grounded in Anderson & Krathwohl's (2001) revision of Bloom's Taxonomy, adapted for the medical domain:

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

Medical question:
{question_text}"""


def create_client():
    return OpenAI(api_key=API_KEY, base_url=BASE_URL)


def call_model(client, prompt):
    response = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0,
        max_tokens=100,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content or ""


def parse_label(raw_response):
    """从模型输出中解析认知层级标签"""
    label_match = re.search(r'\\boxed\{(L[1-4])\}', raw_response)
    if label_match:
        level = label_match.group(1)
        label_map = {"L1": "L1_recall", "L2": "L2_analysis", "L3": "L3_decision", "L4": "L4_synthesis"}
        return label_map.get(level, "L1_recall")

    # fallback: 直接字符串匹配
    upper = raw_response.upper()
    if "L4" in upper:
        return "L4_synthesis"
    elif "L3" in upper:
        return "L3_decision"
    elif "L2" in upper:
        return "L2_analysis"
    return "L1_recall"


def classify_item(args):
    """分类单个 item"""
    idx, item = args
    question_text = item["prompt"]
    prompt = PROMPT_TEMPLATE.format(question_text=question_text)

    client = create_client()

    raw_response = "ERROR"
    for attempt in range(3):
        try:
            raw_response = call_model(client, prompt)
            break
        except Exception as e:
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                print(f"Error on item {idx}, attempt {attempt+1}: {e}")

    item["label"] = parse_label(raw_response)
    item["cognitive_level"] = raw_response
    return idx, item


def main():
    # 读取数据
    with open(input_file, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    print(f"Loaded {len(dataset)} samples from {input_file}")
    print(f"Model: {MODEL_NAME}")

    # 先测试一条，确认 API 可用
    print("Testing API connection...")
    try:
        client = create_client()
        test_resp = call_model(client, "Classify: 'What is the normal range of blood glucose?' Output: \\boxed{L1}")
        print(f"API test OK: {test_resp[:80]}")
    except Exception as e:
        print(f"API test FAILED: {e}")
        return

    # 并行处理
    max_workers = 20
    results = [None] * len(dataset)

    print(f"Starting classification with {max_workers} workers...")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(classify_item, (idx, item)): idx for idx, item in enumerate(dataset)}

        with tqdm(total=len(dataset), desc="Classifying") as pbar:
            for future in as_completed(futures):
                try:
                    idx, item = future.result()
                    results[idx] = item
                except Exception as e:
                    original_idx = futures[future]
                    print(f"Error processing item {original_idx}: {e}")
                pbar.update(1)

    # 过滤 None
    results = [r for r in results if r is not None]

    # 统计分布
    label_counts = Counter(r["label"] for r in results)
    print(f"\nClassification distribution:")
    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count} ({count/len(results)*100:.1f}%)")

    # 保存结果
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(results)} results to {output_file}")


if __name__ == "__main__":
    main()
