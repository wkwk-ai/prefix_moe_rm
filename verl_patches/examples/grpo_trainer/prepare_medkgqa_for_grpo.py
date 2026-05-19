#!/usr/bin/env python3
"""
Convert Fleming-R1 MedKgQA English data to verl GRPO parquet format.

Usage:
    python prepare_medkgqa_for_grpo.py <input_jsonl> <output_parquet>

Example:
    python prepare_medkgqa_for_grpo.py \
        /home/haowang/Fleming-R1/data/MedKgQA_en.jsonl \
        /home/haowang/prefix_moe_rm/prepare_data/medkgqa/medkgqa-train.parquet
"""

import json
import os
import sys

import pandas as pd

MCQ_PROMPT_TEMPLATE = (
    "Please answer the following medical multiple choice question. "
    "Think step by step and provide your reasoning, then give your final answer.\n\n"
    "{question}\n{choices}\n\n"
    "Provide your reasoning, then state your final answer as "
    "\"The answer is X\" where X is A, B, C, or D."
)


def load_jsonl(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def prepare_data(input_file, output_file):
    print(f"Loading data from: {input_file}")
    data = load_jsonl(input_file)
    print(f"Loaded {len(data)} samples")

    records = []
    for item in data:
        question = item["question"]
        choices = item["choices"]
        answer = item["answer"].strip().upper()
        graph_idx = item.get("graph_idx", -1)

        prompt_text = MCQ_PROMPT_TEMPLATE.format(question=question, choices=choices)
        prompt = [{"role": "user", "content": prompt_text}]

        record = {
            "prompt": prompt,
            "raw_prompt": prompt,
            "reward_model": {
                "style": "rule",
                "ground_truth": answer,
            },
            "data_source": "medkgqa",
            "cognitive_level": "",  # to be filled by classify_medkgqa_with_router.py
            "graph_idx": graph_idx,
        }
        records.append(record)

    df = pd.DataFrame(records)

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    print(f"Saving {len(df)} records to: {output_file}")
    df.to_parquet(output_file, index=False, engine="pyarrow")

    # Verify
    verify_df = pd.read_parquet(output_file)
    print(f"Verification: {len(verify_df)} records, columns: {verify_df.columns.tolist()}")

    # Show answer distribution
    answers = [r["reward_model"]["ground_truth"] for r in records]
    from collections import Counter
    print(f"Answer distribution: {dict(Counter(answers))}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python prepare_medkgqa_for_grpo.py <input_jsonl> <output_parquet>")
        sys.exit(1)

    prepare_data(sys.argv[1], sys.argv[2])
