#!/usr/bin/env python3
"""
Construct CoT preference pairs from generated MedKgQA responses.
Pairs correct-answer CoTs (chosen) with incorrect-answer CoTs (rejected),
attaching cognitive level labels for Prefix MoE RM training.

Usage:
    python construct_cot_preference_pairs.py \
        --responses /home/haowang/prefix_moe_rm/prepare_data/medkgqa/medkgqa_cot_responses.jsonl \
        --cognitive_levels /home/haowang/prefix_moe_rm/prepare_data/medkgqa/medkgqa-train.parquet \
        --output /home/haowang/prefix_moe_rm/prepare_data/Ernie-rlhf/MedKgQA-cot-pairs.jsonl \
        --max_pairs_per_question 4
"""

import argparse
import json
import os
import random
from collections import defaultdict
from itertools import product

import pandas as pd

MCQ_PROMPT_TEMPLATE = (
    "Please answer the following medical multiple choice question. "
    "Think step by step and provide your reasoning, then give your final answer.\n\n"
    "{question}\n{choices}\n\n"
    "Provide your reasoning, then state your final answer as "
    '"The answer is X" where X is A, B, C, or D.'
)


def load_jsonl(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def construct_pairs(responses_file, cognitive_levels_file, output_file, max_pairs_per_question, seed):
    random.seed(seed)

    print(f"Loading responses from: {responses_file}")
    responses = load_jsonl(responses_file)
    print(f"Loaded {len(responses)} responses")

    # Load cognitive levels from parquet (keyed by graph_idx)
    print(f"Loading cognitive levels from: {cognitive_levels_file}")
    df = pd.read_parquet(cognitive_levels_file)
    cog_level_map = {}
    for _, row in df.iterrows():
        gidx = row.get("graph_idx", -1)
        cog_level_map[gidx] = row.get("cognitive_level", "L1_recall")
    print(f"Loaded cognitive levels for {len(cog_level_map)} questions")

    # Group responses by question (graph_idx)
    grouped = defaultdict(lambda: {"correct": [], "incorrect": [], "meta": None})
    for resp in responses:
        gidx = resp["graph_idx"]
        if grouped[gidx]["meta"] is None:
            grouped[gidx]["meta"] = {
                "question": resp["question"],
                "choices": resp["choices"],
                "ground_truth": resp["ground_truth"],
            }
        if resp["is_correct"]:
            grouped[gidx]["correct"].append(resp["response"])
        else:
            grouped[gidx]["incorrect"].append(resp["response"])

    # Construct preference pairs
    pairs = []
    skipped_no_correct = 0
    skipped_no_incorrect = 0

    for gidx, group in grouped.items():
        correct_cots = group["correct"]
        incorrect_cots = group["incorrect"]
        meta = group["meta"]

        if not correct_cots:
            skipped_no_correct += 1
            continue
        if not incorrect_cots:
            skipped_no_incorrect += 1
            continue

        # Get cognitive level
        cog_level = cog_level_map.get(gidx, "L1_recall")

        # Format the question as the prompt
        prompt_text = MCQ_PROMPT_TEMPLATE.format(
            question=meta["question"], choices=meta["choices"]
        )

        # Create pairs: sample from correct x incorrect
        all_combinations = list(product(correct_cots, incorrect_cots))
        random.shuffle(all_combinations)
        selected = all_combinations[:max_pairs_per_question]

        for chosen_cot, rejected_cot in selected:
            pair = {
                "src": [prompt_text],
                "response": [chosen_cot, rejected_cot],
                "rank": [1, 6],
                "label": cog_level,
            }
            pairs.append(pair)

    print(f"\nPair construction summary:")
    print(f"  Questions with valid pairs: {len(grouped) - skipped_no_correct - skipped_no_incorrect}")
    print(f"  Skipped (no correct response): {skipped_no_correct}")
    print(f"  Skipped (no incorrect response): {skipped_no_incorrect}")
    print(f"  Total preference pairs: {len(pairs)}")

    # Show cognitive level distribution of pairs
    from collections import Counter
    level_dist = Counter(p["label"] for p in pairs)
    print(f"  Cognitive level distribution: {dict(level_dist)}")

    # Shuffle and save
    random.shuffle(pairs)
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
    print(f"\nSaved {len(pairs)} pairs to: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--responses", required=True, help="CoT responses JSONL from generate_cot_for_medkgqa.py")
    parser.add_argument("--cognitive_levels", required=True, help="Parquet with cognitive_level column")
    parser.add_argument("--output", required=True, help="Output preference pairs JSONL")
    parser.add_argument("--max_pairs_per_question", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    construct_pairs(
        args.responses, args.cognitive_levels, args.output,
        args.max_pairs_per_question, args.seed,
    )
