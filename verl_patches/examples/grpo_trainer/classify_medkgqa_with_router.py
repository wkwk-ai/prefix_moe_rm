#!/usr/bin/env python3
"""
Classify MedKgQA questions with the trained router to get cognitive levels.
Updates the parquet produced by prepare_medkgqa_for_grpo.py with cognitive_level column.

Usage:
    python classify_medkgqa_with_router.py \
        --parquet /home/haowang/prefix_moe_rm/prepare_data/medkgqa/medkgqa-train.parquet \
        --router_ckpt /home/haowang/prefix_moe_rm/results/router \
        --output /home/haowang/prefix_moe_rm/prepare_data/medkgqa/medkgqa-train.parquet
"""

import argparse
import sys
import os

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

# Add project root to path for router import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from router.modeling import Router


CAT_LIST = ["L1_recall", "L2_analysis", "L3_decision", "L4_synthesis"]


def classify_questions(parquet_path, router_ckpt, output_path, batch_size=32):
    print(f"Loading router from: {router_ckpt}")
    router = Router.from_pretrained(router_ckpt)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    router = router.to(device)
    router.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        router.model_name_or_path, trust_remote_code=True
    )

    print(f"Loading parquet from: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    print(f"Loaded {len(df)} records")

    user_token = "<extra_0>"
    end_token = "<extra_2>"

    cognitive_levels = []
    for start in tqdm(range(0, len(df), batch_size), desc="Classifying"):
        batch_rows = df.iloc[start : start + batch_size]
        texts = []
        for _, row in batch_rows.iterrows():
            # Extract the question text from the prompt
            prompt_content = row["prompt"][0]["content"]
            text = f"{user_token}{prompt_content}{end_token}"
            texts.append(text)

        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=512,
            padding=True,
            return_tensors="pt",
        )
        input_ids = tokenized["input_ids"].to(device)
        attention_mask = tokenized["attention_mask"].to(device)

        with torch.no_grad():
            output = router(input_ids=input_ids, attention_mask=attention_mask)
            logits = output["logits"]  # [B, C]
            preds = torch.argmax(logits, dim=-1)  # [B]

        for pred_idx in preds.cpu().tolist():
            cognitive_levels.append(CAT_LIST[pred_idx])

    df["cognitive_level"] = cognitive_levels

    # Print distribution
    from collections import Counter
    dist = Counter(cognitive_levels)
    print(f"Cognitive level distribution: {dict(dist)}")

    # Save
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_parquet(output_path, index=False, engine="pyarrow")
    print(f"Saved {len(df)} records with cognitive levels to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", required=True, help="Input parquet from prepare_medkgqa_for_grpo.py")
    parser.add_argument("--router_ckpt", required=True, help="Path to trained router checkpoint")
    parser.add_argument("--output", required=True, help="Output parquet path (can overwrite input)")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    classify_questions(args.parquet, args.router_ckpt, args.output, args.batch_size)
