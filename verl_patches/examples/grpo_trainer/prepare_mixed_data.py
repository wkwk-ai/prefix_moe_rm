#!/usr/bin/env python3
"""
Merge Ernie-rlhf (open-ended) and MedKgQA (MCQ) parquets into mixed training data.

Usage:
    python prepare_mixed_data.py \
        --ernie /home/haowang/prefix_moe_rm/prepare_data/Ernie-rlhf/Ernie-rlhf-train.parquet \
        --medkgqa /home/haowang/prefix_moe_rm/prepare_data/medkgqa/medkgqa-train.parquet \
        --output /home/haowang/prefix_moe_rm/prepare_data/mixed/mixed-train.parquet \
        --ratio 1.0
"""

import argparse
import os

import pandas as pd


def prepare_mixed(ernie_path, medkgqa_path, output_path, ratio):
    print(f"Loading Ernie-rlhf data from: {ernie_path}")
    df_ernie = pd.read_parquet(ernie_path)
    print(f"  Ernie-rlhf: {len(df_ernie)} records")

    print(f"Loading MedKgQA data from: {medkgqa_path}")
    df_medkgqa = pd.read_parquet(medkgqa_path)
    print(f"  MedKgQA: {len(df_medkgqa)} records")

    # Ensure Ernie-rlhf has required columns (add missing ones with defaults)
    if "cognitive_level" not in df_ernie.columns:
        df_ernie["cognitive_level"] = ""
    if "graph_idx" not in df_ernie.columns:
        df_ernie["graph_idx"] = -1

    # Ensure MedKgQA has all columns Ernie-rlhf has
    if "graph_idx" not in df_medkgqa.columns:
        df_medkgqa["graph_idx"] = -1

    # Apply ratio: subsample MedKgQA if ratio < 1, or Ernie-rlhf if ratio > 1
    # ratio = len(medkgqa_used) / len(ernie_used)
    if ratio < 1.0:
        n_medkgqa = int(len(df_medkgqa) * ratio)
        df_medkgqa = df_medkgqa.sample(n=n_medkgqa, random_state=42)
        print(f"  Subsampled MedKgQA to {len(df_medkgqa)} (ratio={ratio})")
    elif ratio > 1.0:
        n_ernie = int(len(df_ernie) / ratio)
        df_ernie = df_ernie.sample(n=n_ernie, random_state=42)
        print(f"  Subsampled Ernie-rlhf to {len(df_ernie)} (ratio={ratio})")

    # Align columns
    all_cols = sorted(set(df_ernie.columns) | set(df_medkgqa.columns))
    for col in all_cols:
        if col not in df_ernie.columns:
            df_ernie[col] = ""
        if col not in df_medkgqa.columns:
            df_medkgqa[col] = ""

    # Concatenate
    df_mixed = pd.concat([df_ernie, df_medkgqa], ignore_index=True)

    # Shuffle
    df_mixed = df_mixed.sample(frac=1.0, random_state=42).reset_index(drop=True)

    # Drop extra columns that verl's DataProto doesn't handle well (e.g., string columns
    # cause issues with repeat/chunk). Keep only the core columns needed by the pipeline.
    core_columns = ["prompt", "raw_prompt", "reward_model", "data_source"]
    extra_keep = [c for c in df_mixed.columns if c in core_columns]
    df_mixed = df_mixed[extra_keep]

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df_mixed.to_parquet(output_path, index=False, engine="pyarrow")

    print(f"\nMixed dataset summary:")
    print(f"  Total: {len(df_mixed)} records")
    print(f"  Data source distribution:")
    for ds, count in df_mixed["data_source"].value_counts().items():
        print(f"    {ds}: {count}")
    print(f"  Saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ernie", required=True, help="Ernie-rlhf parquet")
    parser.add_argument("--medkgqa", required=True, help="MedKgQA parquet (with cognitive levels)")
    parser.add_argument("--output", required=True, help="Output mixed parquet")
    parser.add_argument("--ratio", type=float, default=1.0,
                        help="Ratio of MedKgQA/Ernie-rlhf. 1.0=use all of both. "
                             "<1.0=subsample MedKgQA. >1.0=subsample Ernie-rlhf.")
    args = parser.parse_args()

    prepare_mixed(args.ernie, args.medkgqa, args.output, args.ratio)
