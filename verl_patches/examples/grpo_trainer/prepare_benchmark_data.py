#!/usr/bin/env python3
"""
Prepare medical benchmark data for verl evaluation during RL training.

Converts MedQA, MMLU (medical), MedXpertQA (text-only), and LLMEval-Med
into parquet format expected by verl's RLHFDataset.

Usage:
    python prepare_benchmark_data.py --output_dir /path/to/benchmarks
"""

import argparse
import csv
import json
import os
import random

import pandas as pd


ANSWER_INSTRUCTION_EN = (
    "\n\nPlease reason step by step, then provide your answer. "
    "The last line of your response should be of the following format: "
    "'Answer: $LETTER' (without quotes) where LETTER is one of the option letters."
)

ANSWER_INSTRUCTION_ZH = (
    "\n\n请逐步推理，然后给出你的答案。"
    "你回答的最后一行应该是如下格式：'答案：$字母'，其中字母是选项字母之一。"
)

MEDICAL_SUBJECTS = [
    "professional_medicine",
    "clinical_knowledge",
    "college_medicine",
    "medical_genetics",
    "anatomy",
    "college_biology",
    "nutrition",
    "virology",
]


def make_record(question_text, answer_letter, data_source):
    """Create a single parquet record in verl format."""
    return {
        "prompt": [{"role": "user", "content": question_text}],
        "reward_model": {"style": "rule", "ground_truth": answer_letter},
        "data_source": data_source,
    }


# ── MedQA ────────────────────────────────────────────────────────────────────

def prepare_medqa(output_dir):
    """Prepare MedQA from MedQA-master US 4-options test set (1,273 items)."""
    source_path = "/home/haowang/prefix_moe_rm/benchmarks/raw/MedQA-master/data_clean/questions/US/4_options/phrases_no_exclude_test.jsonl"
    if not os.path.exists(source_path):
        print(f"[MedQA] Source not found: {source_path}, skipping.")
        return None

    print(f"[MedQA] Loading from {source_path} ...")
    records = []
    with open(source_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line.strip())
            question = item["question"]
            options = item["options"]  # dict: {"A": "...", "B": "...", "C": "...", "D": "..."}
            answer_idx = item["answer_idx"]  # e.g. "C"

            # Format question with options
            formatted = question + "\n"
            for letter in sorted(options.keys()):
                formatted += f"\n{letter}. {options[letter]}"
            formatted += ANSWER_INSTRUCTION_EN

            records.append(make_record(formatted, answer_idx, "medqa"))

    df = pd.DataFrame(records)
    out_path = os.path.join(output_dir, "medqa.parquet")
    df.to_parquet(out_path, index=False, engine="pyarrow")
    print(f"[MedQA] Saved {len(df)} records to {out_path}")
    return out_path


# ── MMLU Medical ─────────────────────────────────────────────────────────────

def prepare_mmlu_medical(output_dir):
    """Prepare MMLU medical subset from simple-evals CSV or Absolute-Zero-Reasoner JSON."""
    # Try simple-evals CSV first (more complete)
    csv_path = "/home/haowang/prefix_moe_rm/benchmarks/raw/simple-evals/mmlu.csv"
    json_path = "/home/haowang/Absolute-Zero-Reasoner/data/mmlu.json"

    if os.path.exists(csv_path):
        print(f"[MMLU] Loading from {csv_path} ...")
        records = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                subject = row.get("Subject", "")
                if subject not in MEDICAL_SUBJECTS:
                    continue
                question = row["Question"]
                choices = {k: row[k] for k in ["A", "B", "C", "D"]}
                answer = row["Answer"].strip().upper()

                formatted = f"Answer the following multiple choice question about {subject.replace('_', ' ')}.\n\n{question}\n"
                for letter in ["A", "B", "C", "D"]:
                    formatted += f"\n{letter}. {choices[letter]}"
                formatted += ANSWER_INSTRUCTION_EN

                records.append(make_record(formatted, answer, "mmlu_medical"))

    elif os.path.exists(json_path):
        print(f"[MMLU] Loading from {json_path} ...")
        with open(json_path, "r", encoding="utf-8") as f:
            all_data = json.load(f)
        records = []
        for item in all_data:
            if item.get("subject") not in MEDICAL_SUBJECTS:
                continue
            # goal field already contains formatted question with instruction
            question = item["goal"]
            answer = item["answer"].strip().upper()
            if answer not in ("A", "B", "C", "D"):
                continue
            records.append(make_record(question + ANSWER_INSTRUCTION_EN, answer, "mmlu_medical"))
    else:
        print("[MMLU] No source found, skipping.")
        return None

    df = pd.DataFrame(records)
    out_path = os.path.join(output_dir, "mmlu_medical.parquet")
    df.to_parquet(out_path, index=False, engine="pyarrow")
    print(f"[MMLU] Saved {len(df)} records to {out_path}")
    return out_path


# ── MedXpertQA ───────────────────────────────────────────────────────────────

def prepare_medxpertqa(output_dir):
    """Prepare MedXpertQA text subset (2,450 items, 10 options A-J).

    Skips the multimodal (MM) subset since it requires image inputs.
    """
    source_path = "/home/haowang/prefix_moe_rm/benchmarks/raw/MedXpertQA/eval/data/medxpertqa/input/medxpertqa_text_input.jsonl"
    if not os.path.exists(source_path):
        print(f"[MedXpertQA] Source not found: {source_path}, skipping.")
        return None

    print(f"[MedXpertQA] Loading from {source_path} ...")
    records = []
    with open(source_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line.strip())
            question = item["question"]  # Already contains "Answer Choices: (A)...(J)..."
            label = item["label"]  # e.g. ["E"]
            answer_letter = label[0].strip().upper()

            # The question field already has options embedded, just add instruction
            formatted = question + ANSWER_INSTRUCTION_EN

            records.append(make_record(formatted, answer_letter, "medxpertqa"))

    df = pd.DataFrame(records)
    out_path = os.path.join(output_dir, "medxpertqa.parquet")
    df.to_parquet(out_path, index=False, engine="pyarrow")
    print(f"[MedXpertQA] Saved {len(df)} records to {out_path}")
    return out_path


# ── LLMEval-Med ──────────────────────────────────────────────────────────────

def prepare_llmeval_med(output_dir):
    """Prepare LLMEval-Med (667 open-ended Chinese medical questions).

    NOTE: This is NOT multiple-choice. Questions are open-ended and the original
    evaluation uses GPT-4 as judge (1-5 score). Here we convert it to the verl
    format so that responses can be generated and dumped for external evaluation.
    The ground_truth contains the expert reference answer for post-hoc comparison.
    Scoring via medical_benchmark.compute_score will return 0.0 (no MCQ answer
    to extract), so the 'accuracy' metric is not meaningful for this benchmark.
    Use the dumped outputs in benchmark_data_dir for manual or GPT-4 evaluation.
    """
    source_path = "/home/haowang/prefix_moe_rm/benchmarks/raw/LLMEval-Med-main/dataset/dataset.json"
    if not os.path.exists(source_path):
        print(f"[LLMEval-Med] Source not found: {source_path}, skipping.")
        return None

    print(f"[LLMEval-Med] Loading from {source_path} ...")
    with open(source_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    records = []
    reference_answers = []  # Save separately since they're long text
    for category, items in data.items():
        for item in items:
            question = item.get("problem", "")
            reference_answer = item.get("sanswer", "")
            if not question:
                continue

            # No MCQ instruction for open-ended questions
            formatted = question

            # Use empty ground_truth for parquet compatibility (long text causes pyarrow issues).
            # Reference answers are saved in a separate JSON file for external evaluation.
            records.append({
                "prompt": [{"role": "user", "content": formatted}],
                "reward_model": {"style": "rule", "ground_truth": ""},
                "data_source": "llmeval_med",
            })
            reference_answers.append({
                "question": question,
                "reference_answer": reference_answer,
                "category": category,
            })

    df = pd.DataFrame(records)
    out_path = os.path.join(output_dir, "llmeval_med.parquet")
    df.to_parquet(out_path, index=False, engine="pyarrow")

    # Save reference answers separately for external GPT-4 evaluation
    ref_path = os.path.join(output_dir, "llmeval_med_references.json")
    with open(ref_path, "w", encoding="utf-8") as f:
        json.dump(reference_answers, f, ensure_ascii=False, indent=2)
    print(f"[LLMEval-Med] Reference answers saved to {ref_path}")
    print(f"[LLMEval-Med] Saved {len(df)} records to {out_path}")
    print(f"[LLMEval-Med] WARNING: This is an open-ended benchmark. Accuracy metric is NOT meaningful.")
    print(f"[LLMEval-Med]   Use dumped outputs for external GPT-4 judge evaluation.")
    return out_path


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Prepare medical benchmark data for verl evaluation")
    parser.add_argument("--output_dir", type=str, default="/home/haowang/prefix_moe_rm/benchmarks",
                        help="Output directory for parquet files")
    parser.add_argument("--skip_llmeval", action="store_true",
                        help="Skip LLMEval-Med (open-ended, requires GPT-4 judge)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    generated_files = []

    out = prepare_medqa(args.output_dir)
    if out:
        generated_files.append(out)

    out = prepare_mmlu_medical(args.output_dir)
    if out:
        generated_files.append(out)

    out = prepare_medxpertqa(args.output_dir)
    if out:
        generated_files.append(out)

    if not args.skip_llmeval:
        out = prepare_llmeval_med(args.output_dir)
        if out:
            generated_files.append(out)

    print(f"\n{'='*60}")
    print(f"Generated {len(generated_files)} benchmark parquet files:")
    for f in generated_files:
        df = pd.read_parquet(f)
        print(f"  {f}: {len(df)} records")

    if generated_files:
        files_str = ",".join(generated_files)
        print(f"\nTo use in training, add to your launch script:")
        print(f'  trainer.benchmark_files="[{files_str}]"')


if __name__ == "__main__":
    main()
