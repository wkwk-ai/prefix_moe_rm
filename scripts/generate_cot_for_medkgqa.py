#!/usr/bin/env python3
"""
Generate CoT responses for MedKgQA questions using vLLM batch inference.
These responses will be used to construct preference pairs for RM training.

Usage:
    python generate_cot_for_medkgqa.py \
        --input /home/haowang/Fleming-R1/data/MedKgQA_en.jsonl \
        --output /home/haowang/prefix_moe_rm/prepare_data/medkgqa/medkgqa_cot_responses.jsonl \
        --model /mnt/data/model/Qwen2.5-1.5B-Instruct \
        --num_samples 8 \
        --temperature 0.7
"""

import argparse
import json
import os
import sys
import re

# Add project root for medical_benchmark import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "verl"))

MCQ_PROMPT_TEMPLATE = (
    "Please answer the following medical multiple choice question. "
    "Think step by step and provide your reasoning, then give your final answer.\n\n"
    "{question}\n{choices}\n\n"
    "Provide your reasoning, then state your final answer as "
    '"The answer is X" where X is A, B, C, or D.'
)


def extract_mcq_answer(response_str):
    """Extract MCQ answer letter from model response (same logic as medical_benchmark.py)."""
    if not response_str:
        return None

    patterns = [
        r"(?i)Answer\s*:\s*([A-J])",
        r"(?i)the\s+answer\s+is\s+\(?([A-J])\)?",
        r"(?i)answer\s+is\s+\(?([A-J])\)?",
        r"\\boxed\{([A-J])\}",
        r"(?i)答案\s*[：:]\s*([A-J])",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, response_str)
        if matches:
            return matches[-1].upper()

    # Fallback: last standalone letter
    fallback = re.findall(r"(?:^|[\s\.\,\:\;\(\)\[\]\{\}])([A-D])(?:[\s\.\,\:\;\)\]\}]|$)", response_str)
    if fallback:
        return fallback[-1].upper()
    return None


def load_jsonl(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def generate_responses(input_file, output_file, model_path, num_samples, temperature, max_tokens, tp_size):
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    print(f"Loading data from: {input_file}")
    data = load_jsonl(input_file)
    print(f"Loaded {len(data)} questions")

    print(f"Loading model from: {model_path}")
    llm = LLM(model=model_path, tensor_parallel_size=tp_size, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    sampling_params = SamplingParams(
        n=num_samples,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=0.95,
    )

    # Build prompts using chat template
    prompts = []
    for item in data:
        prompt_text = MCQ_PROMPT_TEMPLATE.format(
            question=item["question"], choices=item["choices"]
        )
        messages = [{"role": "user", "content": prompt_text}]
        formatted = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        prompts.append(formatted)

    print(f"Generating {num_samples} responses per question for {len(prompts)} questions...")
    outputs = llm.generate(prompts, sampling_params)

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    total_correct = 0
    total_responses = 0

    with open(output_file, "w", encoding="utf-8") as f:
        for item, output in zip(data, outputs):
            ground_truth = item["answer"].strip().upper()
            for completion in output.outputs:
                response_text = completion.text
                answer_extracted = extract_mcq_answer(response_text)
                is_correct = (answer_extracted == ground_truth) if answer_extracted else False

                record = {
                    "graph_idx": item["graph_idx"],
                    "question": item["question"],
                    "choices": item["choices"],
                    "ground_truth": ground_truth,
                    "response": response_text,
                    "answer_extracted": answer_extracted,
                    "is_correct": is_correct,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

                total_responses += 1
                if is_correct:
                    total_correct += 1

    accuracy = total_correct / total_responses if total_responses > 0 else 0
    print(f"Generated {total_responses} responses")
    print(f"Overall accuracy: {total_correct}/{total_responses} = {accuracy:.2%}")
    print(f"Saved to: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="MedKgQA JSONL file")
    parser.add_argument("--output", required=True, help="Output JSONL with CoT responses")
    parser.add_argument("--model", required=True, help="Model path for vLLM")
    parser.add_argument("--num_samples", type=int, default=8, help="Responses per question")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--tp_size", type=int, default=1, help="Tensor parallel size for vLLM")
    args = parser.parse_args()

    generate_responses(
        args.input, args.output, args.model,
        args.num_samples, args.temperature, args.max_tokens, args.tp_size,
    )
