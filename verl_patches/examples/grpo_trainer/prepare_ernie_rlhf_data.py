#!/usr/bin/env python3
"""
Prepare Ernie-rlhf data for verl GRPO training.
Converts Ernie-rlhf-train.jsonl to parquet format expected by verl.
"""

import json
import os
import sys
import pandas as pd
from pathlib import Path

def load_jsonl(file_path):
    """Load data from jsonl file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue
    return data

def prepare_data(input_file, output_file):
    """Convert Ernie-rlhf data to verl format."""
    print(f"Loading data from: {input_file}")
    data = load_jsonl(input_file)
    print(f"Loaded {len(data)} samples")
    
    records = []
    for idx, item in enumerate(data):
        # Extract query (src)
        if "src" in item:
            if isinstance(item["src"], list):
                query = item["src"][-1]  # Take the last one if it's a list
            else:
                query = item["src"]
        else:
            continue
        
        # Create prompt in chat format
        prompt = [{"role": "user", "content": query}]
        
        # Create reward_model metadata
        # Use "rule" style to allow validation, even though we use model-based reward
        # The actual reward computation will still use the Prefix MoE model
        reward_model_info = {
            "style": "rule",  # Set to "rule" to allow validation
            "ground_truth": ""  # Empty ground truth for validation
        }
        
        # Create record
        # Note: raw_prompt is needed for Prefix MoE RM to extract query correctly
        record = {
            "prompt": prompt,
            "raw_prompt": prompt,  # Add raw_prompt for Prefix MoE RM compatibility
            "reward_model": reward_model_info,
            "data_source": item.get("label", "ernie_rlhf"),  # Use label as data_source if available
        }
        
        records.append(record)
    
    # Create DataFrame
    df = pd.DataFrame(records)
    
    # Save to parquet with proper format
    print(f"Saving {len(df)} records to: {output_file}")
    # Use engine='pyarrow' and ensure proper schema
    df.to_parquet(
        output_file, 
        index=False,
        engine='pyarrow'
    )
    print(f"Data preparation complete!")
    
    # Verify the file can be read back
    try:
        verify_df = pd.read_parquet(output_file)
        print(f"Verification: Successfully read back {len(verify_df)} records")
        print(f"Columns: {verify_df.columns.tolist()}")
    except Exception as e:
        print(f"Warning: Could not verify parquet file: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python prepare_ernie_rlhf_data.py <input_jsonl> <output_parquet>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
    
    prepare_data(input_file, output_file)

