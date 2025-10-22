import os
import json
from tqdm import tqdm
from transformers import AutoTokenizer
import random
import pickle
import innermoe.config as config


def load_all_data(phasedata_dir):
    """
    Load and tokenize data from all categories, only from cat.jsonl
    """
    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path, trust_remote_code=True)

    all_data = []

    for cat in config.categories:
        cat_dir = os.path.join(phasedata_dir, cat)
        if not os.path.exists(cat_dir):
            print(f"Warning: {cat_dir} not found, skip.")
            continue

        # --- try to load cached pickle ---
        pkl_path = os.path.join(cat_dir, 'all_data.pkl')
        if os.path.exists(pkl_path):
            with open(pkl_path, 'rb') as f:
                cat_data = pickle.load(f)
        else:
            # --- build dataset from json ---
            with open(os.path.join(cat_dir, 'cat.jsonl'), 'r') as f:
                lines = f.readlines()
            raw_dataset = []
            for line in lines:
                json_data = json.loads(line)
                raw_dataset.append({
                    'query': json_data['src'][-1],
                    'responses': json_data['response'],
                    'rank': json_data['rank']
                })
            cat_data = tokenize_data(tokenizer, raw_dataset)

            # cache for future fast loading
            with open(pkl_path, 'wb') as f:
                pickle.dump(cat_data, f)

        # --- merge into global ---
        all_data.extend(cat_data)

    # shuffle all data
    random.shuffle(all_data)

    return all_data

def load_data(phasedata_dir):
    """
    load data and tokenize it
    """

    if os.path.exists(os.path.join(phasedata_dir, 'phase_data.pkl')):
        with open(os.path.join(phasedata_dir, 'phase_data.pkl'), 'rb') as f:
            phase_data = pickle.load(f)
        return phase_data

    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path, trust_remote_code=True)
    with open(os.path.join(phasedata_dir, 'cat.jsonl'), 'r') as f:
        lines = f.readlines()
    raw_dataset = []
    for line in lines:
        json_data = json.loads(line)
        raw_dataset.append({
            'query': json_data['src'][-1],
            'responses': json_data['response'],
            'rank': json_data['rank']
        })
    tokenized_raw_dataset = tokenize_data(tokenizer, raw_dataset)
    
    with open(os.path.join(phasedata_dir, 'points.jsonl'), 'r') as f:
        lines = f.readlines()
    point_dataset = {}
    for line in lines:
        json_data = json.loads(line)
        point = json_data.pop('point')
        if point not in point_dataset.keys():
            point_dataset[point] = []
        point_dataset[point].append(json_data)
    tokenized_point_dataset = {}
    for key, value in point_dataset.items():
        tokenized_point_dataset[key] = tokenize_data(tokenizer, value)

    split = int(config.phase1_data_rate * len(tokenized_raw_dataset))
    phase_data = (tokenized_raw_dataset[:split], tokenized_point_dataset, tokenized_raw_dataset[split:])
    with open(os.path.join(phasedata_dir, 'phase_data.pkl'), 'wb') as f:  # save pickle file for future fast loading
        pickle.dump(phase_data, f)

    return phase_data


def tokenize_data(tokenizer, json_dataset):
    user = '<extra_0>'
    bot = '<extra_1>'
    end = '<extra_2>'

    tokenized_dataset = []
    print('begin tokenizing data: ')
    for json_data in tqdm(json_dataset):
        tokenized_data = {
            'input_ids': [],
            'attention_mask': [],
            'rank': json_data['rank']
        }
        for response in json_data['responses']:
            prompt = f"{user}{json_data['query']}{bot}{response}{end}"

            tokenized = tokenizer(prompt, return_tensors="pt").to(config.device)
            tokenized_data['input_ids'].append(tokenized['input_ids'])
            tokenized_data['attention_mask'].append(tokenized['attention_mask'])
        tokenized_dataset.append(tokenized_data)

    random.shuffle(tokenized_dataset)
    return tokenized_dataset

