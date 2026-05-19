#!/usr/bin/env python3
"""
数据预处理脚本：将每个prompt的多个response转换为每两个response一条数据
这样保证每个prompt都对应两个response，避免多卡训练时的问题
"""
import json
import os
from itertools import combinations
from tqdm import tqdm
import argparse


def extract_response_content(response_list):
    """从chosen/rejected列表中提取assistant的content"""
    for item in response_list:
        if item.get("role") == "assistant":
            return item.get("content", "")
    return ""


def convert_classified_to_train_format(input_file, output_file):
    """
    将标注后的数据转换为训练格式
    
    Args:
        input_file: 输入的标注json文件
        output_file: 输出的jsonl文件路径
    """
    # 读取标注数据
    with open(input_file, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    
    # 确保输出目录存在
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    converted_count = 0
    skipped_count = 0
    
    with open(output_file, "w", encoding="utf-8") as f_out:
        for item in tqdm(dataset, desc="Converting classified data"):
            # 检查必要字段
            if "prompt" not in item:
                skipped_count += 1
                continue
            
            if "chosen" not in item or "rejected" not in item:
                skipped_count += 1
                continue
            
            # 提取回答内容
            chosen_content = extract_response_content(item["chosen"])
            rejected_content = extract_response_content(item["rejected"])
            
            if not chosen_content or not rejected_content:
                skipped_count += 1
                continue
            
            # 提取认知级别标签
            label = item.get("label", "L1_recall")
            valid_labels = ["L1_recall", "L2_analysis", "L3_decision", "L4_synthesis"]
            if label not in valid_labels:
                # 尝试从cognitive_level中提取
                cognitive_level = item.get("cognitive_level", "")
                if "L4" in cognitive_level.upper():
                    label = "L4_synthesis"
                elif "L3" in cognitive_level.upper():
                    label = "L3_decision"
                elif "L2" in cognitive_level.upper():
                    label = "L2_analysis"
                elif "L1" in cognitive_level.upper():
                    label = "L1_recall"
                else:
                    label = "L1_recall"  # 默认值
            
            # 提取rank信息（从metadata中）
            chosen_rank = 1  # 默认chosen的rank更高
            rejected_rank = 2
            
            if "metadata" in item:
                metadata = item["metadata"]
                if "chosen" in metadata and "rank" in metadata["chosen"]:
                    chosen_rank = metadata["chosen"]["rank"]
                if "rejected" in metadata and "rank" in metadata["rejected"]:
                    rejected_rank = metadata["rejected"]["rank"]
            
            # 确保chosen的rank比rejected低（rank越小越好）
            if chosen_rank > rejected_rank:
                # 交换responses和ranks
                chosen_content, rejected_content = rejected_content, chosen_content
                chosen_rank, rejected_rank = rejected_rank, chosen_rank
            
            # 构建训练数据格式
            train_sample = {
                "src": [item["prompt"]],  # 对话历史，最后一个元素是query
                "response": [chosen_content, rejected_content],  # 多个回答
                "rank": [chosen_rank, rejected_rank],  # 对应的rank（rank越小越好）
                "label": label  # 认知级别
            }
            
            # 写入jsonl文件
            f_out.write(json.dumps(train_sample, ensure_ascii=False) + "\n")
            converted_count += 1
    
    print(f"\n数据转换完成！")
    print(f"  转换成功: {converted_count} 条")
    print(f"  跳过: {skipped_count} 条")
    print(f"  输出文件: {output_file}")


def preprocess_data(input_file, output_file):
    """
    将原始数据转换为每两个response一条数据的格式
    
    Args:
        input_file: 输入的jsonl文件路径
        output_file: 输出的jsonl文件路径
    """
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    total_samples = 0
    processed_samples = 0
    skipped_samples = 0
    
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        
        for line in tqdm(f_in, desc="Processing data"):
            try:
                j = json.loads(line.strip())
            except json.JSONDecodeError:
                skipped_samples += 1
                continue
            
            # 基本检查
            if "src" not in j or "response" not in j:
                skipped_samples += 1
                continue
            
            if "rank" not in j or j['rank'] is None:
                skipped_samples += 1
                continue
            
            query = j["src"][-1] if isinstance(j["src"], list) else j["src"]
            responses = j["response"]
            ranks = j["rank"]
            
            # 检查数据有效性
            if not isinstance(responses, list) or len(responses) < 2:
                skipped_samples += 1
                continue
            
            if not isinstance(ranks, list) or len(ranks) != len(responses):
                skipped_samples += 1
                continue
            
            total_samples += 1
            
            # 将responses和ranks配对
            response_rank_pairs = list(zip(responses, ranks))
            
            # 生成所有可能的response对（每两个response一条数据）
            # 使用combinations确保每个pair只出现一次
            # 只保留 ranks 不同的 pair（跳过 ranks 相同的无效数据）
            for (resp1, rank1), (resp2, rank2) in combinations(response_rank_pairs, 2):
                # 跳过 ranks 相同的数据（无效数据，无法计算 pairwise loss）
                if rank1 == rank2:
                    continue
                
                # 创建新的数据条目
                new_sample = {
                    "src": j["src"],  # 保留原始src
                    "response": [resp1, resp2],  # 只保留两个response
                    "rank": [rank1, rank2],  # 对应的rank
                }
                
                # 保留其他字段（如果有的话）
                if "label" in j:
                    new_sample["label"] = j["label"]
                
                # 写入新文件
                f_out.write(json.dumps(new_sample, ensure_ascii=False) + '\n')
                processed_samples += 1
    
    print(f"\n数据预处理完成！")
    print(f"  原始样本数: {total_samples}")
    print(f"  生成样本数: {processed_samples}")
    print(f"  跳过样本数: {skipped_samples}")
    print(f"  输出文件: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="将数据预处理为每两个response一条数据的格式"
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default="./prepare_data/Ernie-rlhf/Ernie-rlhf-train.jsonl",
        help="输入的jsonl文件路径"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="输出的jsonl文件路径（如果不指定，会根据输入文件名自动生成）"
    )
    parser.add_argument(
        "--process_test",
        action="store_true",
        help="同时处理测试集"
    )
    parser.add_argument(
        "--from_classified",
        action="store_true",
        help="从标注的json文件转换（而不是从jsonl文件）"
    )
    
    args = parser.parse_args()
    
    # 处理训练集
    if args.output_file is None:
        # 根据输入文件名自动生成输出文件名
        if "train" in args.input_file:
            args.output_file = args.input_file.replace("train.jsonl", "train-pairs.jsonl")
        else:
            args.output_file = args.input_file.replace(".jsonl", "-pairs.jsonl")
    
    # 确保输出目录存在
    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # 如果是从标注数据转换
    if args.from_classified:
        convert_classified_to_train_format(args.input_file, args.output_file)
    else:
        preprocess_data(args.input_file, args.output_file)
    
    # 如果指定了处理测试集
    if args.process_test:
        test_input = args.input_file.replace("train.jsonl", "test.jsonl")
        test_output = args.output_file.replace("train-pairs.jsonl", "test-pairs.jsonl")
        
        if os.path.exists(test_input):
            print(f"\n处理测试集...")
            preprocess_data(test_input, test_output)
        else:
            print(f"\n警告: 测试集文件不存在: {test_input}")


if __name__ == "__main__":
    main()

