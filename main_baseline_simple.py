"""
Main script for Simple Baseline Reward Model - 对比实验
只有backbone + value head，没有router和prefix
"""
import argparse
import os
import sys

from innermoe.modeling_baseline import BaselineRM
from innermoe.train_module_baseline import train_baseline_rm
import base_config as config


def train_baseline_only():
    """训练Baseline RM"""
    print("=" * 60)
    print("Baseline Reward Model - Training")
    print("=" * 60)
    
    print("\nInitializing Baseline RM...")
    baseline_rm = BaselineRM(config)
    
    print("\nStarting training...")
    train_baseline_rm(baseline_rm=baseline_rm)
    
    print("\n" + "=" * 60)
    print("Baseline RM training completed!")
    print("=" * 60)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Train Baseline Reward Model (backbone + value head only)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script trains a simple baseline reward model for comparison experiments.
It uses only a backbone model + value head, without router or prefix embeddings.

Example:
  python main_baseline_simple.py
        """
    )
    
    # DeepSpeed will automatically pass --local_rank
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank for distributed training (automatically set by DeepSpeed)"
    )
    
    args = parser.parse_args()
    
    # 执行训练
    train_baseline_only()


if __name__ == "__main__":
    main()

