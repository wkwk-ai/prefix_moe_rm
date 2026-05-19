"""
Main script for Hard Prompt Reward Model - 对比实验
使用固定的硬prompt，不进行训练
"""
import argparse
import os
import sys

from innermoe.modeling_hardprompt import HardPromptRM
from innermoe.train_module_hardprompt import train_hardprompt_rm
import base_config as config
import router.config as router_config

# 硬prompt定义（固定，不训练）
hard_prompts = {
    "L1_recall": "You are a medical knowledge assessment expert. Please rate the answers to basic medical knowledge questions based on factual accuracy, completeness, conceptual understanding, and recall precision. Focus on whether the facts are correct and the explanation is clear.",

    "L2_analysis": "You are a clinical analysis assessment expert. Please rate clinical case analyses based on diagnostic accuracy, information integration ability, depth of analysis, clinical relevance, and logical reasoning. Focus on the ability to apply knowledge to clinical scenarios and integrate multiple findings.",

    "L3_decision": "You are a clinical decision-making assessment expert. Please rate clinical reasoning and management plans based on reasoning depth, treatment plan rationality, handling of uncertainty, risk-benefit analysis, and patient safety consideration. Focus on multi-step clinical reasoning and decision quality.",

    "L4_synthesis": "You are a biomedical synthesis and innovation assessment expert. Please rate research designs, cross-domain analyses, and critical evaluations based on analytical depth, methodological rigor, logical coherence, originality, and balanced consideration of multiple perspectives. Focus on synthesis quality and scholarly contribution."
}

def train_hardprompt_only(router_ckpt_path=None):
    """训练Hard Prompt RM（使用固定的硬prompt，但backbone和value head可训练）"""
    print("=" * 60)
    print("Hard Prompt Reward Model - Training")
    print("=" * 60)

    if router_ckpt_path is None:
        router_ckpt_path = router_config.out_dir

    router_bin_path = os.path.join(router_ckpt_path, "router.bin")
    if not os.path.exists(router_bin_path):
        raise FileNotFoundError(
            f"Router checkpoint not found at {router_ckpt_path}\n"
            f"Please train router first using: python main_prefix.py --mode router"
        )

    print(f"Using router checkpoint: {router_ckpt_path}")

    print("\nInitializing Hard Prompt RM...")
    hardprompt_rm = HardPromptRM(config, hard_prompts=hard_prompts, router_ckpt_path=router_ckpt_path)
    
    print("\nStarting training...")
    train_hardprompt_rm(hardprompt_rm)
    
    print("\n" + "=" * 60)
    print("Hard Prompt RM training completed!")
    print("=" * 60)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Train Hard Prompt Reward Model (fixed prompts, trainable backbone+head)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script trains a Hard Prompt Reward Model for comparison experiments.
It uses fixed hard prompts (not trainable), but backbone and value head are trainable.

Example:
  python main_hardprompt.py
        """
    )
    
    # DeepSpeed will automatically pass --local_rank
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank for distributed training (automatically set by DeepSpeed)"
    )

    parser.add_argument(
        "--router_ckpt",
        type=str,
        default=None,
        help="Path to router checkpoint directory"
    )

    args = parser.parse_args()

    # 执行训练
    train_hardprompt_only(router_ckpt_path=args.router_ckpt)


if __name__ == "__main__":
    main()

