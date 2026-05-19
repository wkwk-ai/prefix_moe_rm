"""
Main training script with unified entry point.
Supports training router, reward model, or both sequentially.
"""
import argparse
import os
import sys

# Import training functions
from router.train import train_router
from innermoe.modeling_prefix import InnerMoERM
from innermoe.train_module_prefix import train_rm
import base_config as config
import router.config as router_config
import innermoe.config as innermoe_config

prefix_prompts = {
    "L1_recall": "You are a medical knowledge assessment expert. Please rate the answers to basic medical knowledge questions based on factual accuracy, completeness, conceptual understanding, and recall precision. Focus on whether the facts are correct and the explanation is clear.",

    "L2_analysis": "You are a clinical analysis assessment expert. Please rate clinical case analyses based on diagnostic accuracy, information integration ability, depth of analysis, clinical relevance, and logical reasoning. Focus on the ability to apply knowledge to clinical scenarios and integrate multiple findings.",

    "L3_decision": "You are a clinical decision-making assessment expert. Please rate clinical reasoning and management plans based on reasoning depth, treatment plan rationality, handling of uncertainty, risk-benefit analysis, and patient safety consideration. Focus on multi-step clinical reasoning and decision quality.",

    "L4_synthesis": "You are a biomedical synthesis and innovation assessment expert. Please rate research designs, cross-domain analyses, and critical evaluations based on analytical depth, methodological rigor, logical coherence, originality, and balanced consideration of multiple perspectives. Focus on synthesis quality and scholarly contribution."
}

def train_router_only():
    """Train only the router."""
    print("=" * 60)
    print("Training Router Only")
    print("=" * 60)
    train_router()
    print("\nRouter training completed!")


def train_rm_only(router_ckpt_path=None, rm_ckpt_path=None, train_data_path=None,
                   learning_rate=None, num_epochs=None):
    """Train only the reward model."""
    print("=" * 60)
    print("Training Reward Model Only")
    print("=" * 60)

    # Use provided router checkpoint or default
    if router_ckpt_path is None:
        router_ckpt_path = router_config.out_dir

    # Check if router checkpoint exists
    router_bin_path = os.path.join(router_ckpt_path, "router.bin")
    if not os.path.exists(router_bin_path):
        raise FileNotFoundError(
            f"Router checkpoint not found at {router_ckpt_path}\n"
            f"Please train router first using: python main_prefix.py --mode router"
        )

    print(f"Using router checkpoint: {router_ckpt_path}")
    if rm_ckpt_path:
        print(f"Fine-tuning from RM checkpoint: {rm_ckpt_path}")
    if train_data_path:
        print(f"Using custom training data: {train_data_path}")

    # Override config if custom values provided
    if learning_rate is not None:
        innermoe_config.learning_rate = learning_rate
    if num_epochs is not None:
        innermoe_config.num_epochs = num_epochs
    if train_data_path is not None:
        innermoe_config.train_data_path = train_data_path

    # Initialize and train InnerMoERM
    print("\nInitializing InnerMoERM...")
    innerMoERM = InnerMoERM(config, prefix_prompts, router_ckpt_path=router_ckpt_path,
                            ckpt_path=rm_ckpt_path)

    print("\nStarting training...")
    train_rm(innerMoERM=innerMoERM)

    print("\n" + "=" * 60)
    print("Reward Model training completed!")
    print("=" * 60)


def train_both():
    """Train router first, then reward model."""
    print("=" * 60)
    print("Training Router and Reward Model Sequentially")
    print("=" * 60)
    
    # Step 1: Train router
    print("\n[Step 1/2] Training Router...")
    print("-" * 60)
    train_router()
    
    # Step 2: Train reward model using the trained router
    print("\n[Step 2/2] Training Reward Model...")
    print("-" * 60)
    train_rm_only(router_ckpt_path=router_config.out_dir)
    
    print("\n" + "=" * 60)
    print("All training completed!")
    print("=" * 60)


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Train Router and/or Reward Model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train router only
  python main_prefix.py --mode router
  
  # Train reward model only (requires existing router checkpoint)
  python main_prefix.py --mode rm --router_ckpt ./results/router
  
  # Train both sequentially (default)
  python main_prefix.py --mode both
  
  # Train both with custom router checkpoint path
  python main_prefix.py --mode both --router_ckpt ./results/router
        """
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=["router", "rm", "both"],
        default="both",
        help="Training mode: 'router' (train router only), 'rm' (train reward model only), 'both' (train both sequentially)"
    )
    
    parser.add_argument(
        "--router_ckpt",
        type=str,
        default=None,
        help="Path to router checkpoint directory (required for 'rm' mode, optional for 'both' mode)"
    )

    parser.add_argument(
        "--rm_ckpt",
        type=str,
        default=None,
        help="Path to existing RM checkpoint for fine-tuning (optional, for 'rm' mode)"
    )

    parser.add_argument(
        "--train_data",
        type=str,
        default=None,
        help="Path to custom training data JSONL (overrides default Ernie-rlhf-train.jsonl)"
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="Override learning rate (e.g., 1e-5 for fine-tuning)"
    )

    parser.add_argument(
        "--num_epochs",
        type=int,
        default=None,
        help="Override number of training epochs"
    )
    
    # DeepSpeed will automatically pass --local_rank, we need to accept it
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank for distributed training (automatically set by DeepSpeed)"
    )
    
    args = parser.parse_args()
    
    # Execute based on mode
    if args.mode == "router":
        train_router_only()
    elif args.mode == "rm":
        router_ckpt = args.router_ckpt or router_config.out_dir
        train_rm_only(router_ckpt_path=router_ckpt, rm_ckpt_path=args.rm_ckpt,
                      train_data_path=args.train_data, learning_rate=args.learning_rate,
                      num_epochs=args.num_epochs)
    elif args.mode == "both":
        train_both()


if __name__ == "__main__":
    main()
