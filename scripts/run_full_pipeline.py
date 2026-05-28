"""
Full Training Pipeline: End-to-end SFT → GRPO → Evaluation
============================================================

Usage:
    python scripts/run_full_pipeline.py --config-dir configs/

This script runs the complete training pipeline:
1. Phase 1: SFT on CoT-formatted data
2. Phase 2: GRPO reinforcement learning
3. Evaluation on GSM8K, MMLU, StrategyQA
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import yaml

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.dataset_loader import DatasetLoader
from src.data.data_formatter import DataFormatter
from src.training.sft_trainer import SFTTrainerWrapper
from src.training.grpo_trainer import GRPOTrainerWrapper
from src.training.curriculum import CurriculumScheduler
from src.evaluation.evaluator import Evaluator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("training.log"),
    ],
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def run_sft_phase(config_dir: str) -> str:
    """
    Phase 1: Supervised Fine-Tuning.
    
    Returns:
        Path to the saved SFT model.
    """
    logger.info("=" * 60)
    logger.info("PHASE 1: Supervised Fine-Tuning (SFT)")
    logger.info("=" * 60)

    config = load_config(os.path.join(config_dir, "sft_config.yaml"))

    # Load and prepare data
    loader = DatasetLoader()
    formatter = DataFormatter(
        system_prompt=config.get("formatting", {}).get("system_prompt"),
    )

    # Load mixed dataset
    dataset = loader.load_mixed(
        config["data"]["datasets"],
        split="train",
        max_samples=config["data"].get("max_samples"),
    )

    # Format for SFT
    formatted = formatter.format_for_sft(dataset)

    # Split train/eval
    val_split = config["data"].get("validation_split", 0.05)
    split = formatted.train_test_split(test_size=val_split, seed=42)
    train_dataset = split["train"]
    eval_dataset = split["test"]

    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(eval_dataset)}")

    # Train
    trainer = SFTTrainerWrapper(config)
    trainer.setup()
    trainer.train(train_dataset, eval_dataset)

    output_dir = os.path.join(config["training"]["output_dir"], "final")
    logger.info(f"SFT Phase complete. Model saved to: {output_dir}")
    return output_dir


def run_grpo_phase(config_dir: str, sft_model_path: str) -> str:
    """
    Phase 2: GRPO Reinforcement Learning.
    
    Args:
        sft_model_path: Path to the SFT model checkpoint.
    
    Returns:
        Path to the saved GRPO model.
    """
    logger.info("=" * 60)
    logger.info("PHASE 2: Group Relative Policy Optimization (GRPO)")
    logger.info("=" * 60)

    config = load_config(os.path.join(config_dir, "grpo_config.yaml"))
    
    # Override SFT checkpoint path
    config["model"]["sft_checkpoint"] = sft_model_path

    # Load and prepare data
    loader = DatasetLoader()
    formatter = DataFormatter(
        system_prompt=config.get("formatting", {}).get("system_prompt"),
    )

    # Load dataset
    dataset = loader.load_mixed(
        config["data"]["datasets"],
        split="train",
        max_samples=config["data"].get("max_samples"),
    )

    # Format for RL (prompts only)
    formatted = formatter.format_for_rl(dataset)

    logger.info(f"RL training samples: {len(formatted)}")

    # Setup curriculum if enabled
    curriculum_config = config.get("curriculum", {})
    if curriculum_config.get("enabled", False):
        scheduler = CurriculumScheduler(curriculum_config)
        logger.info(scheduler.get_schedule_summary())

    # Train
    trainer = GRPOTrainerWrapper(config)
    trainer.setup()
    trainer.train(formatted)

    output_dir = os.path.join(config["training"]["output_dir"], "final")
    logger.info(f"GRPO Phase complete. Model saved to: {output_dir}")
    return output_dir


def run_evaluation(config_dir: str, model_path: str):
    """
    Phase 3: Evaluation on benchmarks.
    
    Args:
        model_path: Path to the model to evaluate.
    """
    logger.info("=" * 60)
    logger.info("PHASE 3: Evaluation")
    logger.info("=" * 60)

    config = load_config(os.path.join(config_dir, "eval_config.yaml"))
    config["model"]["checkpoint"] = model_path

    # Load evaluation datasets
    loader = DatasetLoader()
    formatter = DataFormatter()

    eval_datasets = {}
    for benchmark in config["evaluation"]["benchmarks"]:
        name = benchmark["name"]
        try:
            dataset = loader.load(name, split=benchmark.get("split", "test"))
            normalized = loader._normalize_schema(dataset, name)
            formatted = formatter.format_for_eval(
                normalized,
                num_fewshot=benchmark.get("num_fewshot", 0),
            )
            eval_datasets[name] = formatted
            logger.info(f"Loaded eval dataset: {name} ({len(formatted)} samples)")
        except Exception as e:
            logger.warning(f"Failed to load {name}: {e}")

    # Run evaluation
    evaluator = Evaluator(config)
    evaluator.setup(model_path=model_path)
    results = evaluator.evaluate_all(eval_datasets)

    # Print final results
    logger.info("\n" + "=" * 60)
    logger.info("FINAL RESULTS")
    logger.info("=" * 60)
    for name, result in results.items():
        if name == "latency":
            logger.info(f"  Latency: avg={result['avg_latency_s']:.3f}s")
        else:
            logger.info(f"  {name}: {result['accuracy']*100:.2f}%")
    logger.info("=" * 60)

    # Compare with baselines if configured
    if config.get("baselines", {}).get("run_baseline", False):
        logger.info("\nRunning baseline comparisons...")
        evaluator.compare_models(
            config["baselines"]["models"],
            eval_datasets,
        )

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run the full SLM reasoning enhancement pipeline"
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        default="configs",
        help="Directory containing configuration files",
    )
    parser.add_argument(
        "--phase",
        type=str,
        choices=["all", "sft", "grpo", "eval"],
        default="all",
        help="Which phase to run (default: all)",
    )
    parser.add_argument(
        "--sft-model",
        type=str,
        default=None,
        help="Path to existing SFT model (skip Phase 1)",
    )
    parser.add_argument(
        "--grpo-model",
        type=str,
        default=None,
        help="Path to existing GRPO model (skip Phase 1 & 2)",
    )
    args = parser.parse_args()

    logger.info("Starting SLM Reasoning Enhancement Pipeline")
    logger.info(f"Config directory: {args.config_dir}")
    logger.info(f"Phase: {args.phase}")

    if args.phase == "all":
        # Run full pipeline
        sft_path = args.sft_model or run_sft_phase(args.config_dir)
        grpo_path = args.grpo_model or run_grpo_phase(args.config_dir, sft_path)
        run_evaluation(args.config_dir, grpo_path)

    elif args.phase == "sft":
        run_sft_phase(args.config_dir)

    elif args.phase == "grpo":
        sft_path = args.sft_model or "./outputs/sft/final"
        if not os.path.exists(sft_path):
            logger.error(f"SFT model not found at {sft_path}. Run SFT first or provide --sft-model")
            sys.exit(1)
        run_grpo_phase(args.config_dir, sft_path)

    elif args.phase == "eval":
        model_path = args.grpo_model or "./outputs/grpo/final"
        if not os.path.exists(model_path):
            logger.error(f"Model not found at {model_path}. Provide --grpo-model")
            sys.exit(1)
        run_evaluation(args.config_dir, model_path)

    logger.info("Pipeline complete!")


if __name__ == "__main__":
    main()
