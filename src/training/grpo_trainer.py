"""
GRPO Trainer: Phase 2 of the training pipeline.
Group Relative Policy Optimization for enhancing reasoning in SLMs.

GRPO eliminates the need for a separate reward/critic model by using
group-relative advantages computed from multiple sampled completions.
"""

import logging
import os
from typing import Dict, List, Optional, Callable

import torch
import torch.nn.functional as F
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from peft import LoraConfig, get_peft_model, PeftModel, TaskType
from trl import GRPOConfig, GRPOTrainer

from ..rewards.combined_reward import CombinedReward

logger = logging.getLogger(__name__)


class GRPOTrainerWrapper:
    """
    Wrapper around TRL's GRPOTrainer for Phase 2 RL training.
    
    Implements Group Relative Policy Optimization:
    1. For each prompt, generate G completions
    2. Score each with the reward function
    3. Compute group-relative advantages
    4. Update policy with clipped objective + KL penalty
    """

    def __init__(self, config: Dict):
        """
        Initialize GRPO trainer.

        Args:
            config: Full configuration dictionary (from grpo_config.yaml).
        """
        self.config = config
        self.model_config = config["model"]
        self.lora_config = config["lora"]
        self.grpo_config = config["grpo"]
        self.training_config = config["training"]
        self.reward_config = config["reward"]

        self.model = None
        self.tokenizer = None
        self.ref_model = None
        self.reward_fn = None
        self.trainer = None

    def setup(self):
        """Load model, tokenizer, reference model, and reward function."""
        logger.info("Setting up GRPO trainer...")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_config["name"],
            trust_remote_code=self.model_config.get("trust_remote_code", True),
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Determine dtype
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(
            self.model_config.get("dtype", "bfloat16"), torch.bfloat16
        )

        # Load the SFT checkpoint as starting point
        sft_checkpoint = self.model_config.get("sft_checkpoint")
        model_name = sft_checkpoint if sft_checkpoint and os.path.exists(sft_checkpoint) else self.model_config["name"]
        
        logger.info(f"Loading model from: {model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            trust_remote_code=self.model_config.get("trust_remote_code", True),
            device_map="auto",
        )

        # Apply LoRA if enabled
        if self.lora_config.get("enabled", True):
            self._apply_lora()

        # Initialize reward function
        self.reward_fn = CombinedReward.from_config(self.reward_config)
        logger.info("GRPO setup complete.")

    def _build_reward_function(self) -> Callable:
        """
        Build the reward function compatible with TRL's GRPOTrainer.
        
        TRL's GRPOTrainer expects a reward function that takes:
        - completions: list of strings (model outputs)
        And returns:
        - rewards: list of floats
        
        We also need access to the ground truth, which we pass via the dataset.
        """
        reward_fn = self.reward_fn

        def compute_rewards(completions: List[str], **kwargs) -> List[float]:
            """Compute rewards for a batch of completions."""
            # Ground truths are passed via prompts metadata
            ground_truths = kwargs.get("ground_truth", [""] * len(completions))
            
            rewards = []
            for completion, gt in zip(completions, ground_truths):
                reward = reward_fn.compute(completion, gt)
                rewards.append(reward)
            
            return rewards

        return compute_rewards

    def train(self, train_dataset: Dataset, eval_dataset: Optional[Dataset] = None):
        """
        Run GRPO training.

        Args:
            train_dataset: Dataset with 'prompt' and 'ground_truth' fields.
            eval_dataset: Optional evaluation dataset.
        """
        if self.model is None:
            self.setup()

        # Build reward function
        reward_function = self._build_reward_function()

        # Configure GRPO training
        grpo_training_config = GRPOConfig(
            output_dir=self.training_config["output_dir"],
            num_train_epochs=self.training_config["num_epochs"],
            per_device_train_batch_size=self.training_config["per_device_train_batch_size"],
            gradient_accumulation_steps=self.training_config["gradient_accumulation_steps"],
            learning_rate=self.training_config["learning_rate"],
            weight_decay=self.training_config.get("weight_decay", 0.01),
            warmup_steps=self.training_config.get("warmup_steps", 100),
            lr_scheduler_type=self.training_config.get("lr_scheduler_type", "cosine"),
            max_grad_norm=self.training_config.get("max_grad_norm", 1.0),
            logging_steps=self.training_config.get("logging_steps", 5),
            save_steps=self.training_config.get("save_steps", 100),
            save_total_limit=self.training_config.get("save_total_limit", 5),
            bf16=self.training_config.get("bf16", True),
            gradient_checkpointing=self.training_config.get("gradient_checkpointing", True),
            report_to="wandb" if self.config.get("wandb", {}).get("enabled", False) else "none",
            run_name=self.config.get("wandb", {}).get("run_name", "grpo-run"),
            # GRPO specific
            num_generations=self.grpo_config.get("group_size", 8),
            max_completion_length=self.grpo_config.get("max_new_tokens", 512),
            temperature=self.grpo_config.get("temperature", 0.7),
            beta=self.grpo_config.get("kl_coeff", 0.04),
        )

        # Initialize GRPO trainer
        self.trainer = GRPOTrainer(
            model=self.model,
            args=grpo_training_config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=self.tokenizer,
            reward_funcs=reward_function,
        )

        # Train
        logger.info("Starting GRPO training...")
        logger.info(
            f"  Group size: {self.grpo_config.get('group_size', 8)}"
        )
        logger.info(f"  KL coeff: {self.grpo_config.get('kl_coeff', 0.04)}")
        logger.info(f"  Temperature: {self.grpo_config.get('temperature', 0.7)}")

        train_result = self.trainer.train()

        # Save final model
        final_output_dir = os.path.join(self.training_config["output_dir"], "final")
        self.trainer.save_model(final_output_dir)
        self.tokenizer.save_pretrained(final_output_dir)

        logger.info(f"GRPO training complete. Model saved to {final_output_dir}")
        logger.info(f"Training metrics: {train_result.metrics}")

        return train_result

    def _apply_lora(self):
        """Apply LoRA adapters to the model."""
        # Check if model already has LoRA (from SFT checkpoint)
        if isinstance(self.model, PeftModel):
            logger.info("Model already has LoRA adapters (from SFT checkpoint)")
            return

        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.lora_config["rank"],
            lora_alpha=self.lora_config["alpha"],
            lora_dropout=self.lora_config.get("dropout", 0.05),
            target_modules=self.lora_config.get(
                "target_modules",
                ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            ),
            bias="none",
        )

        self.model = get_peft_model(self.model, lora_cfg)
        logger.info(f"LoRA applied: rank={lora_cfg.r}, alpha={lora_cfg.lora_alpha}")
