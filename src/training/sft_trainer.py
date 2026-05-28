"""
SFT Trainer: Phase 1 of the training pipeline.
Fine-tunes the base model on Chain-of-Thought formatted data using LoRA.
"""

import logging
import os
from typing import Dict, Optional

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig

logger = logging.getLogger(__name__)


class SFTTrainerWrapper:
    """
    Wrapper around TRL's SFTTrainer for Phase 1 training.
    Teaches the model structured Chain-of-Thought reasoning format.
    """

    def __init__(self, config: Dict):
        """
        Initialize SFT trainer.

        Args:
            config: Full configuration dictionary (from sft_config.yaml).
        """
        self.config = config
        self.model_config = config["model"]
        self.lora_config = config["lora"]
        self.training_config = config["training"]
        
        self.model = None
        self.tokenizer = None
        self.trainer = None

    def setup(self):
        """Load model and tokenizer, apply LoRA."""
        logger.info(f"Loading model: {self.model_config['name']}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_config["name"],
            trust_remote_code=self.model_config.get("trust_remote_code", True),
        )
        
        # Set padding token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Load model
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(
            self.model_config.get("dtype", "bfloat16"), torch.bfloat16
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_config["name"],
            torch_dtype=torch_dtype,
            trust_remote_code=self.model_config.get("trust_remote_code", True),
            device_map="auto",
            attn_implementation="flash_attention_2" if self._has_flash_attn() else "eager",
        )

        # Apply LoRA if enabled
        if self.lora_config.get("enabled", True):
            self._apply_lora()

        logger.info(
            f"Model loaded. Trainable params: "
            f"{self._count_trainable_params():,} / {self._count_total_params():,} "
            f"({self._count_trainable_params() / self._count_total_params() * 100:.2f}%)"
        )

    def train(self, train_dataset: Dataset, eval_dataset: Optional[Dataset] = None):
        """
        Run SFT training.

        Args:
            train_dataset: Training dataset with 'text' or 'messages' field.
            eval_dataset: Optional evaluation dataset.
        """
        if self.model is None:
            self.setup()

        # Configure SFT training
        sft_config = SFTConfig(
            output_dir=self.training_config["output_dir"],
            num_train_epochs=self.training_config["num_epochs"],
            per_device_train_batch_size=self.training_config["per_device_train_batch_size"],
            per_device_eval_batch_size=self.training_config.get("per_device_eval_batch_size", 8),
            gradient_accumulation_steps=self.training_config["gradient_accumulation_steps"],
            learning_rate=self.training_config["learning_rate"],
            weight_decay=self.training_config.get("weight_decay", 0.01),
            warmup_ratio=self.training_config.get("warmup_ratio", 0.03),
            lr_scheduler_type=self.training_config.get("lr_scheduler_type", "cosine"),
            max_seq_length=self.training_config.get("max_seq_length", 1024),
            logging_steps=self.training_config.get("logging_steps", 10),
            save_steps=self.training_config.get("save_steps", 200),
            eval_steps=self.training_config.get("eval_steps", 200),
            eval_strategy="steps" if eval_dataset is not None else "no",
            save_total_limit=self.training_config.get("save_total_limit", 3),
            fp16=self.training_config.get("fp16", False),
            bf16=self.training_config.get("bf16", True),
            gradient_checkpointing=self.training_config.get("gradient_checkpointing", True),
            dataloader_num_workers=self.training_config.get("dataloader_num_workers", 4),
            report_to="wandb" if self.config.get("wandb", {}).get("enabled", False) else "none",
            run_name=self.config.get("wandb", {}).get("run_name", "sft-run"),
            dataset_text_field="text",
        )

        # Initialize trainer
        self.trainer = SFTTrainer(
            model=self.model,
            args=sft_config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
        )

        # Train
        logger.info("Starting SFT training...")
        train_result = self.trainer.train()

        # Save final model
        final_output_dir = os.path.join(self.training_config["output_dir"], "final")
        self.trainer.save_model(final_output_dir)
        self.tokenizer.save_pretrained(final_output_dir)
        
        logger.info(f"SFT training complete. Model saved to {final_output_dir}")
        logger.info(f"Training metrics: {train_result.metrics}")

        return train_result

    def _apply_lora(self):
        """Apply LoRA adapters to the model."""
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

    def _count_trainable_params(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def _count_total_params(self) -> int:
        """Count total parameters."""
        return sum(p.numel() for p in self.model.parameters())

    def _has_flash_attn(self) -> bool:
        """Check if Flash Attention is available."""
        try:
            import flash_attn  # noqa: F401
            return True
        except ImportError:
            return False
