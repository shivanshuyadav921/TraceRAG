"""
Data formatter for converting datasets into Chain-of-Thought (CoT) training format.
Formats prompts and completions for SFT and RL training.
"""

import logging
from typing import Dict, List, Optional

from datasets import Dataset

logger = logging.getLogger(__name__)


class DataFormatter:
    """Formats datasets into structured CoT format for training."""

    DEFAULT_SYSTEM_PROMPT = (
        "You are a helpful assistant that solves problems step by step. "
        "Show your reasoning in <think> tags and give your final answer in <answer> tags."
    )

    def __init__(
        self,
        system_prompt: Optional[str] = None,
        think_open: str = "<think>",
        think_close: str = "</think>",
        answer_open: str = "<answer>",
        answer_close: str = "</answer>",
    ):
        """
        Initialize the formatter.

        Args:
            system_prompt: System instruction for the model.
            think_open: Opening tag for reasoning section.
            think_close: Closing tag for reasoning section.
            answer_open: Opening tag for answer section.
            answer_close: Closing tag for answer section.
        """
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        self.think_open = think_open
        self.think_close = think_close
        self.answer_open = answer_open
        self.answer_close = answer_close

    def format_for_sft(self, dataset: Dataset, tokenizer=None) -> Dataset:
        """
        Format dataset for Supervised Fine-Tuning.
        
        Creates chat-style messages with system, user, and assistant roles.
        The assistant response includes structured CoT reasoning.

        Args:
            dataset: Normalized dataset with question, answer, rationale fields.
            tokenizer: Optional tokenizer for applying chat template.

        Returns:
            Dataset with 'text' or 'messages' field ready for SFT.
        """

        def format_fn(example):
            question = example["question"]
            answer = example["answer"]
            rationale = example.get("rationale", "")

            # Build the reasoning section
            if rationale:
                reasoning = self._format_rationale(rationale)
            else:
                reasoning = f"Let me think about this step by step.\nThe answer is {answer}."

            # Build the full assistant response
            assistant_response = (
                f"{self.think_open}\n"
                f"{reasoning}\n"
                f"{self.think_close}\n"
                f"{self.answer_open}{answer}{self.answer_close}"
            )

            # Build messages in chat format
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": question},
                {"role": "assistant", "content": assistant_response},
            ]

            result = {"messages": messages}

            # If tokenizer provided, also create the full text
            if tokenizer is not None:
                try:
                    text = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=False
                    )
                    result["text"] = text
                except Exception:
                    # Fallback: simple concatenation
                    result["text"] = self._simple_format(
                        question, assistant_response
                    )
            else:
                result["text"] = self._simple_format(
                    question, assistant_response
                )

            return result

        formatted = dataset.map(format_fn, desc="Formatting for SFT")
        logger.info(f"Formatted {len(formatted)} samples for SFT")
        return formatted

    def format_for_rl(self, dataset: Dataset, tokenizer=None) -> Dataset:
        """
        Format dataset for RL training (prompts only, no completions).

        For GRPO, we only need the prompts. The model generates completions
        during training, which are then scored by the reward function.

        Args:
            dataset: Normalized dataset with question and answer fields.
            tokenizer: Optional tokenizer for applying chat template.

        Returns:
            Dataset with 'prompt' and 'ground_truth' fields.
        """

        def format_fn(example):
            question = example["question"]
            answer = example["answer"]

            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": question},
            ]

            if tokenizer is not None:
                try:
                    prompt = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                except Exception:
                    prompt = self._simple_prompt(question)
            else:
                prompt = self._simple_prompt(question)

            return {
                "prompt": prompt,
                "ground_truth": answer,
                "source": example.get("source", "unknown"),
            }

        formatted = dataset.map(
            format_fn,
            remove_columns=dataset.column_names,
            desc="Formatting for RL",
        )
        logger.info(f"Formatted {len(formatted)} samples for RL")
        return formatted

    def format_for_eval(
        self,
        dataset: Dataset,
        num_fewshot: int = 0,
        fewshot_examples: Optional[List[Dict]] = None,
    ) -> Dataset:
        """
        Format dataset for evaluation with optional few-shot examples.

        Args:
            dataset: Normalized dataset.
            num_fewshot: Number of few-shot examples to prepend.
            fewshot_examples: Pre-selected few-shot examples.

        Returns:
            Dataset with evaluation prompts.
        """

        def format_fn(example):
            question = example["question"]
            answer = example["answer"]

            # Build few-shot prefix
            prefix = ""
            if num_fewshot > 0 and fewshot_examples:
                for fs in fewshot_examples[:num_fewshot]:
                    prefix += (
                        f"Question: {fs['question']}\n"
                        f"{self.think_open}\n{fs.get('rationale', 'Let me solve this.')}\n{self.think_close}\n"
                        f"{self.answer_open}{fs['answer']}{self.answer_close}\n\n"
                    )

            prompt = f"{prefix}Question: {question}\n"

            return {
                "prompt": prompt,
                "ground_truth": answer,
                "source": example.get("source", "unknown"),
            }

        formatted = dataset.map(
            format_fn,
            remove_columns=dataset.column_names,
            desc="Formatting for evaluation",
        )
        return formatted

    def _format_rationale(self, rationale: str) -> str:
        """Format rationale into numbered steps."""
        lines = [line.strip() for line in rationale.split("\n") if line.strip()]
        
        if len(lines) <= 1:
            # Try splitting by sentences
            import re
            sentences = re.split(r'(?<=[.!?])\s+', rationale.strip())
            lines = [s.strip() for s in sentences if s.strip()]

        # Number the steps
        formatted_steps = []
        for i, line in enumerate(lines, 1):
            # Don't re-number if already numbered
            if line[0].isdigit() and (line[1] == "." or line[1] == ")"):
                formatted_steps.append(line)
            else:
                formatted_steps.append(f"Step {i}: {line}")

        return "\n".join(formatted_steps)

    def _simple_format(self, question: str, response: str) -> str:
        """Simple text format without chat template."""
        return (
            f"### System:\n{self.system_prompt}\n\n"
            f"### User:\n{question}\n\n"
            f"### Assistant:\n{response}"
        )

    def _simple_prompt(self, question: str) -> str:
        """Simple prompt format without chat template."""
        return (
            f"### System:\n{self.system_prompt}\n\n"
            f"### User:\n{question}\n\n"
            f"### Assistant:\n"
        )
