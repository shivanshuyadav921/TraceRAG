"""
Dataset loader for GSM8K, AQuA-RAT, and MMLU datasets.
Handles downloading, caching, and splitting.
"""

import logging
from typing import Dict, List, Optional, Union

from datasets import load_dataset, Dataset, DatasetDict, concatenate_datasets

logger = logging.getLogger(__name__)


class DatasetLoader:
    """Loads and manages datasets for SLM reasoning training."""

    SUPPORTED_DATASETS = {
        "gsm8k": {
            "hf_name": "openai/gsm8k",
            "subset": "main",
            "question_field": "question",
            "answer_field": "answer",
        },
        "aqua_rat": {
            "hf_name": "deepmind/aqua_rat",
            "subset": "raw",
            "question_field": "question",
            "answer_field": "correct",
            "options_field": "options",
            "rationale_field": "rationale",
        },
        "mmlu": {
            "hf_name": "cais/mmlu",
            "subset": "all",
            "question_field": "question",
            "answer_field": "answer",
            "choices_field": "choices",
        },
        "strategyqa": {
            "hf_name": "wics/strategy-qa",
            "subset": None,
            "question_field": "question",
            "answer_field": "answer",
        },
    }

    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize the dataset loader.

        Args:
            cache_dir: Directory to cache downloaded datasets.
        """
        self.cache_dir = cache_dir
        self._loaded_datasets: Dict[str, Dataset] = {}

    def load(
        self,
        dataset_name: str,
        split: str = "train",
        max_samples: Optional[int] = None,
    ) -> Dataset:
        """
        Load a dataset by name.

        Args:
            dataset_name: Name of the dataset (gsm8k, aqua_rat, mmlu, strategyqa)
            split: Dataset split to load (train, test, validation)
            max_samples: Maximum number of samples to load (None for all)

        Returns:
            HuggingFace Dataset object.
        """
        if dataset_name not in self.SUPPORTED_DATASETS:
            raise ValueError(
                f"Unsupported dataset: {dataset_name}. "
                f"Supported: {list(self.SUPPORTED_DATASETS.keys())}"
            )

        config = self.SUPPORTED_DATASETS[dataset_name]
        cache_key = f"{dataset_name}_{split}"

        if cache_key in self._loaded_datasets:
            logger.info(f"Using cached dataset: {cache_key}")
            dataset = self._loaded_datasets[cache_key]
        else:
            logger.info(f"Loading dataset: {dataset_name} (split={split})")
            dataset = load_dataset(
                config["hf_name"],
                config["subset"],
                split=split,
                cache_dir=self.cache_dir,
            )
            self._loaded_datasets[cache_key] = dataset

        if max_samples is not None and len(dataset) > max_samples:
            dataset = dataset.select(range(max_samples))
            logger.info(f"Trimmed dataset to {max_samples} samples")

        logger.info(f"Loaded {dataset_name}: {len(dataset)} samples")
        return dataset

    def load_mixed(
        self,
        dataset_configs: List[Dict],
        split: str = "train",
        max_samples: Optional[int] = None,
    ) -> Dataset:
        """
        Load and mix multiple datasets with specified weights.

        Args:
            dataset_configs: List of dicts with 'name' and 'weight' keys.
                Example: [{"name": "gsm8k", "weight": 0.6}, {"name": "aqua_rat", "weight": 0.4}]
            split: Dataset split to load.
            max_samples: Total maximum samples after mixing.

        Returns:
            Combined and shuffled Dataset.
        """
        datasets = []
        total_weight = sum(cfg["weight"] for cfg in dataset_configs)

        for cfg in dataset_configs:
            dataset = self.load(cfg["name"], split=split)
            weight = cfg["weight"] / total_weight

            if max_samples is not None:
                n_samples = int(max_samples * weight)
            else:
                n_samples = int(len(dataset) * weight)

            n_samples = min(n_samples, len(dataset))
            dataset = dataset.shuffle(seed=42).select(range(n_samples))
            datasets.append(dataset)

        # We can't directly concatenate datasets with different schemas,
        # so we normalize them first
        normalized = [self._normalize_schema(ds, cfg["name"]) 
                      for ds, cfg in zip(datasets, dataset_configs)]
        
        combined = concatenate_datasets(normalized)
        combined = combined.shuffle(seed=42)

        logger.info(f"Mixed dataset: {len(combined)} total samples")
        return combined

    def _normalize_schema(self, dataset: Dataset, dataset_name: str) -> Dataset:
        """
        Normalize dataset to a common schema: {question, answer, rationale, source}.
        """
        config = self.SUPPORTED_DATASETS[dataset_name]

        def normalize_fn(example):
            question = example[config["question_field"]]
            
            if dataset_name == "gsm8k":
                # GSM8K answer field contains rationale + #### answer
                full_answer = example[config["answer_field"]]
                parts = full_answer.split("####")
                rationale = parts[0].strip() if len(parts) > 1 else ""
                answer = parts[-1].strip()
            elif dataset_name == "aqua_rat":
                answer = example[config["answer_field"]]
                rationale = example.get(config.get("rationale_field", ""), "")
                # Include options in question
                options = example.get(config.get("options_field", ""), [])
                if options:
                    options_str = "\n".join(options)
                    question = f"{question}\n\nOptions:\n{options_str}"
            elif dataset_name == "mmlu":
                choices = example.get(config.get("choices_field", ""), [])
                answer_idx = example[config["answer_field"]]
                answer = choices[answer_idx] if isinstance(answer_idx, int) and choices else str(answer_idx)
                rationale = ""
                if choices:
                    labels = ["A", "B", "C", "D"]
                    options_str = "\n".join(
                        f"{labels[i]}. {c}" for i, c in enumerate(choices)
                    )
                    question = f"{question}\n\n{options_str}"
            elif dataset_name == "strategyqa":
                answer = str(example[config["answer_field"]])
                rationale = ""
            else:
                answer = str(example[config["answer_field"]])
                rationale = ""

            return {
                "question": question,
                "answer": answer,
                "rationale": rationale,
                "source": dataset_name,
            }

        return dataset.map(
            normalize_fn,
            remove_columns=dataset.column_names,
            desc=f"Normalizing {dataset_name}",
        )

    def get_difficulty_splits(
        self, dataset: Dataset, easy_max: int = 3, medium_max: int = 6
    ) -> Dict[str, Dataset]:
        """
        Split dataset by difficulty based on rationale length (number of steps).

        Args:
            dataset: Normalized dataset with 'rationale' field.
            easy_max: Max steps for easy problems.
            medium_max: Max steps for medium problems.

        Returns:
            Dict with 'easy', 'medium', 'hard' datasets.
        """

        def count_steps(example):
            rationale = example.get("rationale", "")
            # Count steps by newlines or sentence endings
            steps = len([s for s in rationale.split("\n") if s.strip()])
            if steps == 0:
                steps = len([s for s in rationale.split(".") if s.strip()])
            example["num_steps"] = max(steps, 1)
            return example

        dataset = dataset.map(count_steps, desc="Counting steps")

        easy = dataset.filter(lambda x: x["num_steps"] <= easy_max)
        medium = dataset.filter(
            lambda x: easy_max < x["num_steps"] <= medium_max
        )
        hard = dataset.filter(lambda x: x["num_steps"] > medium_max)

        logger.info(
            f"Difficulty splits: easy={len(easy)}, medium={len(medium)}, hard={len(hard)}"
        )

        return {"easy": easy, "medium": medium, "hard": hard}
