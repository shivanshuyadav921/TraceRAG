"""
Curriculum Learning Scheduler.
Gradually increases problem difficulty during RL training.
"""

import logging
from typing import Dict, List, Optional

from datasets import Dataset

logger = logging.getLogger(__name__)


class CurriculumScheduler:
    """
    Manages curriculum learning for GRPO training.
    
    Starts with easy problems and gradually increases difficulty,
    allowing the model to build reasoning skills incrementally.
    """

    def __init__(self, config: Dict):
        """
        Initialize curriculum scheduler.

        Args:
            config: Curriculum configuration from grpo_config.yaml.
                Expected format:
                {
                    "enabled": true,
                    "difficulty_metric": "num_steps",
                    "stages": [
                        {"name": "easy", "epochs": [1, 2], "max_steps": 3},
                        {"name": "medium", "epochs": [3, 4], "max_steps": 6},
                        {"name": "hard", "epochs": [5], "max_steps": null}
                    ]
                }
        """
        self.enabled = config.get("enabled", True)
        self.difficulty_metric = config.get("difficulty_metric", "num_steps")
        self.stages = config.get("stages", [])
        self.current_stage_idx = 0

        if self.enabled:
            logger.info(
                f"Curriculum learning enabled with {len(self.stages)} stages: "
                f"{[s['name'] for s in self.stages]}"
            )

    def get_dataset_for_epoch(
        self, full_dataset: Dataset, epoch: int
    ) -> Dataset:
        """
        Get the appropriate dataset subset for the current epoch.

        Args:
            full_dataset: Full training dataset with difficulty annotations.
            epoch: Current training epoch (1-indexed).

        Returns:
            Filtered dataset appropriate for current difficulty level.
        """
        if not self.enabled:
            return full_dataset

        stage = self._get_stage_for_epoch(epoch)
        if stage is None:
            logger.info(f"Epoch {epoch}: No stage found, using full dataset")
            return full_dataset

        max_steps = stage.get("max_steps")
        stage_name = stage["name"]

        if max_steps is None:
            # No filter - use all data
            logger.info(
                f"Epoch {epoch}: Stage '{stage_name}' - using full dataset "
                f"({len(full_dataset)} samples)"
            )
            return full_dataset

        # Filter by difficulty
        filtered = full_dataset.filter(
            lambda x: x.get("num_steps", 1) <= max_steps,
            desc=f"Filtering for stage '{stage_name}'",
        )

        logger.info(
            f"Epoch {epoch}: Stage '{stage_name}' (max_steps={max_steps}) - "
            f"{len(filtered)}/{len(full_dataset)} samples"
        )

        return filtered

    def get_current_stage(self, epoch: int) -> Optional[Dict]:
        """Get the current curriculum stage info."""
        return self._get_stage_for_epoch(epoch)

    def _get_stage_for_epoch(self, epoch: int) -> Optional[Dict]:
        """Find which stage the given epoch belongs to."""
        for stage in self.stages:
            epochs = stage.get("epochs", [])
            if epoch in epochs:
                return stage
        
        # If epoch is beyond defined stages, use the last stage
        if self.stages and epoch > max(
            max(s.get("epochs", [0])) for s in self.stages
        ):
            return self.stages[-1]

        return None

    def get_schedule_summary(self) -> str:
        """Get a human-readable summary of the curriculum schedule."""
        if not self.enabled:
            return "Curriculum learning disabled - using full dataset for all epochs."

        lines = ["Curriculum Schedule:"]
        for stage in self.stages:
            epochs = stage.get("epochs", [])
            max_steps = stage.get("max_steps", "unlimited")
            lines.append(
                f"  - {stage['name']}: epochs {epochs}, max_steps={max_steps}"
            )
        return "\n".join(lines)
