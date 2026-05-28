"""
Combined Reward: Weighted combination of outcome and process rewards.
Used as the main reward signal for GRPO training.
"""

import logging
from typing import Dict, List

from .outcome_reward import OutcomeReward
from .process_reward import ProcessReward

logger = logging.getLogger(__name__)


class CombinedReward:
    """
    Combined reward function that merges outcome (correctness) and process
    (reasoning quality) rewards.

    R_total = outcome_weight * R_outcome + process_weight * R_process
    """

    def __init__(
        self,
        outcome_weight: float = 0.7,
        process_weight: float = 0.3,
        outcome_config: Dict = None,
        process_config: Dict = None,
    ):
        """
        Initialize combined reward.

        Args:
            outcome_weight: Weight for outcome (correctness) reward.
            process_weight: Weight for process (reasoning) reward.
            outcome_config: Config dict for OutcomeReward.
            process_config: Config dict for ProcessReward.
        """
        self.outcome_weight = outcome_weight
        self.process_weight = process_weight

        # Initialize sub-rewards
        self.outcome_reward = OutcomeReward(**(outcome_config or {}))
        self.process_reward = ProcessReward(**(process_config or {}))

        logger.info(
            f"CombinedReward initialized: "
            f"outcome_weight={outcome_weight}, process_weight={process_weight}"
        )

    def compute(self, response: str, ground_truth: str) -> float:
        """
        Compute combined reward for a single response.

        Args:
            response: Full model response (reasoning + answer).
            ground_truth: The correct answer.

        Returns:
            Combined reward score.
        """
        r_outcome = self.outcome_reward.compute(response, ground_truth)
        r_process = self.process_reward.compute(response)

        combined = (
            self.outcome_weight * r_outcome + self.process_weight * r_process
        )
        return combined

    def compute_batch(
        self, responses: List[str], ground_truths: List[str]
    ) -> List[float]:
        """
        Compute combined rewards for a batch of responses.

        Args:
            responses: List of model responses.
            ground_truths: List of correct answers.

        Returns:
            List of combined reward scores.
        """
        return [
            self.compute(resp, truth)
            for resp, truth in zip(responses, ground_truths)
        ]

    def compute_detailed(
        self, response: str, ground_truth: str
    ) -> Dict[str, float]:
        """
        Compute detailed breakdown of all reward components.

        Args:
            response: Full model response.
            ground_truth: The correct answer.

        Returns:
            Dict with detailed scores.
        """
        r_outcome = self.outcome_reward.compute(response, ground_truth)
        process_details = self.process_reward.compute_detailed(response)
        r_process = process_details["total"]

        combined = (
            self.outcome_weight * r_outcome + self.process_weight * r_process
        )

        return {
            "combined_reward": combined,
            "outcome_reward": r_outcome,
            "outcome_weighted": self.outcome_weight * r_outcome,
            "process_reward": r_process,
            "process_weighted": self.process_weight * r_process,
            "process_details": process_details,
            "extracted_answer": self.outcome_reward.extract_answer(response),
            "ground_truth": ground_truth,
            "is_correct": r_outcome >= self.outcome_reward.exact_match_score,
        }

    def compute_for_grpo(
        self, responses: List[str], ground_truth: str
    ) -> List[float]:
        """
        Compute rewards for a group of responses to the same prompt.
        Used in GRPO where multiple completions are generated per prompt.

        Args:
            responses: List of G completions for one prompt.
            ground_truth: The correct answer for this prompt.

        Returns:
            List of rewards for each completion.
        """
        rewards = [self.compute(resp, ground_truth) for resp in responses]
        return rewards

    @classmethod
    def from_config(cls, config: Dict) -> "CombinedReward":
        """
        Create CombinedReward from a config dictionary.

        Args:
            config: Dictionary with reward configuration.

        Returns:
            CombinedReward instance.
        """
        outcome_config = config.get("outcome", {})
        process_config = config.get("process", {})

        return cls(
            outcome_weight=config.get("outcome_weight", 0.7),
            process_weight=config.get("process_weight", 0.3),
            outcome_config=outcome_config,
            process_config=process_config,
        )
