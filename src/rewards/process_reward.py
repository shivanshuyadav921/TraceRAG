"""
Process Reward: Evaluates the quality of reasoning steps.
Rewards structured thinking, multi-step reasoning, and format compliance.
"""

import re
import logging
from typing import Optional
from collections import Counter

logger = logging.getLogger(__name__)


class ProcessReward:
    """
    Reward based on the quality of the reasoning process.
    
    Evaluates:
    - Format compliance (use of think/answer tags)
    - Multi-step reasoning (number of distinct steps)
    - No repetition (penalizes repeated phrases)
    - Conciseness (length penalty for overly verbose output)
    """

    def __init__(
        self,
        think_tag_score: float = 0.2,
        step_score: float = 0.05,
        max_step_score: float = 0.3,
        no_repetition_score: float = 0.2,
        answer_tag_score: float = 0.1,
        length_penalty_threshold: int = 1000,
        length_penalty_rate: float = 0.0001,
        max_process_score: float = 0.5,
        think_open_tag: str = "<think>",
        think_close_tag: str = "</think>",
        answer_open_tag: str = "<answer>",
        answer_close_tag: str = "</answer>",
    ):
        """
        Initialize process reward.

        Args:
            think_tag_score: Score for using think tags.
            step_score: Score per reasoning step.
            max_step_score: Maximum score from steps.
            no_repetition_score: Score for not repeating content.
            answer_tag_score: Score for having answer tags.
            length_penalty_threshold: Character count before penalty applies.
            length_penalty_rate: Penalty rate per character over threshold.
            max_process_score: Maximum total process reward.
            think_open_tag: Opening tag for thinking section.
            think_close_tag: Closing tag for thinking section.
            answer_open_tag: Opening tag for answer section.
            answer_close_tag: Closing tag for answer section.
        """
        self.think_tag_score = think_tag_score
        self.step_score = step_score
        self.max_step_score = max_step_score
        self.no_repetition_score = no_repetition_score
        self.answer_tag_score = answer_tag_score
        self.length_penalty_threshold = length_penalty_threshold
        self.length_penalty_rate = length_penalty_rate
        self.max_process_score = max_process_score
        self.think_open_tag = think_open_tag
        self.think_close_tag = think_close_tag
        self.answer_open_tag = answer_open_tag
        self.answer_close_tag = answer_close_tag

    def compute(self, response: str) -> float:
        """
        Compute process reward for a single response.

        Args:
            response: Full model response.

        Returns:
            Float reward score (0 to max_process_score).
        """
        score = 0.0

        # 1. Format compliance - Think tags
        if self._has_think_tags(response):
            score += self.think_tag_score

        # 2. Step count - Reward multi-step reasoning
        num_steps = self._count_steps(response)
        step_reward = min(num_steps * self.step_score, self.max_step_score)
        score += step_reward

        # 3. No repetition penalty
        if not self._has_repetition(response):
            score += self.no_repetition_score

        # 4. Answer tag present
        if self._has_answer_tags(response):
            score += self.answer_tag_score

        # 5. Length penalty (discourage overly verbose output)
        length_penalty = self._compute_length_penalty(response)
        score -= length_penalty

        # Clamp to [0, max_process_score]
        return max(0.0, min(score, self.max_process_score))

    def compute_batch(self, responses: list) -> list:
        """Compute process rewards for a batch of responses."""
        return [self.compute(resp) for resp in responses]

    def compute_detailed(self, response: str) -> dict:
        """
        Compute detailed breakdown of process reward components.
        Useful for debugging and analysis.

        Args:
            response: Full model response.

        Returns:
            Dict with individual component scores.
        """
        has_think = self._has_think_tags(response)
        num_steps = self._count_steps(response)
        has_repetition = self._has_repetition(response)
        has_answer = self._has_answer_tags(response)
        length_penalty = self._compute_length_penalty(response)

        think_score = self.think_tag_score if has_think else 0.0
        step_reward = min(num_steps * self.step_score, self.max_step_score)
        repetition_score = self.no_repetition_score if not has_repetition else 0.0
        answer_score = self.answer_tag_score if has_answer else 0.0

        total = max(
            0.0,
            min(
                think_score + step_reward + repetition_score + answer_score - length_penalty,
                self.max_process_score,
            ),
        )

        return {
            "total": total,
            "think_tags": think_score,
            "step_count": num_steps,
            "step_reward": step_reward,
            "has_repetition": has_repetition,
            "repetition_score": repetition_score,
            "answer_tags": answer_score,
            "length_penalty": length_penalty,
            "response_length": len(response),
        }

    def _has_think_tags(self, response: str) -> bool:
        """Check if response contains properly formatted think tags."""
        return (
            self.think_open_tag in response
            and self.think_close_tag in response
        )

    def _has_answer_tags(self, response: str) -> bool:
        """Check if response contains properly formatted answer tags."""
        return (
            self.answer_open_tag in response
            and self.answer_close_tag in response
        )

    def _count_steps(self, response: str) -> int:
        """
        Count the number of distinct reasoning steps in the response.
        Looks for:
        - "Step N:" patterns
        - Numbered lines (1., 2., etc.)
        - Lines within think tags
        """
        # Extract content between think tags
        think_content = self._extract_think_content(response)
        if think_content is None:
            think_content = response

        # Count by explicit step markers
        step_patterns = [
            r"Step \d+",
            r"^\d+[\.\)]\s",
            r"^[-•]\s",
        ]

        max_steps = 0
        for pattern in step_patterns:
            matches = re.findall(pattern, think_content, re.MULTILINE)
            max_steps = max(max_steps, len(matches))

        if max_steps > 0:
            return max_steps

        # Fallback: count non-empty lines in think section
        lines = [
            line.strip()
            for line in think_content.split("\n")
            if line.strip() and len(line.strip()) > 10  # Meaningful lines
        ]
        return len(lines)

    def _has_repetition(self, response: str, threshold: float = 0.3) -> bool:
        """
        Check if response has significant repetition.
        
        Uses n-gram overlap to detect repetitive content.

        Args:
            response: Text to check.
            threshold: Repetition ratio threshold (0-1).

        Returns:
            True if repetition is detected.
        """
        # Split into sentences
        sentences = re.split(r'[.!?\n]', response)
        sentences = [s.strip().lower() for s in sentences if len(s.strip()) > 20]

        if len(sentences) <= 2:
            return False

        # Check for duplicate sentences
        unique_sentences = set(sentences)
        if len(unique_sentences) / len(sentences) < (1 - threshold):
            return True

        # Check trigram repetition
        words = response.lower().split()
        if len(words) < 10:
            return False

        trigrams = [tuple(words[i:i+3]) for i in range(len(words) - 2)]
        trigram_counts = Counter(trigrams)
        
        if not trigrams:
            return False

        # If any trigram appears more than 3 times, flag as repetitive
        max_count = max(trigram_counts.values())
        if max_count > 3 and max_count / len(trigrams) > 0.05:
            return True

        return False

    def _compute_length_penalty(self, response: str) -> float:
        """Compute length penalty for overly verbose responses."""
        excess = max(0, len(response) - self.length_penalty_threshold)
        return excess * self.length_penalty_rate

    def _extract_think_content(self, response: str) -> Optional[str]:
        """Extract content between think tags."""
        pattern = (
            re.escape(self.think_open_tag)
            + r"(.*?)"
            + re.escape(self.think_close_tag)
        )
        match = re.search(pattern, response, re.DOTALL)
        return match.group(1) if match else None
