"""
Outcome Reward: Evaluates correctness of the final answer.
Supports exact match, numeric comparison, and multiple-choice matching.
"""

import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class OutcomeReward:
    """
    Reward based on whether the model's final answer is correct.
    
    Scoring:
    - Exact match: 1.0
    - Numerically close: 0.8
    - Wrong answer: 0.0
    """

    def __init__(
        self,
        exact_match_score: float = 1.0,
        numeric_close_score: float = 0.8,
        numeric_tolerance: float = 0.01,
        wrong_answer_score: float = 0.0,
        answer_open_tag: str = "<answer>",
        answer_close_tag: str = "</answer>",
    ):
        """
        Initialize outcome reward.

        Args:
            exact_match_score: Score for exact match.
            numeric_close_score: Score for numerically close answers.
            numeric_tolerance: Tolerance for numeric comparison.
            wrong_answer_score: Score for incorrect answers.
            answer_open_tag: Opening tag for answer extraction.
            answer_close_tag: Closing tag for answer extraction.
        """
        self.exact_match_score = exact_match_score
        self.numeric_close_score = numeric_close_score
        self.numeric_tolerance = numeric_tolerance
        self.wrong_answer_score = wrong_answer_score
        self.answer_open_tag = answer_open_tag
        self.answer_close_tag = answer_close_tag

    def compute(self, response: str, ground_truth: str) -> float:
        """
        Compute outcome reward for a single response.

        Args:
            response: Full model response (may include reasoning + answer).
            ground_truth: The correct answer string.

        Returns:
            Float reward score.
        """
        extracted = self.extract_answer(response)

        if extracted is None:
            # No answer tag found - try to extract from the end
            extracted = self._extract_last_number_or_word(response)
            if extracted is None:
                return self.wrong_answer_score

        # Normalize both answers
        pred_normalized = self._normalize(extracted)
        truth_normalized = self._normalize(ground_truth)

        # Check exact match (after normalization)
        if pred_normalized == truth_normalized:
            return self.exact_match_score

        # Check numeric closeness
        pred_num = self._to_number(extracted)
        truth_num = self._to_number(ground_truth)

        if pred_num is not None and truth_num is not None:
            if abs(pred_num - truth_num) <= self.numeric_tolerance:
                return self.exact_match_score
            # Relative tolerance for larger numbers
            if truth_num != 0 and abs(pred_num - truth_num) / abs(truth_num) <= self.numeric_tolerance:
                return self.numeric_close_score

        # Check multiple choice match (A, B, C, D)
        pred_choice = self._extract_choice(extracted)
        truth_choice = self._extract_choice(ground_truth)
        if pred_choice and truth_choice and pred_choice == truth_choice:
            return self.exact_match_score

        return self.wrong_answer_score

    def compute_batch(self, responses: list, ground_truths: list) -> list:
        """Compute rewards for a batch of responses."""
        return [
            self.compute(resp, truth)
            for resp, truth in zip(responses, ground_truths)
        ]

    def extract_answer(self, response: str) -> Optional[str]:
        """
        Extract answer from between answer tags.

        Args:
            response: Full model response.

        Returns:
            Extracted answer string or None.
        """
        # Try to find content between answer tags
        pattern = re.escape(self.answer_open_tag) + r"(.*?)" + re.escape(self.answer_close_tag)
        match = re.search(pattern, response, re.DOTALL)
        
        if match:
            return match.group(1).strip()

        # Fallback: look for "The answer is" pattern
        patterns = [
            r"[Tt]he (?:final )?answer is[:\s]*(.+?)(?:\.|$)",
            r"[Aa]nswer[:\s]*(.+?)(?:\.|$)",
            r"####\s*(.+?)(?:\n|$)",
        ]
        
        for pat in patterns:
            match = re.search(pat, response)
            if match:
                return match.group(1).strip()

        return None

    def _extract_last_number_or_word(self, response: str) -> Optional[str]:
        """Extract the last number or meaningful word from response."""
        # Try to find the last number
        numbers = re.findall(r'-?\d+\.?\d*', response)
        if numbers:
            return numbers[-1]
        
        # Try last line
        lines = [l.strip() for l in response.strip().split('\n') if l.strip()]
        if lines:
            last_line = lines[-1]
            # Remove common prefixes
            for prefix in ["Therefore,", "So,", "Thus,", "Hence,"]:
                if last_line.startswith(prefix):
                    last_line = last_line[len(prefix):].strip()
            return last_line

        return None

    def _normalize(self, text: str) -> str:
        """Normalize text for comparison."""
        text = text.strip().lower()
        # Remove common units and formatting
        text = re.sub(r'[\$,€£¥%]', '', text)
        text = re.sub(r'\s+', ' ', text)
        # Remove trailing period
        text = text.rstrip('.')
        return text

    def _to_number(self, text: str) -> Optional[float]:
        """Try to convert text to a number."""
        text = text.strip()
        # Remove currency symbols, commas, spaces
        text = re.sub(r'[\$,€£¥\s]', '', text)
        # Remove units
        text = re.sub(r'[a-zA-Z]+$', '', text).strip()
        
        try:
            return float(text)
        except (ValueError, TypeError):
            return None

    def _extract_choice(self, text: str) -> Optional[str]:
        """Extract multiple choice letter (A, B, C, D, E)."""
        text = text.strip().upper()
        # Direct single letter
        if len(text) == 1 and text in "ABCDE":
            return text
        # Letter with parenthesis or period
        match = re.match(r'^([A-E])[\.\)\s]', text)
        if match:
            return match.group(1)
        # "Option A" pattern
        match = re.search(r'option\s+([A-E])', text, re.IGNORECASE)
        if match:
            return match.group(1).upper()
        return None
