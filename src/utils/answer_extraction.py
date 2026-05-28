"""
Advanced Answer Extraction with Self-Consistency.
Implements majority voting across multiple sampled responses for more
robust answer selection during evaluation.
"""

import re
import logging
from collections import Counter
from typing import List, Optional, Tuple

from .math_verify import MathVerifier

logger = logging.getLogger(__name__)


class AnswerExtractor:
    """
    Extracts and validates answers from model responses.
    Supports self-consistency (majority voting) for improved accuracy.
    """

    def __init__(
        self,
        answer_open_tag: str = "<answer>",
        answer_close_tag: str = "</answer>",
    ):
        self.answer_open_tag = answer_open_tag
        self.answer_close_tag = answer_close_tag
        self.math_verifier = MathVerifier()

    def extract(self, response: str) -> Optional[str]:
        """
        Extract the final answer from a model response.
        Tries multiple extraction strategies in order of reliability.

        Args:
            response: Full model response text.

        Returns:
            Extracted answer string or None.
        """
        # Strategy 1: Extract from answer tags
        answer = self._from_tags(response)
        if answer:
            return answer

        # Strategy 2: "The answer is X" pattern
        answer = self._from_answer_pattern(response)
        if answer:
            return answer

        # Strategy 3: "#### X" pattern (GSM8K format)
        answer = self._from_hash_pattern(response)
        if answer:
            return answer

        # Strategy 4: Last number in response
        answer = self._last_number(response)
        if answer:
            return answer

        # Strategy 5: Last meaningful line
        answer = self._last_line(response)
        return answer

    def extract_with_confidence(self, response: str) -> Tuple[Optional[str], float]:
        """
        Extract answer with confidence score based on extraction method used.

        Returns:
            Tuple of (answer, confidence) where confidence is 0.0 to 1.0.
        """
        # Try each strategy with decreasing confidence
        strategies = [
            (self._from_tags, 1.0),
            (self._from_answer_pattern, 0.9),
            (self._from_hash_pattern, 0.85),
            (self._last_number, 0.6),
            (self._last_line, 0.3),
        ]

        for strategy_fn, confidence in strategies:
            answer = strategy_fn(response)
            if answer:
                return answer, confidence

        return None, 0.0

    def self_consistency_vote(
        self,
        responses: List[str],
        temperature: float = 0.7,
        num_samples: int = 8,
    ) -> Tuple[Optional[str], float]:
        """
        Apply self-consistency (majority voting) across multiple responses.
        
        This is a key technique to boost accuracy:
        - Generate multiple responses at temperature > 0
        - Extract answer from each
        - Return the most common answer (majority vote)
        
        Args:
            responses: List of model responses (sampled at temperature > 0).
            temperature: Temperature used for sampling (for logging).
            num_samples: Number of samples used.

        Returns:
            Tuple of (majority_answer, agreement_ratio).
            agreement_ratio: fraction of responses that agree with majority.
        """
        answers = []
        for resp in responses:
            answer = self.extract(resp)
            if answer is not None:
                # Normalize the answer for comparison
                normalized = self._normalize_for_voting(answer)
                answers.append((normalized, answer))

        if not answers:
            return None, 0.0

        # Group equivalent answers
        groups = self._group_equivalent_answers(answers)
        
        # Find the largest group (majority)
        largest_group = max(groups, key=lambda g: len(g))
        majority_answer = largest_group[0][1]  # Original (unnormalized) answer
        agreement_ratio = len(largest_group) / len(answers)

        logger.debug(
            f"Self-consistency: {len(answers)}/{len(responses)} extracted, "
            f"majority={majority_answer} ({agreement_ratio:.0%} agreement)"
        )

        return majority_answer, agreement_ratio

    def _group_equivalent_answers(
        self, answers: List[Tuple[str, str]]
    ) -> List[List[Tuple[str, str]]]:
        """Group answers that are mathematically equivalent."""
        groups = []
        
        for normalized, original in answers:
            placed = False
            for group in groups:
                # Check if this answer is equivalent to the group
                group_normalized = group[0][0]
                is_match, _ = self.math_verifier.verify(normalized, group_normalized)
                if is_match or normalized == group_normalized:
                    group.append((normalized, original))
                    placed = True
                    break
            
            if not placed:
                groups.append([(normalized, original)])

        return groups

    def _normalize_for_voting(self, answer: str) -> str:
        """Normalize answer for majority voting comparison."""
        s = answer.strip().lower()
        # Remove common formatting
        s = re.sub(r'[\$,€£¥₹]', '', s)
        s = re.sub(r'\s+', ' ', s)
        s = s.rstrip('.')
        # Try to convert to a canonical number form
        try:
            val = float(s.replace(',', ''))
            # Use consistent decimal representation
            if val == int(val):
                return str(int(val))
            return f"{val:.6f}".rstrip('0').rstrip('.')
        except (ValueError, TypeError):
            pass
        return s

    def _from_tags(self, response: str) -> Optional[str]:
        """Extract from answer tags."""
        pattern = (
            re.escape(self.answer_open_tag) + r"(.*?)" + re.escape(self.answer_close_tag)
        )
        match = re.search(pattern, response, re.DOTALL)
        if match:
            return match.group(1).strip()
        return None

    def _from_answer_pattern(self, response: str) -> Optional[str]:
        """Extract from 'The answer is X' patterns."""
        patterns = [
            r"[Tt]he\s+(?:final\s+)?answer\s+is[:\s]+(.+?)(?:\.|$|\n)",
            r"[Aa]nswer[:\s]+(.+?)(?:\.|$|\n)",
            r"[Tt]herefore[,:\s]+(?:the answer is\s+)?(.+?)(?:\.|$|\n)",
            r"[Ss]o[,:\s]+(?:the answer is\s+)?(.+?)(?:\.|$|\n)",
        ]
        for pat in patterns:
            match = re.search(pat, response)
            if match:
                answer = match.group(1).strip()
                if len(answer) < 100:  # Sanity check
                    return answer
        return None

    def _from_hash_pattern(self, response: str) -> Optional[str]:
        """Extract from #### pattern (GSM8K style)."""
        match = re.search(r'####\s*(.+?)(?:\n|$)', response)
        if match:
            return match.group(1).strip()
        return None

    def _last_number(self, response: str) -> Optional[str]:
        """Extract the last number from the response."""
        # Find all numbers (including decimals and negatives)
        numbers = re.findall(r'-?\d+\.?\d*', response)
        if numbers:
            return numbers[-1]
        return None

    def _last_line(self, response: str) -> Optional[str]:
        """Extract the last meaningful line."""
        lines = [l.strip() for l in response.strip().split('\n') if l.strip()]
        if lines:
            last = lines[-1]
            # Remove common prefixes
            for prefix in ["Therefore,", "So,", "Thus,", "Hence,", "Finally,"]:
                if last.startswith(prefix):
                    last = last[len(prefix):].strip()
            if len(last) < 200:
                return last
        return None
