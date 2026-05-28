"""
Verification Reward: Self-verification and step-level validation.
================================================================

Advanced reward component inspired by Monte Carlo Tree Search (MCTS):
- Scores each reasoning STEP individually (not just the final answer)
- Rewards self-verification (model checking its own work)
- Penalizes logical contradictions within the reasoning chain

This goes beyond simple outcome/process rewards to provide
fine-grained feedback on reasoning quality.
"""

import re
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class VerificationReward:
    """
    Step-level verification reward for reasoning chains.
    
    Innovations:
    1. Step Consistency Check — do later steps follow from earlier ones?
    2. Self-Verification Detection — does the model check its own work?
    3. Arithmetic Verification — are intermediate calculations correct?
    4. Conclusion Alignment — does the final answer match the reasoning?
    """

    def __init__(
        self,
        step_consistency_weight: float = 0.3,
        self_check_weight: float = 0.2,
        arithmetic_weight: float = 0.3,
        conclusion_weight: float = 0.2,
    ):
        self.step_consistency_weight = step_consistency_weight
        self.self_check_weight = self_check_weight
        self.arithmetic_weight = arithmetic_weight
        self.conclusion_weight = conclusion_weight

    def compute(self, response: str, ground_truth: str) -> float:
        """
        Compute verification reward.
        
        Args:
            response: Full model response with reasoning.
            ground_truth: Correct answer for arithmetic verification.
            
        Returns:
            Verification score (0.0 to 1.0).
        """
        scores = self.compute_detailed(response, ground_truth)
        
        total = (
            self.step_consistency_weight * scores["step_consistency"]
            + self.self_check_weight * scores["self_check"]
            + self.arithmetic_weight * scores["arithmetic"]
            + self.conclusion_weight * scores["conclusion_alignment"]
        )
        return min(max(total, 0.0), 1.0)

    def compute_detailed(self, response: str, ground_truth: str) -> Dict[str, float]:
        """Compute detailed verification scores."""
        steps = self._extract_steps(response)
        
        return {
            "step_consistency": self._check_step_consistency(steps),
            "self_check": self._detect_self_verification(response),
            "arithmetic": self._verify_arithmetic(steps, ground_truth),
            "conclusion_alignment": self._check_conclusion_alignment(response, steps),
            "num_steps": len(steps),
        }

    def _extract_steps(self, response: str) -> List[str]:
        """Extract individual reasoning steps."""
        # Try to find content between think tags
        think_match = re.search(r'<think>(.*?)</think>', response, re.DOTALL)
        content = think_match.group(1) if think_match else response
        
        # Split into steps
        step_patterns = [
            r'Step \d+[:.]\s*(.*?)(?=Step \d+|$)',
            r'\d+[\.\)]\s*(.*?)(?=\d+[\.\)]|$)',
        ]
        
        for pattern in step_patterns:
            matches = re.findall(pattern, content, re.DOTALL)
            if matches:
                return [m.strip() for m in matches if m.strip()]
        
        # Fallback: split by newlines
        lines = [l.strip() for l in content.split('\n') if l.strip() and len(l.strip()) > 10]
        return lines

    def _check_step_consistency(self, steps: List[str]) -> float:
        """
        Check if steps are logically consistent with each other.
        Uses heuristic: later steps should reference concepts from earlier steps.
        """
        if len(steps) < 2:
            return 0.5  # Can't assess consistency with one step
        
        score = 0.0
        total_checks = 0
        
        for i in range(1, len(steps)):
            current_step = steps[i].lower()
            # Check if current step references numbers or concepts from previous steps
            prev_numbers = set(re.findall(r'\d+\.?\d*', steps[i-1]))
            curr_numbers = set(re.findall(r'\d+\.?\d*', current_step))
            
            # Good: current step uses numbers from previous step (building on it)
            if prev_numbers and prev_numbers.intersection(curr_numbers):
                score += 1.0
            # Okay: different numbers but related words
            elif any(word in current_step for word in ['therefore', 'so', 'thus', 'from', 'using']):
                score += 0.7
            # Acceptable: at least not contradicting
            else:
                score += 0.3
            
            total_checks += 1
        
        return score / max(total_checks, 1)

    def _detect_self_verification(self, response: str) -> float:
        """
        Detect if the model verifies its own work.
        This is a sign of strong reasoning ability.
        """
        verification_patterns = [
            r'let me (?:verify|check|confirm|validate)',
            r'(?:checking|verifying)[:\s]',
            r'to (?:verify|confirm|check)',
            r'(?:double[- ]check|sanity check)',
            r'(?:this makes sense|this is correct) because',
            r'substitut(?:e|ing) back',
            r'(?:indeed|confirmed)',
            r'✓|✗|√',
        ]
        
        score = 0.0
        for pattern in verification_patterns:
            if re.search(pattern, response, re.IGNORECASE):
                score += 0.25
        
        return min(score, 1.0)

    def _verify_arithmetic(self, steps: List[str], ground_truth: str) -> float:
        """
        Check if arithmetic operations within steps are internally consistent.
        """
        if not steps:
            return 0.0
        
        correct_ops = 0
        total_ops = 0
        
        for step in steps:
            # Find arithmetic expressions like "15 × 240 = 3600"
            arith_patterns = [
                r'(\d+\.?\d*)\s*[×\*x]\s*(\d+\.?\d*)\s*=\s*(\d+\.?\d*)',
                r'(\d+\.?\d*)\s*[\+]\s*(\d+\.?\d*)\s*=\s*(\d+\.?\d*)',
                r'(\d+\.?\d*)\s*[-−]\s*(\d+\.?\d*)\s*=\s*(\d+\.?\d*)',
                r'(\d+\.?\d*)\s*[÷/]\s*(\d+\.?\d*)\s*=\s*(\d+\.?\d*)',
            ]
            
            for i, pattern in enumerate(arith_patterns):
                matches = re.findall(pattern, step)
                for match in matches:
                    try:
                        a, b, result = float(match[0]), float(match[1]), float(match[2])
                        if i == 0:  # multiplication
                            expected = a * b
                        elif i == 1:  # addition
                            expected = a + b
                        elif i == 2:  # subtraction
                            expected = a - b
                        elif i == 3:  # division
                            expected = a / b if b != 0 else None
                        
                        if expected is not None and abs(expected - result) < 0.01:
                            correct_ops += 1
                        total_ops += 1
                    except (ValueError, ZeroDivisionError):
                        continue
        
        if total_ops == 0:
            return 0.5  # No verifiable arithmetic found
        
        return correct_ops / total_ops

    def _check_conclusion_alignment(self, response: str, steps: List[str]) -> float:
        """
        Check if the final answer aligns with the reasoning steps.
        The last step's numbers should relate to the final answer.
        """
        # Extract final answer
        answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
        if not answer_match:
            return 0.3
        
        answer = answer_match.group(1).strip()
        answer_numbers = set(re.findall(r'-?\d+\.?\d*', answer))
        
        if not steps or not answer_numbers:
            return 0.5
        
        # Check if the answer number appears in the last step(s)
        last_steps = steps[-2:] if len(steps) >= 2 else steps
        last_steps_text = ' '.join(last_steps)
        last_numbers = set(re.findall(r'-?\d+\.?\d*', last_steps_text))
        
        # Good alignment: answer number is derived in the last steps
        if answer_numbers.intersection(last_numbers):
            return 1.0
        
        # Partial: answer number appears somewhere in reasoning
        all_text = ' '.join(steps)
        all_numbers = set(re.findall(r'-?\d+\.?\d*', all_text))
        if answer_numbers.intersection(all_numbers):
            return 0.7
        
        # Poor: answer appears to come from nowhere
        return 0.2
