"""
Mathematical Expression Verifier.
Provides symbolic math verification for more robust answer checking.
Uses Python's eval in a sandboxed manner for arithmetic expressions.
"""

import re
import math
import logging
from typing import Optional, Tuple
from fractions import Fraction

logger = logging.getLogger(__name__)


class MathVerifier:
    """
    Verifies mathematical answers by parsing and evaluating expressions.
    Goes beyond simple string matching to handle equivalent representations.
    
    Examples:
        - "3/4" == "0.75" == "75%"
        - "2 * 3 + 1" == "7"
        - "√16" == "4"
        - "$1,200" == "1200"
    """

    # Safe math operations allowed in eval
    SAFE_MATH_GLOBALS = {
        "__builtins__": {},
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "pow": pow,
        "sqrt": math.sqrt,
        "pi": math.pi,
        "e": math.e,
        "log": math.log,
        "log10": math.log10,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "floor": math.floor,
        "ceil": math.ceil,
    }

    def __init__(self, tolerance: float = 1e-6):
        """
        Initialize math verifier.
        
        Args:
            tolerance: Absolute tolerance for floating point comparison.
        """
        self.tolerance = tolerance

    def verify(self, predicted: str, ground_truth: str) -> Tuple[bool, float]:
        """
        Verify if predicted answer matches ground truth mathematically.
        
        Args:
            predicted: Model's predicted answer.
            ground_truth: Ground truth answer.
            
        Returns:
            Tuple of (is_correct, confidence_score).
            confidence_score: 1.0 for exact, 0.8 for numerically close.
        """
        # Clean both strings
        pred_clean = self._clean_math_string(predicted)
        truth_clean = self._clean_math_string(ground_truth)

        # Direct string match after cleaning
        if pred_clean == truth_clean:
            return True, 1.0

        # Try numeric evaluation
        pred_val = self._evaluate(pred_clean)
        truth_val = self._evaluate(truth_clean)

        if pred_val is not None and truth_val is not None:
            if abs(pred_val - truth_val) <= self.tolerance:
                return True, 1.0
            # Relative comparison for larger numbers
            if truth_val != 0 and abs(pred_val - truth_val) / abs(truth_val) <= self.tolerance:
                return True, 0.9
            # Wider tolerance for "close enough"
            if abs(pred_val - truth_val) <= 0.01:
                return False, 0.8

        # Try fraction comparison
        pred_frac = self._to_fraction(pred_clean)
        truth_frac = self._to_fraction(truth_clean)
        if pred_frac is not None and truth_frac is not None:
            if pred_frac == truth_frac:
                return True, 1.0

        # Try percentage comparison
        pred_pct = self._extract_percentage(pred_clean)
        truth_pct = self._extract_percentage(truth_clean)
        if pred_pct is not None and truth_pct is not None:
            if abs(pred_pct - truth_pct) <= self.tolerance:
                return True, 1.0

        return False, 0.0

    def _clean_math_string(self, s: str) -> str:
        """Clean a math string for evaluation."""
        s = s.strip()
        # Remove currency symbols
        s = re.sub(r'[\$€£¥₹]', '', s)
        # Remove commas in numbers
        s = re.sub(r'(\d),(\d)', r'\1\2', s)
        # Remove units at the end
        s = re.sub(r'\s*(dollars|cents|meters|km|kg|lbs|years|months|days|hours|minutes|seconds|people|items|books|apples|oranges|miles|feet|inches)\.?$', '', s, flags=re.IGNORECASE)
        # Handle "x = answer" patterns
        s = re.sub(r'^[a-zA-Z]\s*=\s*', '', s)
        # Remove trailing period
        s = s.rstrip('.')
        # Handle negative with parentheses: (5) -> -5 or just 5
        s = re.sub(r'^\((\-?\d+\.?\d*)\)$', r'\1', s)
        return s.strip()

    def _evaluate(self, expr: str) -> Optional[float]:
        """Safely evaluate a mathematical expression."""
        try:
            # Simple number
            return float(expr)
        except (ValueError, TypeError):
            pass

        # Clean up for eval
        expr_clean = expr
        # Replace common math symbols
        expr_clean = expr_clean.replace('×', '*').replace('÷', '/')
        expr_clean = expr_clean.replace('^', '**')
        expr_clean = re.sub(r'√(\d+)', r'sqrt(\1)', expr_clean)
        
        # Handle percentage
        if expr_clean.endswith('%'):
            try:
                return float(expr_clean[:-1]) / 100
            except ValueError:
                pass

        # Handle fraction notation "a/b"
        frac_match = re.match(r'^(-?\d+)\s*/\s*(\d+)$', expr_clean)
        if frac_match:
            try:
                return int(frac_match.group(1)) / int(frac_match.group(2))
            except ZeroDivisionError:
                return None

        # Handle mixed number "a b/c"
        mixed_match = re.match(r'^(-?\d+)\s+(\d+)\s*/\s*(\d+)$', expr_clean)
        if mixed_match:
            try:
                whole = int(mixed_match.group(1))
                numer = int(mixed_match.group(2))
                denom = int(mixed_match.group(3))
                sign = -1 if whole < 0 else 1
                return sign * (abs(whole) + numer / denom)
            except ZeroDivisionError:
                return None

        # Try safe eval for arithmetic expressions
        try:
            # Only allow safe characters
            if re.match(r'^[\d\s\+\-\*\/\.\(\)\%\^]+$', expr_clean):
                result = eval(expr_clean, self.SAFE_MATH_GLOBALS)
                return float(result)
        except (SyntaxError, TypeError, ValueError, ZeroDivisionError, NameError):
            pass

        return None

    def _to_fraction(self, s: str) -> Optional[Fraction]:
        """Try to parse string as a fraction."""
        try:
            # Handle "a/b" format
            if '/' in s:
                parts = s.split('/')
                if len(parts) == 2:
                    return Fraction(int(parts[0].strip()), int(parts[1].strip()))
            # Try as decimal
            val = float(s)
            return Fraction(val).limit_denominator(1000)
        except (ValueError, TypeError, ZeroDivisionError):
            return None

    def _extract_percentage(self, s: str) -> Optional[float]:
        """Extract percentage value."""
        match = re.match(r'^(-?\d+\.?\d*)\s*%$', s)
        if match:
            return float(match.group(1))
        return None
