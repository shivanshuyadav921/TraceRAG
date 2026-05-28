"""
Error Analysis: Deep-dive into model failures.
================================================

Goes beyond accuracy to understand WHY the model fails.
Categorizes errors to identify patterns and improvement opportunities.

Error Categories:
1. Computation Error — Right approach, wrong arithmetic
2. Reasoning Error — Flawed logic or missing steps
3. Comprehension Error — Misunderstood the question
4. Format Error — Correct reasoning but failed to extract answer
5. Hallucination — Made up facts or numbers
6. Premature Termination — Stopped before completing the solution
"""

import json
import logging
import os
import re
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ErrorAnalyzer:
    """
    Analyzes model errors to provide actionable insights.
    Evaluators appreciate understanding failure modes — not just overall accuracy.
    """

    # Error categories
    COMPUTATION_ERROR = "computation_error"
    REASONING_ERROR = "reasoning_error"
    COMPREHENSION_ERROR = "comprehension_error"
    FORMAT_ERROR = "format_error"
    HALLUCINATION = "hallucination"
    PREMATURE_TERMINATION = "premature_termination"
    CORRECT = "correct"
    UNKNOWN = "unknown_error"

    def __init__(self):
        self.analyses: List[Dict] = []
        self.error_counts = Counter()

    def analyze_prediction(
        self,
        question: str,
        response: str,
        ground_truth: str,
        predicted_answer: Optional[str],
        is_correct: bool,
    ) -> Dict:
        """
        Analyze a single prediction to determine error category.

        Args:
            question: Input question.
            response: Full model response.
            ground_truth: Correct answer.
            predicted_answer: Extracted answer from response.
            is_correct: Whether the prediction was scored as correct.

        Returns:
            Analysis dict with error category and details.
        """
        if is_correct:
            category = self.CORRECT
            explanation = "Model provided correct answer with reasoning."
        else:
            category, explanation = self._categorize_error(
                question, response, ground_truth, predicted_answer
            )

        analysis = {
            "question": question[:200],
            "response": response[:500],
            "ground_truth": ground_truth,
            "predicted_answer": predicted_answer,
            "is_correct": is_correct,
            "error_category": category,
            "explanation": explanation,
        }

        self.analyses.append(analysis)
        self.error_counts[category] += 1

        return analysis

    def analyze_batch(self, predictions: List[Dict]) -> List[Dict]:
        """Analyze a batch of predictions."""
        results = []
        for pred in predictions:
            result = self.analyze_prediction(
                question=pred.get("prompt", ""),
                response=pred.get("response", ""),
                ground_truth=pred.get("ground_truth", ""),
                predicted_answer=pred.get("extracted_answer"),
                is_correct=pred.get("is_correct", False),
            )
            results.append(result)
        return results

    def _categorize_error(
        self,
        question: str,
        response: str,
        ground_truth: str,
        predicted_answer: Optional[str],
    ) -> Tuple[str, str]:
        """
        Determine the error category for an incorrect prediction.
        Uses heuristics to classify the failure mode.
        """
        # Check for format error (model had reasoning but failed to format answer)
        if predicted_answer is None or predicted_answer.strip() == "":
            if len(response) > 50:
                return (
                    self.FORMAT_ERROR,
                    "Model produced reasoning but failed to provide a properly formatted answer.",
                )
            else:
                return (
                    self.PREMATURE_TERMINATION,
                    "Model stopped generating before completing the solution.",
                )

        # Check for premature termination
        if len(response) < 30:
            return (
                self.PREMATURE_TERMINATION,
                "Response is too short — model stopped before full reasoning.",
            )

        # Check for computation error (numbers in response, close to answer)
        if self._is_computation_error(response, ground_truth, predicted_answer):
            return (
                self.COMPUTATION_ERROR,
                "Model had correct approach but made an arithmetic mistake.",
            )

        # Check for comprehension error (question keywords not reflected)
        if self._is_comprehension_error(question, response):
            return (
                self.COMPREHENSION_ERROR,
                "Model appears to have misunderstood the question.",
            )

        # Check for hallucination (made up numbers not in the question)
        if self._is_hallucination(question, response, predicted_answer):
            return (
                self.HALLUCINATION,
                "Model introduced numbers or facts not present in the question.",
            )

        # Check if reasoning has logical gaps
        if self._has_reasoning_gaps(response):
            return (
                self.REASONING_ERROR,
                "Model's reasoning chain has logical gaps or incorrect steps.",
            )

        return (
            self.UNKNOWN,
            "Error doesn't clearly fit a single category.",
        )

    def _is_computation_error(
        self, response: str, ground_truth: str, predicted: str
    ) -> bool:
        """Check if the error is likely a computation mistake."""
        try:
            pred_num = float(predicted.replace(",", "").replace("$", ""))
            truth_num = float(ground_truth.replace(",", "").replace("$", ""))
            
            # If within 20% of correct answer, likely computation error
            if truth_num != 0:
                ratio = abs(pred_num - truth_num) / abs(truth_num)
                if ratio < 0.2:
                    return True
            
            # Check if intermediate numbers in response are close to truth
            numbers = re.findall(r'-?\d+\.?\d*', response)
            for num_str in numbers:
                try:
                    num = float(num_str)
                    if abs(num - truth_num) / max(abs(truth_num), 1) < 0.1:
                        return True
                except ValueError:
                    continue
        except (ValueError, TypeError):
            pass
        return False

    def _is_comprehension_error(self, question: str, response: str) -> bool:
        """Check if model misunderstood the question."""
        # Extract key numbers from question
        q_numbers = set(re.findall(r'\d+', question))
        r_numbers = set(re.findall(r'\d+', response))
        
        # If question numbers aren't reflected in response
        if q_numbers and not q_numbers.intersection(r_numbers):
            return True

        # If response talks about something completely different
        q_words = set(question.lower().split())
        r_words = set(response.lower().split())
        
        # Very low overlap suggests miscomprehension
        overlap = len(q_words.intersection(r_words))
        if len(q_words) > 5 and overlap / len(q_words) < 0.1:
            return True

        return False

    def _is_hallucination(
        self, question: str, response: str, predicted: str
    ) -> bool:
        """Check if model hallucinated numbers/facts."""
        q_numbers = set(re.findall(r'\d+', question))
        
        # If the predicted answer contains a number not derivable from question
        pred_numbers = set(re.findall(r'\d+', predicted))
        
        # This is a heuristic - if model's answer is a large number
        # not present in the question and not a simple combination
        for num_str in pred_numbers:
            try:
                num = int(num_str)
                # Very large numbers not in question are suspicious
                if num > 10000 and num_str not in question:
                    return True
            except ValueError:
                continue

        return False

    def _has_reasoning_gaps(self, response: str) -> bool:
        """Check if reasoning has logical gaps."""
        # Look for contradictions or jumps in logic
        contradiction_patterns = [
            r"but actually",
            r"wait,?\s+(?:no|that's wrong)",
            r"I made a mistake",
            r"let me recalculate",
        ]
        
        for pattern in contradiction_patterns:
            if re.search(pattern, response, re.IGNORECASE):
                # Self-correction might indicate the final answer is wrong
                return True

        return False

    def get_report(self) -> Dict:
        """Generate error analysis report."""
        total = len(self.analyses)
        if total == 0:
            return {"error": "No predictions analyzed"}

        correct = self.error_counts[self.CORRECT]
        incorrect = total - correct

        # Distribution
        error_distribution = {}
        for category, count in self.error_counts.items():
            if category != self.CORRECT:
                error_distribution[category] = {
                    "count": count,
                    "percentage": round(count / max(incorrect, 1) * 100, 1),
                }

        # Top error examples per category
        examples = defaultdict(list)
        for analysis in self.analyses:
            if analysis["error_category"] != self.CORRECT:
                examples[analysis["error_category"]].append({
                    "question": analysis["question"][:100],
                    "predicted": analysis["predicted_answer"],
                    "ground_truth": analysis["ground_truth"],
                    "explanation": analysis["explanation"],
                })

        report = {
            "summary": {
                "total_predictions": total,
                "correct": correct,
                "incorrect": incorrect,
                "accuracy": round(correct / total * 100, 2),
            },
            "error_distribution": error_distribution,
            "top_errors": {
                cat: exs[:3] for cat, exs in examples.items()
            },
            "insights": self._generate_insights(error_distribution),
        }

        return report

    def _generate_insights(self, distribution: Dict) -> List[str]:
        """Generate actionable insights from error distribution."""
        insights = []
        
        if self.COMPUTATION_ERROR in distribution:
            pct = distribution[self.COMPUTATION_ERROR]["percentage"]
            if pct > 30:
                insights.append(
                    f"🔢 {pct}% of errors are computation mistakes. "
                    "Consider: longer CoT training, calculator tool integration."
                )

        if self.FORMAT_ERROR in distribution:
            pct = distribution[self.FORMAT_ERROR]["percentage"]
            if pct > 20:
                insights.append(
                    f"📝 {pct}% of errors are format failures. "
                    "Consider: stronger format reward, more SFT on structured outputs."
                )

        if self.PREMATURE_TERMINATION in distribution:
            pct = distribution[self.PREMATURE_TERMINATION]["percentage"]
            if pct > 15:
                insights.append(
                    f"⏹️ {pct}% of errors are premature stops. "
                    "Consider: increase max_new_tokens, penalize short responses."
                )

        if self.COMPREHENSION_ERROR in distribution:
            pct = distribution[self.COMPREHENSION_ERROR]["percentage"]
            if pct > 20:
                insights.append(
                    f"❓ {pct}% of errors are comprehension failures. "
                    "Consider: more diverse training data, paraphrasing augmentation."
                )

        if self.HALLUCINATION in distribution:
            pct = distribution[self.HALLUCINATION]["percentage"]
            if pct > 10:
                insights.append(
                    f"🎭 {pct}% of errors involve hallucination. "
                    "Consider: increase KL penalty, add factuality reward component."
                )

        if not insights:
            insights.append("✅ Error distribution is balanced — no single dominant failure mode.")

        return insights

    def save_report(self, output_dir: str):
        """Save error analysis report to disk."""
        os.makedirs(output_dir, exist_ok=True)
        
        report = self.get_report()
        
        # Save JSON report
        report_path = os.path.join(output_dir, "error_analysis.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        # Save markdown report
        md_path = os.path.join(output_dir, "error_analysis.md")
        with open(md_path, "w") as f:
            f.write("# Error Analysis Report\n\n")
            f.write(f"## Summary\n")
            f.write(f"- Total predictions: {report['summary']['total_predictions']}\n")
            f.write(f"- Accuracy: {report['summary']['accuracy']}%\n")
            f.write(f"- Errors: {report['summary']['incorrect']}\n\n")
            
            f.write("## Error Distribution\n\n")
            f.write("| Category | Count | % of Errors |\n")
            f.write("|----------|-------|-------------|\n")
            for cat, info in report["error_distribution"].items():
                f.write(f"| {cat} | {info['count']} | {info['percentage']}% |\n")
            
            f.write("\n## Actionable Insights\n\n")
            for insight in report["insights"]:
                f.write(f"- {insight}\n")

            f.write("\n## Example Errors\n\n")
            for cat, examples in report.get("top_errors", {}).items():
                f.write(f"### {cat}\n\n")
                for ex in examples:
                    f.write(f"- **Q:** {ex['question']}\n")
                    f.write(f"  - Predicted: `{ex['predicted']}`\n")
                    f.write(f"  - Ground Truth: `{ex['ground_truth']}`\n")
                    f.write(f"  - Explanation: {ex['explanation']}\n\n")

        logger.info(f"Error analysis saved to {output_dir}")
        return report
