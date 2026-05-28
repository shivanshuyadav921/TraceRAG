"""
Unit tests for reward functions.
Validates that the reward system correctly scores model outputs.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rewards.outcome_reward import OutcomeReward
from src.rewards.process_reward import ProcessReward
from src.rewards.combined_reward import CombinedReward


class TestOutcomeReward:
    """Tests for OutcomeReward."""

    def setup_method(self):
        self.reward = OutcomeReward()

    # ---- Exact match tests ----

    def test_exact_match_with_tags(self):
        """Test exact match when answer is in tags."""
        response = "<think>Let me solve this.</think>\n<answer>42</answer>"
        assert self.reward.compute(response, "42") == 1.0

    def test_exact_match_string(self):
        """Test exact match with string answers."""
        response = "<answer>yes</answer>"
        assert self.reward.compute(response, "yes") == 1.0

    def test_exact_match_case_insensitive(self):
        """Test case-insensitive matching."""
        response = "<answer>True</answer>"
        assert self.reward.compute(response, "true") == 1.0

    # ---- Numeric matching tests ----

    def test_numeric_exact(self):
        """Test numeric exact match with different formats."""
        response = "<answer>1200</answer>"
        assert self.reward.compute(response, "$1,200") == 1.0

    def test_numeric_close(self):
        """Test numerically close answers."""
        response = "<answer>3.14159</answer>"
        # Within tolerance
        assert self.reward.compute(response, "3.14159") == 1.0

    def test_numeric_with_units(self):
        """Test numeric match ignoring units."""
        response = "<answer>25</answer>"
        assert self.reward.compute(response, "25 dollars") == 1.0

    # ---- Multiple choice tests ----

    def test_multiple_choice_letter(self):
        """Test multiple choice letter matching."""
        response = "<answer>B</answer>"
        assert self.reward.compute(response, "B") == 1.0

    def test_multiple_choice_with_period(self):
        """Test 'A.' style answers."""
        response = "<answer>A</answer>"
        assert self.reward.compute(response, "A.") == 1.0

    # ---- Wrong answer tests ----

    def test_wrong_answer(self):
        """Test completely wrong answer."""
        response = "<answer>100</answer>"
        assert self.reward.compute(response, "42") == 0.0

    def test_no_answer_tag(self):
        """Test when no answer tag is present but answer is extractable."""
        response = "The answer is 42."
        assert self.reward.compute(response, "42") == 1.0

    def test_no_answer_at_all(self):
        """Test when no answer can be extracted."""
        response = "I'm not sure about this problem."
        # Should still try to extract something
        result = self.reward.compute(response, "42")
        assert result == 0.0

    # ---- Answer extraction tests ----

    def test_extract_from_tags(self):
        """Test answer extraction from tags."""
        response = "<think>Steps here</think>\n<answer>7.5</answer>"
        assert self.reward.extract_answer(response) == "7.5"

    def test_extract_from_pattern(self):
        """Test answer extraction from 'the answer is' pattern."""
        response = "After calculation, the answer is 156."
        assert self.reward.extract_answer(response) == "156"

    def test_extract_from_hash(self):
        """Test answer extraction from #### pattern."""
        response = "Step 1: ...\n#### 42"
        assert self.reward.extract_answer(response) == "42"


class TestProcessReward:
    """Tests for ProcessReward."""

    def setup_method(self):
        self.reward = ProcessReward()

    def test_perfect_format(self):
        """Test response with perfect format gets high score."""
        response = (
            "<think>\n"
            "Step 1: Identify the problem.\n"
            "Step 2: Set up the equation.\n"
            "Step 3: Solve for x.\n"
            "Step 4: Verify the answer.\n"
            "</think>\n"
            "<answer>42</answer>"
        )
        score = self.reward.compute(response)
        assert score > 0.4  # Should get high process score

    def test_no_tags(self):
        """Test response without tags gets low score."""
        response = "The answer is 42."
        score = self.reward.compute(response)
        assert score < 0.3

    def test_think_tags_only(self):
        """Test response with think tags but no answer tag."""
        response = "<think>Let me think about this.</think>\nSo the answer is 42."
        score = self.reward.compute(response)
        # Gets think_tag_score but not answer_tag_score
        assert score >= 0.2

    def test_repetition_penalty(self):
        """Test that repetitive content gets penalized."""
        response = (
            "<think>\n"
            + "The number is important. " * 20  # Excessive repetition
            + "\n</think>\n<answer>42</answer>"
        )
        score = self.reward.compute(response)
        detailed = self.reward.compute_detailed(response)
        assert detailed["has_repetition"] is True

    def test_length_penalty(self):
        """Test that overly long responses get penalized."""
        response = (
            "<think>\n"
            + "x" * 2000  # Very long
            + "\n</think>\n<answer>42</answer>"
        )
        detailed = self.reward.compute_detailed(response)
        assert detailed["length_penalty"] > 0

    def test_multi_step_bonus(self):
        """Test that multi-step reasoning gets rewarded."""
        # Few steps
        short = "<think>\nStep 1: Done.\n</think>\n<answer>42</answer>"
        # Many steps
        long = (
            "<think>\n"
            "Step 1: Read the problem.\n"
            "Step 2: Identify variables.\n"
            "Step 3: Set up equation.\n"
            "Step 4: Solve.\n"
            "Step 5: Verify.\n"
            "</think>\n"
            "<answer>42</answer>"
        )
        short_score = self.reward.compute(short)
        long_score = self.reward.compute(long)
        assert long_score > short_score

    def test_detailed_breakdown(self):
        """Test detailed score breakdown."""
        response = (
            "<think>\n"
            "Step 1: Analysis.\n"
            "Step 2: Solution.\n"
            "</think>\n"
            "<answer>42</answer>"
        )
        details = self.reward.compute_detailed(response)
        assert "total" in details
        assert "think_tags" in details
        assert "step_count" in details
        assert "answer_tags" in details
        assert details["think_tags"] > 0
        assert details["answer_tags"] > 0
        assert details["step_count"] >= 2


class TestCombinedReward:
    """Tests for CombinedReward."""

    def setup_method(self):
        self.reward = CombinedReward(outcome_weight=0.7, process_weight=0.3)

    def test_perfect_response(self):
        """Test response that is both correct and well-formatted."""
        response = (
            "<think>\n"
            "Step 1: The question asks for 2 + 2.\n"
            "Step 2: 2 + 2 = 4.\n"
            "Step 3: Let me verify: 4 - 2 = 2. ✓\n"
            "</think>\n"
            "<answer>4</answer>"
        )
        score = self.reward.compute(response, "4")
        # Should get high combined score
        assert score > 0.8

    def test_correct_but_badly_formatted(self):
        """Test correct answer but poor formatting."""
        response = "4"
        score = self.reward.compute(response, "4")
        # Gets outcome reward but low process reward
        assert 0.5 < score < 0.8

    def test_wrong_but_well_formatted(self):
        """Test wrong answer but good reasoning format."""
        response = (
            "<think>\n"
            "Step 1: Analysis.\n"
            "Step 2: Calculation.\n"
            "Step 3: Conclusion.\n"
            "</think>\n"
            "<answer>5</answer>"
        )
        score = self.reward.compute(response, "4")
        # Gets process reward but no outcome reward
        assert score < 0.3
        assert score > 0.0  # Process reward still contributes

    def test_completely_wrong(self):
        """Test completely wrong and poorly formatted response."""
        response = "I don't know"
        score = self.reward.compute(response, "42")
        assert score < 0.1

    def test_grpo_group_scoring(self):
        """Test scoring a group of responses (as in GRPO)."""
        responses = [
            "<think>Step 1: 2+2=4</think>\n<answer>4</answer>",  # Correct + formatted
            "4",  # Correct but no format
            "<think>Step 1: 2+2=5</think>\n<answer>5</answer>",  # Wrong but formatted
            "I don't know",  # Wrong and no format
        ]
        rewards = self.reward.compute_for_grpo(responses, "4")
        
        assert len(rewards) == 4
        # Best response should score highest
        assert rewards[0] > rewards[1] > rewards[2] > rewards[3]

    def test_detailed_breakdown(self):
        """Test detailed combined reward breakdown."""
        response = "<think>Step 1: 2+2=4</think>\n<answer>4</answer>"
        details = self.reward.compute_detailed(response, "4")
        
        assert "combined_reward" in details
        assert "outcome_reward" in details
        assert "process_reward" in details
        assert "is_correct" in details
        assert details["is_correct"] is True
        assert details["outcome_reward"] == 1.0

    def test_from_config(self):
        """Test creating from config dict."""
        config = {
            "outcome_weight": 0.6,
            "process_weight": 0.4,
            "outcome": {"exact_match_score": 1.0},
            "process": {"think_tag_score": 0.3},
        }
        reward = CombinedReward.from_config(config)
        assert reward.outcome_weight == 0.6
        assert reward.process_weight == 0.4


class TestMathVerifier:
    """Tests for MathVerifier utility."""

    def setup_method(self):
        from src.utils.math_verify import MathVerifier
        self.verifier = MathVerifier()

    def test_integer_match(self):
        """Test integer comparison."""
        is_correct, _ = self.verifier.verify("42", "42")
        assert is_correct

    def test_float_match(self):
        """Test float comparison."""
        is_correct, _ = self.verifier.verify("3.14", "3.14")
        assert is_correct

    def test_fraction_to_decimal(self):
        """Test fraction equals decimal."""
        is_correct, _ = self.verifier.verify("0.75", "3/4")
        assert is_correct

    def test_currency_match(self):
        """Test currency formatting."""
        is_correct, _ = self.verifier.verify("1200", "$1,200")
        assert is_correct

    def test_percentage(self):
        """Test percentage matching."""
        is_correct, _ = self.verifier.verify("25%", "25%")
        assert is_correct

    def test_arithmetic_expression(self):
        """Test arithmetic expression evaluation."""
        is_correct, _ = self.verifier.verify("7", "3 + 4")
        assert is_correct

    def test_different_numbers(self):
        """Test different numbers don't match."""
        is_correct, _ = self.verifier.verify("41", "42")
        assert not is_correct


class TestAnswerExtraction:
    """Tests for AnswerExtractor with self-consistency."""

    def setup_method(self):
        from src.utils.answer_extraction import AnswerExtractor
        self.extractor = AnswerExtractor()

    def test_extract_from_tags(self):
        """Test basic tag extraction."""
        response = "<think>reasoning</think>\n<answer>42</answer>"
        assert self.extractor.extract(response) == "42"

    def test_extract_with_confidence(self):
        """Test confidence scores."""
        # Tag-based should have highest confidence
        response = "<answer>42</answer>"
        answer, confidence = self.extractor.extract_with_confidence(response)
        assert answer == "42"
        assert confidence == 1.0

        # Pattern-based should have lower confidence
        response = "The answer is 42."
        answer, confidence = self.extractor.extract_with_confidence(response)
        assert answer == "42"
        assert confidence < 1.0

    def test_self_consistency_unanimous(self):
        """Test self-consistency when all responses agree."""
        responses = [
            "<answer>42</answer>",
            "<think>math</think><answer>42</answer>",
            "The answer is 42.",
        ]
        answer, agreement = self.extractor.self_consistency_vote(responses)
        assert answer == "42"
        assert agreement == 1.0

    def test_self_consistency_majority(self):
        """Test self-consistency with majority voting."""
        responses = [
            "<answer>42</answer>",
            "<answer>42</answer>",
            "<answer>42</answer>",
            "<answer>43</answer>",
            "<answer>41</answer>",
        ]
        answer, agreement = self.extractor.self_consistency_vote(responses)
        assert answer == "42"
        assert agreement == 3 / 5

    def test_self_consistency_equivalent_answers(self):
        """Test that equivalent answers are grouped (e.g., 42 and 42.0)."""
        responses = [
            "<answer>42</answer>",
            "<answer>42.0</answer>",
            "<answer>42.00</answer>",
            "<answer>43</answer>",
        ]
        answer, agreement = self.extractor.self_consistency_vote(responses)
        # 42, 42.0, 42.00 should all be grouped together
        assert agreement >= 3 / 4
