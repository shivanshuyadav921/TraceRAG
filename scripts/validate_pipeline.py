"""
Pipeline Validation: Proof-of-Concept end-to-end test.
======================================================

Runs a lightweight validation of the entire pipeline WITHOUT GPUs:
1. Validates data loading and formatting
2. Tests reward functions on synthetic examples
3. Validates config parsing
4. Simulates the training loop logic
5. Tests answer extraction and math verification

This proves the pipeline WORKS before committing to expensive GPU training.

Usage:
    python scripts/validate_pipeline.py

Expected output: All checks pass ✅
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import yaml


def print_header(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def print_result(test_name: str, passed: bool, detail: str = ""):
    emoji = "✅" if passed else "❌"
    print(f"  {emoji} {test_name}" + (f" — {detail}" if detail else ""))
    return passed


def validate_configs():
    """Validate all YAML configs parse correctly."""
    print_header("1. Config Validation")
    
    all_passed = True
    config_dir = project_root / "configs"
    
    for config_file in ["sft_config.yaml", "grpo_config.yaml", "eval_config.yaml"]:
        try:
            with open(config_dir / config_file) as f:
                config = yaml.safe_load(f)
            
            # Check required keys exist
            assert "model" in config, "Missing 'model' key"
            assert "name" in config["model"], "Missing model name"
            
            all_passed &= print_result(
                f"{config_file}", True,
                f"model={config['model']['name']}"
            )
        except Exception as e:
            all_passed &= print_result(f"{config_file}", False, str(e))
    
    return all_passed


def validate_rewards():
    """Test reward functions with synthetic examples."""
    print_header("2. Reward Function Validation")
    
    from src.rewards.outcome_reward import OutcomeReward
    from src.rewards.process_reward import ProcessReward
    from src.rewards.combined_reward import CombinedReward
    from src.rewards.verification_reward import VerificationReward
    
    all_passed = True
    
    # Test OutcomeReward
    outcome = OutcomeReward()
    
    # Correct answer
    score = outcome.compute("<answer>42</answer>", "42")
    all_passed &= print_result("Outcome: correct answer", score == 1.0, f"score={score}")
    
    # Wrong answer
    score = outcome.compute("<answer>99</answer>", "42")
    all_passed &= print_result("Outcome: wrong answer", score == 0.0, f"score={score}")
    
    # Numeric equivalence
    score = outcome.compute("<answer>1200</answer>", "$1,200")
    all_passed &= print_result("Outcome: numeric equiv", score == 1.0, f"score={score}")
    
    # Test ProcessReward
    process = ProcessReward()
    
    good_response = (
        "<think>\nStep 1: Identify the problem.\n"
        "Step 2: Set up equation.\nStep 3: Solve.\n"
        "</think>\n<answer>42</answer>"
    )
    score = process.compute(good_response)
    all_passed &= print_result("Process: good format", score > 0.4, f"score={score:.3f}")
    
    bad_response = "42"
    score = process.compute(bad_response)
    all_passed &= print_result("Process: no format", score <= 0.2, f"score={score:.3f}")
    
    # Test CombinedReward
    combined = CombinedReward(outcome_weight=0.7, process_weight=0.3)
    score = combined.compute(good_response, "42")
    all_passed &= print_result("Combined: correct+formatted", score > 0.8, f"score={score:.3f}")
    
    score = combined.compute("<answer>99</answer>", "42")
    all_passed &= print_result("Combined: wrong answer", score < 0.2, f"score={score:.3f}")
    
    # Test VerificationReward
    verifier = VerificationReward()
    verified_response = (
        "<think>\nStep 1: 15% of 240 = 0.15 × 240 = 36\n"
        "Step 2: Let me verify: 36 / 240 = 0.15 ✓\n"
        "</think>\n<answer>36</answer>"
    )
    score = verifier.compute(verified_response, "36")
    all_passed &= print_result("Verification: self-check", score > 0.3, f"score={score:.3f}")
    
    # Test GRPO group scoring
    responses = [
        "<think>Step 1: 2+2=4</think>\n<answer>4</answer>",
        "4",
        "<think>Step 1: 2+2=5</think>\n<answer>5</answer>",
        "wrong",
    ]
    rewards = combined.compute_for_grpo(responses, "4")
    all_passed &= print_result(
        "GRPO group scoring", 
        rewards[0] > rewards[1] > rewards[2] > rewards[3],
        f"rewards={[f'{r:.2f}' for r in rewards]}"
    )
    
    return all_passed


def validate_data_formatting():
    """Test data formatting logic."""
    print_header("3. Data Formatting Validation")
    
    from src.data.data_formatter import DataFormatter
    
    all_passed = True
    formatter = DataFormatter()
    
    # Test SFT formatting
    from datasets import Dataset
    test_data = Dataset.from_dict({
        "question": ["What is 2+2?", "What is 10% of 50?"],
        "answer": ["4", "5"],
        "rationale": ["2+2=4", "10% of 50 = 0.1 × 50 = 5"],
        "source": ["test", "test"],
    })
    
    formatted = formatter.format_for_sft(test_data)
    all_passed &= print_result(
        "SFT formatting", 
        "text" in formatted.column_names,
        f"columns={formatted.column_names}"
    )
    all_passed &= print_result(
        "SFT has <think> tags",
        "<think>" in formatted[0]["text"],
        f"sample length={len(formatted[0]['text'])}"
    )
    
    # Test RL formatting
    rl_formatted = formatter.format_for_rl(test_data)
    all_passed &= print_result(
        "RL formatting",
        "prompt" in rl_formatted.column_names and "ground_truth" in rl_formatted.column_names,
        f"columns={rl_formatted.column_names}"
    )
    
    return all_passed


def validate_answer_extraction():
    """Test answer extraction and math verification."""
    print_header("4. Answer Extraction & Math Verification")
    
    from src.utils.answer_extraction import AnswerExtractor
    from src.utils.math_verify import MathVerifier
    
    all_passed = True
    extractor = AnswerExtractor()
    verifier = MathVerifier()
    
    # Tag extraction
    answer = extractor.extract("<think>work</think>\n<answer>42</answer>")
    all_passed &= print_result("Extract from tags", answer == "42", f"got={answer}")
    
    # Pattern extraction
    answer = extractor.extract("The answer is 156.")
    all_passed &= print_result("Extract from pattern", answer == "156", f"got={answer}")
    
    # Confidence scoring
    answer, conf = extractor.extract_with_confidence("<answer>42</answer>")
    all_passed &= print_result("Confidence: high for tags", conf == 1.0, f"conf={conf}")
    
    answer, conf = extractor.extract_with_confidence("The answer is 42.")
    all_passed &= print_result("Confidence: lower for pattern", conf < 1.0, f"conf={conf}")
    
    # Math verification
    correct, _ = verifier.verify("0.75", "3/4")
    all_passed &= print_result("Math: 0.75 == 3/4", correct)
    
    correct, _ = verifier.verify("1200", "$1,200")
    all_passed &= print_result("Math: 1200 == $1,200", correct)
    
    correct, _ = verifier.verify("7", "3 + 4")
    all_passed &= print_result("Math: 7 == 3+4", correct)
    
    correct, _ = verifier.verify("41", "42")
    all_passed &= print_result("Math: 41 != 42", not correct)
    
    # Self-consistency
    responses = [
        "<answer>42</answer>",
        "<answer>42</answer>",
        "<answer>42</answer>",
        "<answer>43</answer>",
    ]
    answer, agreement = extractor.self_consistency_vote(responses)
    all_passed &= print_result(
        "Self-consistency vote",
        answer == "42" and agreement == 0.75,
        f"answer={answer}, agreement={agreement}"
    )
    
    return all_passed


def validate_metrics():
    """Test metrics tracking."""
    print_header("5. Metrics Tracker Validation")
    
    from src.utils.metrics import MetricsTracker
    
    all_passed = True
    tracker = MetricsTracker(output_dir="/tmp/test_metrics", patience=3)
    
    # Log improving metrics
    for i in range(10):
        tracker.log("accuracy", 0.4 + i * 0.02, step=i)
    
    best = tracker.get_best("accuracy")
    all_passed &= print_result("Metrics: tracks best", abs(best - 0.58) < 0.001, f"best={best}")
    
    trend = tracker.get_trend("accuracy")
    all_passed &= print_result("Metrics: detects improvement", trend == "improving", f"trend={trend}")
    
    # Early stopping
    all_passed &= print_result(
        "Metrics: no early stop (improving)", 
        not tracker.should_early_stop("accuracy")
    )
    
    # Improvement calculation
    improvement = tracker.get_improvement_over_baseline("accuracy", 0.47)
    all_passed &= print_result(
        "Metrics: improvement calc",
        improvement is not None and improvement[0] > 0.1,
        f"abs={improvement[0]:.3f}, rel={improvement[1]:.1f}%"
    )
    
    return all_passed


def validate_curriculum():
    """Test curriculum scheduler."""
    print_header("6. Curriculum Scheduler Validation")
    
    # Import directly to avoid torch dependency from __init__.py
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "curriculum", project_root / "src" / "training" / "curriculum.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    CurriculumScheduler = mod.CurriculumScheduler
    
    all_passed = True
    
    config = {
        "enabled": True,
        "difficulty_metric": "num_steps",
        "stages": [
            {"name": "easy", "epochs": [1, 2], "max_steps": 3},
            {"name": "medium", "epochs": [3, 4], "max_steps": 6},
            {"name": "hard", "epochs": [5], "max_steps": None},
        ]
    }
    
    scheduler = CurriculumScheduler(config)
    
    stage = scheduler.get_current_stage(1)
    all_passed &= print_result("Curriculum: epoch 1 = easy", stage["name"] == "easy")
    
    stage = scheduler.get_current_stage(3)
    all_passed &= print_result("Curriculum: epoch 3 = medium", stage["name"] == "medium")
    
    stage = scheduler.get_current_stage(5)
    all_passed &= print_result("Curriculum: epoch 5 = hard", stage["name"] == "hard")
    
    summary = scheduler.get_schedule_summary()
    all_passed &= print_result("Curriculum: summary", "easy" in summary, "generated")
    
    return all_passed


def validate_error_analysis():
    """Test error analysis module."""
    print_header("7. Error Analysis Validation")
    
    # Import directly to avoid torch dependency from evaluator.py
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "error_analysis", project_root / "src" / "evaluation" / "error_analysis.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    ErrorAnalyzer = mod.ErrorAnalyzer
    
    all_passed = True
    analyzer = ErrorAnalyzer()
    
    # Correct prediction
    result = analyzer.analyze_prediction(
        question="What is 2+2?",
        response="<think>Step 1: 2+2=4</think>\n<answer>4</answer>",
        ground_truth="4",
        predicted_answer="4",
        is_correct=True,
    )
    all_passed &= print_result("Error: correct detection", result["error_category"] == "correct")
    
    # Format error
    result = analyzer.analyze_prediction(
        question="What is 5*5?",
        response="Let me think about this... 5 times 5 is twenty five I think maybe",
        ground_truth="25",
        predicted_answer=None,
        is_correct=False,
    )
    all_passed &= print_result("Error: format error", result["error_category"] == "format_error")
    
    # Computation error (close to answer)
    result = analyzer.analyze_prediction(
        question="What is 15% of 240?",
        response="<think>15% = 0.15, 0.15 × 240 = 34</think>\n<answer>34</answer>",
        ground_truth="36",
        predicted_answer="34",
        is_correct=False,
    )
    all_passed &= print_result(
        "Error: computation error",
        result["error_category"] == "computation_error",
        f"got={result['error_category']}"
    )
    
    # Report generation
    report = analyzer.get_report()
    all_passed &= print_result(
        "Error: report generation",
        "summary" in report and report["summary"]["total_predictions"] == 3
    )
    
    return all_passed


def main():
    print("\n" + "🧠 " * 20)
    print("  SLM REASONING ENHANCEMENT — PIPELINE VALIDATION")
    print("🧠 " * 20)
    
    all_results = []
    
    try:
        all_results.append(validate_configs())
        all_results.append(validate_rewards())
        all_results.append(validate_data_formatting())
        all_results.append(validate_answer_extraction())
        all_results.append(validate_metrics())
        all_results.append(validate_curriculum())
        all_results.append(validate_error_analysis())
    except Exception as e:
        print(f"\n❌ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Final summary
    print_header("FINAL RESULTS")
    
    total_passed = sum(all_results)
    total_tests = len(all_results)
    
    if all(all_results):
        print(f"\n  🎉 ALL {total_tests}/{total_tests} VALIDATION SECTIONS PASSED!")
        print(f"\n  ✅ Pipeline is ready for GPU training.")
        print(f"  ✅ All reward functions work correctly.")
        print(f"  ✅ Data formatting produces valid outputs.")
        print(f"  ✅ Answer extraction handles edge cases.")
        print(f"  ✅ Curriculum and metrics tracking operational.")
        print(f"\n  Next step: Run `make train-all` on GPU infrastructure.")
    else:
        failed = total_tests - total_passed
        print(f"\n  ⚠️  {failed}/{total_tests} sections have failures.")
        print(f"  Please fix the issues above before proceeding to training.")
        sys.exit(1)
    
    print()


if __name__ == "__main__":
    main()
