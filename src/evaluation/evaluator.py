"""
Evaluator: Runs benchmarks (GSM8K, MMLU, StrategyQA) on trained models.
Supports baseline comparison, latency measurement, and detailed reporting.
"""

import json
import logging
import os
import time
from typing import Dict, List, Optional

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from ..rewards.outcome_reward import OutcomeReward

logger = logging.getLogger(__name__)


class Evaluator:
    """
    Evaluates model performance on reasoning benchmarks.
    
    Supports:
    - GSM8K (math reasoning)
    - MMLU (multitask understanding)
    - StrategyQA (yes/no reasoning)
    """

    def __init__(self, config: Dict):
        """
        Initialize evaluator.

        Args:
            config: Evaluation configuration dictionary.
        """
        self.config = config
        self.model_config = config["model"]
        self.eval_config = config["evaluation"]
        self.reporting_config = config.get("reporting", {})
        
        self.model = None
        self.tokenizer = None
        self.outcome_reward = OutcomeReward()

    def setup(self, model_path: Optional[str] = None):
        """
        Load model and tokenizer for evaluation.

        Args:
            model_path: Override model path (for evaluating different checkpoints).
        """
        path = model_path or self.model_config.get("checkpoint", self.model_config["name"])
        
        logger.info(f"Loading model for evaluation: {path}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            path if os.path.exists(path) else self.model_config["name"],
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(
            self.model_config.get("dtype", "bfloat16"), torch.bfloat16
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            device_map="auto",
        )
        self.model.eval()
        logger.info("Model loaded for evaluation.")

    def evaluate_all(self, datasets: Dict[str, Dataset]) -> Dict[str, Dict]:
        """
        Run all configured benchmarks.

        Args:
            datasets: Dict mapping benchmark name to eval dataset.

        Returns:
            Dict of benchmark results.
        """
        results = {}
        
        for benchmark_cfg in self.eval_config["benchmarks"]:
            name = benchmark_cfg["name"]
            if name not in datasets:
                logger.warning(f"Dataset for '{name}' not provided, skipping.")
                continue
            
            logger.info(f"Evaluating on {name}...")
            result = self.evaluate_benchmark(
                dataset=datasets[name],
                benchmark_name=name,
                config=benchmark_cfg,
            )
            results[name] = result
            logger.info(f"  {name}: {result['accuracy']:.4f} ({result['correct']}/{result['total']})")

        # Measure latency if configured
        if self.reporting_config.get("measure_latency", False):
            latency = self._measure_latency(datasets)
            results["latency"] = latency

        # Save results
        if self.reporting_config.get("save_metrics", True):
            self._save_results(results)

        return results

    def evaluate_benchmark(
        self,
        dataset: Dataset,
        benchmark_name: str,
        config: Dict,
    ) -> Dict:
        """
        Evaluate on a single benchmark.

        Args:
            dataset: Evaluation dataset with 'prompt' and 'ground_truth'.
            benchmark_name: Name of the benchmark.
            config: Benchmark-specific config.

        Returns:
            Dict with accuracy, correct count, total, and predictions.
        """
        gen_config = self.eval_config.get("generation", {})
        batch_size = self.eval_config.get("batch_size", 16)

        correct = 0
        total = 0
        predictions = []

        # Process in batches
        for i in tqdm(range(0, len(dataset), batch_size), desc=f"Eval {benchmark_name}"):
            batch = dataset[i : i + batch_size]
            prompts = batch["prompt"]
            ground_truths = batch["ground_truth"]

            # Generate responses
            responses = self._generate_batch(prompts, gen_config)

            # Score responses
            for prompt, response, gt in zip(prompts, responses, ground_truths):
                reward = self.outcome_reward.compute(response, gt)
                is_correct = reward >= 1.0

                if is_correct:
                    correct += 1
                total += 1

                predictions.append({
                    "prompt": prompt[:200],  # Truncate for storage
                    "response": response,
                    "ground_truth": gt,
                    "extracted_answer": self.outcome_reward.extract_answer(response),
                    "is_correct": is_correct,
                    "reward": reward,
                })

        accuracy = correct / total if total > 0 else 0.0

        return {
            "benchmark": benchmark_name,
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "predictions": predictions if self.reporting_config.get("save_predictions", False) else [],
        }

    def _generate_batch(self, prompts: List[str], gen_config: Dict) -> List[str]:
        """Generate responses for a batch of prompts."""
        responses = []
        
        for prompt in prompts:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048,
            ).to(self.model.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=gen_config.get("max_new_tokens", 512),
                    temperature=gen_config.get("temperature", 0.0) or 1.0,
                    top_p=gen_config.get("top_p", 1.0),
                    do_sample=gen_config.get("do_sample", False),
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            # Decode only the generated tokens
            generated = outputs[0][inputs["input_ids"].shape[1]:]
            response = self.tokenizer.decode(generated, skip_special_tokens=True)
            responses.append(response)

        return responses

    def _measure_latency(self, datasets: Dict[str, Dataset]) -> Dict:
        """Measure inference latency."""
        num_samples = self.reporting_config.get("latency_samples", 100)
        warmup = self.reporting_config.get("latency_warmup", 10)

        # Use first available dataset
        dataset = next(iter(datasets.values()))
        prompts = dataset["prompt"][:num_samples + warmup]

        # Warmup
        for prompt in prompts[:warmup]:
            self._generate_batch([prompt], {"max_new_tokens": 256})

        # Measure
        latencies = []
        for prompt in prompts[warmup:]:
            start = time.time()
            self._generate_batch([prompt], {"max_new_tokens": 256})
            latencies.append(time.time() - start)

        avg_latency = sum(latencies) / len(latencies)
        p50 = sorted(latencies)[len(latencies) // 2]
        p95 = sorted(latencies)[int(len(latencies) * 0.95)]

        result = {
            "avg_latency_s": avg_latency,
            "p50_latency_s": p50,
            "p95_latency_s": p95,
            "num_samples": len(latencies),
        }

        logger.info(
            f"Latency: avg={avg_latency:.3f}s, p50={p50:.3f}s, p95={p95:.3f}s"
        )
        return result

    def _save_results(self, results: Dict):
        """Save evaluation results to disk."""
        output_dir = self.reporting_config.get("output_dir", "./outputs/evaluation")
        os.makedirs(output_dir, exist_ok=True)

        # Save summary
        summary = {}
        for name, result in results.items():
            if name == "latency":
                summary["latency"] = result
            else:
                summary[name] = {
                    "accuracy": result["accuracy"],
                    "correct": result["correct"],
                    "total": result["total"],
                }

        summary_path = os.path.join(output_dir, "results_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Results saved to {summary_path}")

        # Save detailed predictions if configured
        if self.reporting_config.get("save_predictions", True):
            for name, result in results.items():
                if name == "latency":
                    continue
                pred_path = os.path.join(output_dir, f"{name}_predictions.json")
                with open(pred_path, "w") as f:
                    json.dump(result.get("predictions", []), f, indent=2)

    def compare_models(self, model_paths: List[Dict], datasets: Dict[str, Dataset]) -> Dict:
        """
        Compare multiple models on the same benchmarks.

        Args:
            model_paths: List of {"name": "...", "label": "..."} model configs.
            datasets: Evaluation datasets.

        Returns:
            Comparative results dict.
        """
        all_results = {}

        for model_info in model_paths:
            label = model_info["label"]
            path = model_info["name"]
            
            logger.info(f"\n{'='*60}")
            logger.info(f"Evaluating: {label}")
            logger.info(f"{'='*60}")

            self.setup(model_path=path)
            results = self.evaluate_all(datasets)
            all_results[label] = results

        # Print comparison table
        self._print_comparison(all_results)
        return all_results

    def _print_comparison(self, all_results: Dict):
        """Print a comparison table of results."""
        benchmarks = set()
        for results in all_results.values():
            benchmarks.update(k for k in results if k != "latency")

        header = f"{'Model':<30} | " + " | ".join(f"{b:<12}" for b in sorted(benchmarks))
        logger.info(f"\n{'='*len(header)}")
        logger.info(header)
        logger.info(f"{'-'*len(header)}")

        for model_label, results in all_results.items():
            scores = []
            for b in sorted(benchmarks):
                if b in results:
                    scores.append(f"{results[b]['accuracy']*100:.1f}%")
                else:
                    scores.append("N/A")
            row = f"{model_label:<30} | " + " | ".join(f"{s:<12}" for s in scores)
            logger.info(row)

        logger.info(f"{'='*len(header)}\n")
