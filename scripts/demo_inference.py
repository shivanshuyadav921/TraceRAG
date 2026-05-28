"""
Interactive Demo / Inference Script
====================================

Usage:
    # Interactive mode
    python scripts/demo_inference.py --model ./outputs/grpo/final

    # Single question
    python scripts/demo_inference.py --model ./outputs/grpo/final --question "What is 15% of 240?"

    # With self-consistency (majority voting)
    python scripts/demo_inference.py --model ./outputs/grpo/final --self-consistency --num-samples 8

    # Batch inference from file
    python scripts/demo_inference.py --model ./outputs/grpo/final --input questions.txt --output answers.json

This script provides a clean interface for demonstrating the trained model's
reasoning capabilities during hackathon presentations.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.answer_extraction import AnswerExtractor

console = Console()
logger = logging.getLogger(__name__)


class ReasoningDemo:
    """Interactive demonstration of the enhanced reasoning model."""

    SYSTEM_PROMPT = (
        "You are a helpful assistant that solves problems step by step. "
        "Show your reasoning in <think> tags and give your final answer in <answer> tags."
    )

    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        dtype: str = "bfloat16",
    ):
        self.model_path = model_path
        self.extractor = AnswerExtractor()
        
        # Load model
        console.print(f"[bold blue]Loading model from:[/] {model_path}")
        
        dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
        torch_dtype = dtype_map.get(dtype, torch.bfloat16)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            device_map=device,
        )
        self.model.eval()
        console.print("[bold green]✓ Model loaded successfully![/]\n")

    def generate(
        self,
        question: str,
        temperature: float = 0.0,
        max_new_tokens: int = 512,
        show_reasoning: bool = True,
    ) -> dict:
        """
        Generate a response for a single question.

        Returns:
            Dict with response, answer, reasoning, latency.
        """
        # Build prompt
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]

        try:
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            prompt = f"### System:\n{self.SYSTEM_PROMPT}\n\n### User:\n{question}\n\n### Assistant:\n"

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        # Generate
        start_time = time.time()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if temperature > 0 else 1.0,
                do_sample=temperature > 0,
                top_p=0.95 if temperature > 0 else 1.0,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        latency = time.time() - start_time

        # Decode
        generated = outputs[0][inputs["input_ids"].shape[1]:]
        response = self.tokenizer.decode(generated, skip_special_tokens=True)

        # Extract answer
        answer, confidence = self.extractor.extract_with_confidence(response)

        return {
            "question": question,
            "response": response,
            "answer": answer,
            "confidence": confidence,
            "latency_s": latency,
            "tokens_generated": len(generated),
        }

    def generate_with_self_consistency(
        self,
        question: str,
        num_samples: int = 8,
        temperature: float = 0.7,
        max_new_tokens: int = 512,
    ) -> dict:
        """
        Generate with self-consistency (majority voting).
        
        Samples multiple responses and picks the most common answer.
        """
        responses = []
        for i in range(num_samples):
            result = self.generate(
                question, temperature=temperature, max_new_tokens=max_new_tokens
            )
            responses.append(result["response"])

        # Majority vote
        majority_answer, agreement = self.extractor.self_consistency_vote(
            responses, temperature=temperature, num_samples=num_samples
        )

        return {
            "question": question,
            "answer": majority_answer,
            "agreement_ratio": agreement,
            "num_samples": num_samples,
            "all_responses": responses,
        }

    def display_result(self, result: dict):
        """Pretty-print a result for presentation."""
        console.print(Panel(result["question"], title="❓ Question", border_style="blue"))

        response = result["response"]
        
        # Highlight reasoning vs answer
        import re
        think_match = re.search(r'<think>(.*?)</think>', response, re.DOTALL)
        answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)

        if think_match:
            reasoning = think_match.group(1).strip()
            console.print(Panel(
                reasoning,
                title="💭 Reasoning",
                border_style="yellow",
            ))

        if answer_match:
            answer = answer_match.group(1).strip()
            console.print(Panel(
                f"[bold green]{answer}[/]",
                title="✅ Answer",
                border_style="green",
            ))
        elif result.get("answer"):
            console.print(Panel(
                f"[bold green]{result['answer']}[/]",
                title="✅ Answer (extracted)",
                border_style="green",
            ))

        # Metadata
        latency = result.get("latency_s", 0)
        tokens = result.get("tokens_generated", 0)
        confidence = result.get("confidence", 0)
        console.print(
            f"  [dim]⏱️ {latency:.2f}s | 🎯 Confidence: {confidence:.0%} | "
            f"📝 {tokens} tokens[/]\n"
        )

    def interactive_mode(self, use_self_consistency: bool = False, num_samples: int = 8):
        """Run in interactive mode for live demos."""
        console.print(Panel(
            "[bold]🧠 SLM Reasoning Enhancement Demo[/]\n\n"
            "Enter math/reasoning questions and see the model think step-by-step.\n"
            "Type 'quit' or 'exit' to stop.\n"
            f"Mode: {'Self-Consistency (k=' + str(num_samples) + ')' if use_self_consistency else 'Greedy'}",
            border_style="bold blue",
        ))

        while True:
            try:
                console.print("\n[bold cyan]Enter your question:[/]")
                question = input("> ").strip()

                if question.lower() in ("quit", "exit", "q"):
                    console.print("[bold]Goodbye! 👋[/]")
                    break

                if not question:
                    continue

                if use_self_consistency:
                    console.print(f"[dim]Generating {num_samples} samples...[/]")
                    result = self.generate_with_self_consistency(
                        question, num_samples=num_samples
                    )
                    console.print(Panel(
                        f"[bold green]{result['answer']}[/]\n"
                        f"[dim]Agreement: {result['agreement_ratio']:.0%} across {num_samples} samples[/]",
                        title="✅ Self-Consistency Answer",
                        border_style="green",
                    ))
                else:
                    result = self.generate(question)
                    self.display_result(result)

            except KeyboardInterrupt:
                console.print("\n[bold]Interrupted. Goodbye! 👋[/]")
                break
            except Exception as e:
                console.print(f"[red]Error: {e}[/]")


def main():
    parser = argparse.ArgumentParser(description="SLM Reasoning Demo")
    parser.add_argument(
        "--model", type=str, required=True,
        help="Path to the trained model",
    )
    parser.add_argument(
        "--question", type=str, default=None,
        help="Single question to answer (otherwise enters interactive mode)",
    )
    parser.add_argument(
        "--input", type=str, default=None,
        help="File with questions (one per line)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output JSON file for batch results",
    )
    parser.add_argument(
        "--self-consistency", action="store_true",
        help="Use self-consistency (majority voting)",
    )
    parser.add_argument(
        "--num-samples", type=int, default=8,
        help="Number of samples for self-consistency",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0,
        help="Generation temperature (0 = greedy)",
    )
    parser.add_argument(
        "--dtype", type=str, default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
    )
    args = parser.parse_args()

    demo = ReasoningDemo(model_path=args.model, dtype=args.dtype)

    if args.question:
        # Single question mode
        if args.self_consistency:
            result = demo.generate_with_self_consistency(
                args.question, num_samples=args.num_samples
            )
            console.print(Panel(
                f"Answer: {result['answer']}\n"
                f"Agreement: {result['agreement_ratio']:.0%}",
                title="Self-Consistency Result",
            ))
        else:
            result = demo.generate(args.question, temperature=args.temperature)
            demo.display_result(result)

    elif args.input:
        # Batch mode
        with open(args.input) as f:
            questions = [line.strip() for line in f if line.strip()]

        results = []
        for q in questions:
            console.print(f"[dim]Processing: {q[:60]}...[/]")
            result = demo.generate(q, temperature=args.temperature)
            results.append(result)
            demo.display_result(result)

        if args.output:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
            console.print(f"[green]Results saved to {args.output}[/]")

    else:
        # Interactive mode
        demo.interactive_mode(
            use_self_consistency=args.self_consistency,
            num_samples=args.num_samples,
        )


if __name__ == "__main__":
    main()
