# ==============================================================================
# SLM Reasoning Enhancement - Makefile
# ==============================================================================
# Quick commands for training, evaluation, and demo

.PHONY: help install train-sft train-grpo train-all eval demo test clean

# Default target
help:
	@echo "╔══════════════════════════════════════════════════════════════╗"
	@echo "║  🧠 SLM Reasoning Enhancement via RL                       ║"
	@echo "╠══════════════════════════════════════════════════════════════╣"
	@echo "║                                                            ║"
	@echo "║  Setup:                                                    ║"
	@echo "║    make install        Install all dependencies            ║"
	@echo "║                                                            ║"
	@echo "║  Training:                                                 ║"
	@echo "║    make train-sft      Phase 1: Supervised Fine-Tuning     ║"
	@echo "║    make train-grpo     Phase 2: GRPO RL Training           ║"
	@echo "║    make train-all      Full pipeline (SFT → GRPO → Eval)  ║"
	@echo "║                                                            ║"
	@echo "║  Evaluation:                                               ║"
	@echo "║    make eval           Run all benchmarks                  ║"
	@echo "║    make eval-baseline  Evaluate base model (no training)   ║"
	@echo "║                                                            ║"
	@echo "║  Demo:                                                     ║"
	@echo "║    make demo           Interactive demo mode               ║"
	@echo "║    make demo-sc        Demo with self-consistency          ║"
	@echo "║                                                            ║"
	@echo "║  Testing:                                                  ║"
	@echo "║    make test           Run unit tests                      ║"
	@echo "║    make test-rewards   Test reward functions               ║"
	@echo "║                                                            ║"
	@echo "║  Utilities:                                                ║"
	@echo "║    make clean          Remove output artifacts             ║"
	@echo "║    make check-gpu      Check GPU availability              ║"
	@echo "║                                                            ║"
	@echo "╚══════════════════════════════════════════════════════════════╝"

# ==============================================================================
# Setup
# ==============================================================================

install:
	pip install -r requirements.txt
	@echo "✅ Dependencies installed"

install-dev:
	pip install -r requirements.txt
	pip install pytest pytest-cov black isort flake8
	@echo "✅ Dev dependencies installed"

# ==============================================================================
# Training
# ==============================================================================

train-sft:
	@echo "🏋️ Phase 1: Supervised Fine-Tuning..."
	python scripts/run_full_pipeline.py --phase sft --config-dir configs/

train-grpo:
	@echo "🎯 Phase 2: GRPO Reinforcement Learning..."
	python scripts/run_full_pipeline.py --phase grpo --config-dir configs/

train-all:
	@echo "🚀 Running full pipeline: SFT → GRPO → Evaluation..."
	python scripts/run_full_pipeline.py --phase all --config-dir configs/

# Resume from checkpoints
train-grpo-resume:
	python scripts/run_full_pipeline.py --phase grpo --config-dir configs/ --sft-model ./outputs/sft/final

# ==============================================================================
# Evaluation
# ==============================================================================

eval:
	@echo "📊 Evaluating trained model..."
	python scripts/run_full_pipeline.py --phase eval --config-dir configs/

eval-baseline:
	@echo "📊 Evaluating baseline (no training)..."
	python scripts/run_full_pipeline.py --phase eval --config-dir configs/ \
		--grpo-model microsoft/Phi-3-mini-4k-instruct

eval-sft:
	@echo "📊 Evaluating SFT model..."
	python scripts/run_full_pipeline.py --phase eval --config-dir configs/ \
		--grpo-model ./outputs/sft/final

# ==============================================================================
# Demo / Inference
# ==============================================================================

MODEL_PATH ?= ./outputs/grpo/final

demo:
	@echo "🎮 Starting interactive demo..."
	python scripts/demo_inference.py --model $(MODEL_PATH)

demo-sc:
	@echo "🎮 Starting demo with self-consistency (k=8)..."
	python scripts/demo_inference.py --model $(MODEL_PATH) --self-consistency --num-samples 8

demo-question:
	@echo "Asking: $(Q)"
	python scripts/demo_inference.py --model $(MODEL_PATH) --question "$(Q)"

# ==============================================================================
# Testing
# ==============================================================================

test:
	python -m pytest tests/ -v --tb=short

test-rewards:
	python -m pytest tests/test_rewards.py -v

test-coverage:
	python -m pytest tests/ --cov=src --cov-report=html
	@echo "📊 Coverage report: htmlcov/index.html"

# ==============================================================================
# Utilities
# ==============================================================================

check-gpu:
	@python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); \
	print(f'GPU count: {torch.cuda.device_count()}'); \
	[print(f'  GPU {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_mem / 1e9:.1f} GB)') for i in range(torch.cuda.device_count())]"

clean:
	rm -rf outputs/
	rm -rf __pycache__ src/__pycache__ src/**/__pycache__
	rm -rf .pytest_cache
	rm -f training.log
	@echo "🧹 Cleaned output artifacts"

format:
	black src/ scripts/ tests/
	isort src/ scripts/ tests/

lint:
	flake8 src/ scripts/ --max-line-length 100 --ignore E501,W503
