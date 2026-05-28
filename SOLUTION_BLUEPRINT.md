# Solution Blueprint: Enhancing Reasoning in SLMs using Reinforcement Learning

## Executive Summary

We propose a **GRPO-based (Group Relative Policy Optimization)** reinforcement learning approach combined with a hybrid SFT+RL pipeline to enhance reasoning capabilities in Small Language Models. Our approach eliminates the need for a separate reward model, uses curriculum-based training, and incorporates both outcome and process-based rewards.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     TRAINING PIPELINE                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Phase 1: SFT Warm-up          Phase 2: RL Fine-tuning           │
│  ┌──────────────────┐          ┌──────────────────────────┐     │
│  │ Base Model        │          │ SFT Model                 │     │
│  │ (Phi-3-Mini 3.8B) │──────▶  │ + GRPO Training           │     │
│  │                    │          │ + Curriculum Learning      │     │
│  │ + CoT SFT Data    │          │ + Process + Outcome Reward │     │
│  └──────────────────┘          └──────────────────────────┘     │
│                                          │                        │
│                                          ▼                        │
│                              ┌──────────────────────┐            │
│                              │  Final RL-Tuned Model  │            │
│                              └──────────────────────┘            │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                     REWARD SYSTEM                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌─────────────────┐    ┌──────────────────┐                    │
│  │ Outcome Reward   │    │ Process Reward    │                    │
│  │ (Correctness)    │    │ (Reasoning Steps) │                    │
│  │                   │    │                    │                    │
│  │ • Answer match    │    │ • Step validity    │                    │
│  │ • Partial credit  │    │ • CoT format       │                    │
│  │                   │    │ • Self-correction  │                    │
│  └────────┬──────────┘    └────────┬───────────┘                    │
│           │                         │                              │
│           └────────┬────────────────┘                              │
│                    ▼                                                │
│           ┌────────────────┐                                       │
│           │ Combined Reward │                                       │
│           │ R = α·R_out +   │                                       │
│           │     β·R_proc    │                                       │
│           └────────────────┘                                       │
└─────────────────────────────────────────────────────────────────┘
```

---

## Technical Approach

### 1. Model Selection: Phi-3-Mini (3.8B)

**Why Phi-3-Mini?**
- 3.8B parameters — lightweight, fits dual-GPU setup easily
- Strong baseline reasoning capability for its size
- Well-documented, active community
- Allows more RL training iterations within compute budget
- Can demonstrate larger relative improvement (easier to show +5%)

**Secondary Model**: Qwen 2.5 7B (for comparison/ablation)

### 2. Phase 1: Supervised Fine-Tuning (SFT)

**Goal**: Teach the model Chain-of-Thought (CoT) reasoning format

**Data Preparation**:
- GSM8K training set with step-by-step solutions
- AQuA-RAT with rationales
- Reformatted into structured CoT format:
  ```
  <think>
  Step 1: [reasoning step]
  Step 2: [reasoning step]
  ...
  </think>
  <answer>[final answer]</answer>
  ```

**Training Config**:
- LoRA (rank=64, alpha=128) for parameter efficiency
- Learning rate: 2e-5
- Epochs: 3
- Batch size: 4 (gradient accumulation: 8)

### 3. Phase 2: GRPO Reinforcement Learning

**Why GRPO over PPO?**
- No separate reward model needed (saves compute)
- Uses group-relative scoring — generates multiple responses, ranks them
- More stable for small models
- Proven effective (DeepSeek-R1 approach)

**GRPO Algorithm**:
1. For each prompt, generate G completions from the policy
2. Score each completion with reward function
3. Compute advantages relative to the group mean
4. Update policy using clipped surrogate objective

**Key Hyperparameters**:
- Group size (G): 8 completions per prompt
- KL coefficient (β): 0.04 (starts low, anneals up)
- Clip ratio (ε): 0.2
- Temperature: 0.7 for generation
- Max new tokens: 512

### 4. Reward Design (Hybrid)

#### Outcome Reward (R_outcome)
```python
def outcome_reward(response, ground_truth):
    extracted_answer = extract_answer(response)
    if exact_match(extracted_answer, ground_truth):
        return 1.0
    elif numeric_close(extracted_answer, ground_truth, tol=0.01):
        return 0.8
    else:
        return 0.0
```

#### Process Reward (R_process)
```python
def process_reward(response):
    score = 0.0
    # Format compliance
    if has_think_tags(response):
        score += 0.2
    # Step count (reward multi-step reasoning)
    steps = count_steps(response)
    score += min(steps * 0.05, 0.3)
    # No repetition penalty
    if not has_repetition(response):
        score += 0.2
    # Has final answer tag
    if has_answer_tag(response):
        score += 0.1
    # Length penalty (discourage overly verbose)
    score -= max(0, (len(response) - 1000) * 0.0001)
    return min(score, 0.5)
```

#### Combined Reward
```
R_total = 0.7 * R_outcome + 0.3 * R_process
```

### 5. Stability Techniques

- **KL Divergence Penalty**: Prevents catastrophic forgetting, annealed from 0.02 → 0.1
- **Entropy Bonus**: Maintains exploration, coefficient 0.01
- **Curriculum Learning**: 
  - Epoch 1-2: Easy problems (1-3 step solutions)
  - Epoch 3-4: Medium (4-6 steps)
  - Epoch 5+: Hard (7+ steps)
- **Gradient Clipping**: max_norm = 1.0
- **Warmup Steps**: 100 steps with linear warmup

### 6. Evaluation Plan

| Benchmark | Evaluation Method | Metric |
|-----------|------------------|--------|
| GSM8K | 8-shot CoT | Accuracy |
| MMLU | 5-shot | Accuracy |
| StrategyQA | 0-shot CoT | Accuracy |

**Baseline**: Phi-3-Mini without any fine-tuning
**Target**: ≥ +5% improvement on at least 2/3 benchmarks

---

## Project Structure

```
hackathon/
├── PROBLEM_STATEMENT.md
├── SOLUTION_BLUEPRINT.md
├── README.md
├── requirements.txt
├── configs/
│   ├── sft_config.yaml
│   ├── grpo_config.yaml
│   └── eval_config.yaml
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset_loader.py      # Load GSM8K, AQuA-RAT, MMLU
│   │   └── data_formatter.py      # Format into CoT structure
│   ├── rewards/
│   │   ├── __init__.py
│   │   ├── outcome_reward.py      # Answer correctness reward
│   │   ├── process_reward.py      # Reasoning quality reward
│   │   └── combined_reward.py     # Weighted combination
│   ├── training/
│   │   ├── __init__.py
│   │   ├── sft_trainer.py         # Phase 1: SFT training
│   │   ├── grpo_trainer.py        # Phase 2: GRPO RL training
│   │   └── curriculum.py          # Curriculum learning scheduler
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── evaluator.py           # Benchmark evaluation
│   │   ├── gsm8k_eval.py          # GSM8K-specific evaluation
│   │   ├── mmlu_eval.py           # MMLU-specific evaluation
│   │   └── strategyqa_eval.py     # StrategyQA-specific evaluation
│   └── utils/
│       ├── __init__.py
│       ├── answer_extraction.py   # Extract answers from model output
│       └── metrics.py             # Accuracy, latency metrics
├── scripts/
│   ├── run_sft.py                 # Run SFT training
│   ├── run_grpo.py                # Run GRPO training
│   ├── run_eval.py                # Run evaluation
│   └── run_full_pipeline.py       # End-to-end pipeline
└── notebooks/
    └── analysis.ipynb             # Results analysis
```

---

## Technology Stack

| Component | Technology |
|-----------|-----------|
| Base Framework | PyTorch 2.x |
| Model Loading | HuggingFace Transformers |
| RL Training | TRL (Transformer Reinforcement Learning) |
| Parameter Efficient | PEFT (LoRA) |
| Data | HuggingFace Datasets |
| Evaluation | lm-evaluation-harness |
| Experiment Tracking | Weights & Biases (wandb) |
| Config Management | YAML + dataclasses |

---

## Timeline & Milestones

| Week | Milestone |
|------|-----------|
| Week 1 | Data preparation + SFT baseline |
| Week 2 | GRPO implementation + reward design |
| Week 3 | Training + hyperparameter tuning |
| Week 4 | Evaluation + ablation studies |
| Week 5 | Documentation + presentation prep |

---

## Expected Results

| Benchmark | Baseline (Phi-3-Mini) | Target After RL |
|-----------|----------------------|-----------------|
| GSM8K | ~45-48% | ≥ 53% (+5%) |
| MMLU | ~42-44% | ≥ 47% (+5%) |
| StrategyQA | ~60-63% | ≥ 68% (+5%) |

---

## Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| Reward hacking | Process reward + format constraints |
| Mode collapse | KL penalty + entropy bonus |
| Training instability | Curriculum learning + gradient clipping |
| Compute constraints | LoRA + small batch with accumulation |
| Catastrophic forgetting | KL divergence from reference model |
