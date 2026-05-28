# Problem Statement #06: Enhancing Reasoning in Small Language Models (SLMs) using Reinforcement Learning

**Hackathon:** Samsung ennovateX™ AX Hackathon 2026  
**Source:** https://ennovatex.io/ax-hackathon/

---

## Problem Statement

Large Language Models (LLMs) demonstrate strong reasoning across domains such as mathematics, logical inference, and multi-step decision-making. However, their deployment is often constrained by latency, cost, and hardware limitations, especially in on-device or real-time applications.

Small Language Models (SLMs) (≤ 7B parameters) offer a promising alternative due to their efficiency, but they significantly lag behind in:

- **Multi-step reasoning** (Limited capacity → weak reasoning chains)
- **Logical consistency**
- **Planning and self-correction**

Reinforcement Learning (RL) has shown strong improvements in reasoning for large models, but directly applying these techniques to SLMs is ineffective without redesigning the pipeline. Also, SLMs have high sensitivity to reward design.

---

## Key Expectations & Deliverables

### Directions to Explore

- Outcome + process-based reward design
- Lightweight / efficient reward mechanisms
- Stability techniques (KL tuning, entropy, curriculum)
- Hybrid pipelines (SFT + RL, distillation + RL)

### Deliverables

1. A RL-based training code tailored for SLMs that improves reasoning while maintaining efficiency.
2. A final model with inference code for evaluation.
3. Detailed approach, results, and insights gained.

---

## Definitive Target KPIs & Benchmarks

| Benchmark   | Minimum Target | Expected Improvement                    |
|-------------|---------------|-----------------------------------------|
| GSM8K       | ≥ 50%        | ≥ +5% improvement over baseline model  |
| MMLU        | ≥ 45%        | ≥ +5% improvement over baseline model  |
| StrategyQA  | ≥ 65%        | ≥ +5% improvement over baseline model  |

- Improvement should be demonstrated on **at least two** of the above three benchmarks.
- The solution should also be efficient and effective, ensuring **minimal latency overhead**.

---

## Suggested Open Models & Datasets

### Models

| Model           | Parameters |
|-----------------|-----------|
| Qwen 2.5 7B    | 7B        |
| Gemma 4 E4B    | ~4B       |
| Phi-3-Mini      | 3.8B      |

### Datasets

- **GSM8K** — Grade school math word problems
- **AQuA-RAT** — Algebraic word problems with rationales
- **MMLU** — Massive Multitask Language Understanding

---

## Recommended Infrastructure

- 2× NVIDIA GPU with ≥24 GB VRAM
- 12+ CPU cores
- 64 GB system RAM

---

## Key Dates

| Milestone | Date |
|-----------|------|
| Hackathon Begins | Apr 15, 2026 |
| Team Registration & Solution Blueprint Deadline | ~May 13, 2026 |
| Qualified Teams for Phase 2 | May 26, 2026 |
| Full Solution Submission | Jun 22, 2026 |
| Top Teams for Phase 3 | Jun 29, 2026 |
| Presentation by Shortlisted Teams | Jul 3, 2026 |
