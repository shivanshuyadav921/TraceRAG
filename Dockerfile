# ==============================================================================
# Dockerfile: SLM Reasoning Enhancement via RL
# Ensures 100% reproducibility of training and evaluation
# ==============================================================================

FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# System dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-venv \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set Python
RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# Working directory
WORKDIR /app

# Install dependencies (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Set reproducibility seeds
ENV PYTHONHASHSEED=42
ENV CUBLAS_WORKSPACE_CONFIG=:4096:8
ENV TOKENIZERS_PARALLELISM=false

# Default command
CMD ["python", "scripts/run_full_pipeline.py", "--config-dir", "configs/"]

# ==============================================================================
# Usage:
#   docker build -t slm-reasoning .
#   docker run --gpus all -v $(pwd)/outputs:/app/outputs slm-reasoning
#
# Individual phases:
#   docker run --gpus all slm-reasoning python scripts/run_full_pipeline.py --phase sft
#   docker run --gpus all slm-reasoning python scripts/run_full_pipeline.py --phase grpo
#   docker run --gpus all slm-reasoning python scripts/run_full_pipeline.py --phase eval
#
# Interactive demo:
#   docker run --gpus all -it slm-reasoning python scripts/demo_inference.py --model ./outputs/grpo/final
# ==============================================================================
