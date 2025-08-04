# syntax=docker/dockerfile:1

# ---------- base ----------
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# ---------- system ----------
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev python3-setuptools python3-wheel \
    python3-distutils python3-venv \
    git curl ca-certificates \
    build-essential gcc g++ make cmake pkg-config \
    libjpeg-dev zlib1g-dev libffi-dev libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic links to ensure headers are found
RUN ln -sf /usr/bin/python3 /usr/bin/python && \
    pip3 install --upgrade pip && \
    # Create symlinks for Python headers if they're not in the standard location
    ln -sf $(python3 -c "import sysconfig; print(sysconfig.get_path('include'))") /usr/local/include/python3.10

# Add a workaround for Triton compiler
RUN mkdir -p /usr/local/lib/python3.10/dist-packages/triton/backends/nvidia/include

# ---------- python ----------
WORKDIR /app
COPY requirements.txt .

# Install PyTorch and dependencies
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 && \
    pip install --no-cache-dir vllm==0.8.4 && \
    pip install --no-cache-dir -r requirements.txt

# Add environment variable to suppress PyTorch compiler errors
ENV TORCH_INDUCTOR_COMPILE_TO_EAGER=1

# ---------- runtime ----------
ENV NVIDIA_VISIBLE_DEVICES=0,1,2,3 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    VLLM_USE_CUDA_GRAPH=1 \
    VLLM_GPU_MEMORY_UTILIZATION=0.5 \
    VLLM_LOGGING_LEVEL=INFO \
    TORCH_COMPILE_MODE=reduce-overhead

EXPOSE 5000
CMD ["python","-m","vllm.entrypoints.openai.api_server", \
    "--host","0.0.0.0","--port","5000", \
    "--model","google/gemma-3-27b-it", \
    "--tensor-parallel-size","4", \
    "--max-model-len","4096", \
    "--trust-remote-code"]
