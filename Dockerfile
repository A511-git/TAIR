# ==============================================================================
# Blackwell GPU AI Model Runner Dockerfile for TAIR
# Optimized for NVIDIA Blackwell GPUs (RTX PRO 4500, B100, B200 - sm_100/sm_120)
# ==============================================================================

FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV UV_SYSTEM_PYTHON=1
ENV FORCE_CUDA=1
ENV TORCH_CUDA_ARCH_LIST="12.0;10.0"

# Install essential system dependencies and Python 3.10
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    wget \
    curl \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    libglib2.0-0 \
    build-essential \
    ninja-build \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Set python3.10 as default python
RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/python3.10 /usr/bin/python3

# Install uv for fast, deterministic dependency resolution
RUN pip install --no-cache-dir uv ninja gdown wheel setuptools

# Install PyTorch with CUDA 12.8 (Native Blackwell sm_100/sm_120 Support)
RUN uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Bypass NVCC version mismatch check in PyTorch cpp_extension
RUN python -c 'import torch.utils.cpp_extension as ce; p = ce.__file__; s = open(p).read(); t = "raise RuntimeError(CUDA_MISMATCH_MESSAGE, cuda_str_version, torch.version.cuda)"; open(p, "w").write(s.replace(t, "pass")) if t in s else None'

# Set working directory inside container
WORKDIR /workspace/TAIR

# Install repo dependencies from requirements.txt
COPY requirements.txt* /workspace/TAIR/
RUN if [ -f /workspace/TAIR/requirements.txt ]; then \
        sed -i '/xformers/d' /workspace/TAIR/requirements.txt && \
        sed -i '/^torch==/d' /workspace/TAIR/requirements.txt && \
        sed -i '/^torchvision==/d' /workspace/TAIR/requirements.txt && \
        sed -i '/^torchaudio==/d' /workspace/TAIR/requirements.txt && \
        sed -i '/^triton==/d' /workspace/TAIR/requirements.txt && \
        uv pip install --extra-index-url https://download.pytorch.org/whl/cu128 -r /workspace/TAIR/requirements.txt || true ; \
    fi

# Build and install submodules (detectron2, testr) into system site-packages
COPY detectron2 /workspace/TAIR/detectron2
COPY testr /workspace/TAIR/testr
RUN cd /workspace/TAIR/detectron2 && (uv pip install --no-build-isolation . || NO_EXT=1 uv pip install --no-build-isolation .) && \
    cd /workspace/TAIR/testr && (uv pip install --no-build-isolation . || NO_EXT=1 uv pip install --no-build-isolation .)

# Ensure required runtime libraries are installed
RUN uv pip install --extra-index-url https://download.pytorch.org/whl/cu128 \
    "transformers>=4.37.0" "accelerate>=0.28.0" diffusers peft \
    opencv-python-headless gradio matplotlib pyyaml einops ftfy sentencepiece lightning prodigyopt \
    ujson easydict scikit-image Levenshtein pandas pandarallel webcolors av lpips \
    mmengine modelscope safetensors datasets "numpy<2" tqdm requests "openai-clip>=1.0.1" pyiqa gdown

# Configure PYTHONPATH to recognize TAIR and TESTR modules
ENV PYTHONPATH=/workspace/TAIR:/workspace/TAIR/testr:$PYTHONPATH

# Default entry command
CMD ["/bin/bash"]
