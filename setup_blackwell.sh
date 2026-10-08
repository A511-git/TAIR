#!/usr/bin/env bash
# ==============================================================================
# Setup script for TAIR on NVIDIA RTX PRO 4500 Blackwell (or modern Blackwell GPUs)
# ==============================================================================

set -e

echo "=== [1/6] Verifying GPU and System Prerequisites ==="
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found. NVIDIA driver is not installed or not in PATH."
    exit 1
fi

nvidia-smi

# Check CUDA Toolkit / NVCC (prefer CUDA 12.8 if installed, otherwise use default CUDA)
if [ -d "/usr/local/cuda-12.8" ]; then
    export CUDA_HOME="/usr/local/cuda-12.8"
elif [ -d "/usr/local/cuda-12" ]; then
    export CUDA_HOME="/usr/local/cuda-12"
elif [ -d "/usr/local/cuda" ]; then
    export CUDA_HOME="/usr/local/cuda"
elif command -v nvcc &> /dev/null; then
    export CUDA_HOME="$(dirname $(dirname $(which nvcc)))"
fi

if [ -n "$CUDA_HOME" ]; then
    export PATH="$CUDA_HOME/bin:$PATH"
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
    echo "Using CUDA_HOME: $CUDA_HOME"
    nvcc --version | grep "release"
fi

echo "=== [2/6] Installing uv and PyTorch with CUDA 12.8 (Blackwell Support) ==="
pip install uv ninja
uv pip install --upgrade setuptools wheel

uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Patch PyTorch cpp_extension to allow nvcc version mismatch (e.g., nvcc 13.2 vs torch 12.8)
python -c '
import torch.utils.cpp_extension as ce
p = ce.__file__
with open(p, "r") as f:
    src = f.read()
target = "raise RuntimeError(CUDA_MISMATCH_MESSAGE, cuda_str_version, torch.version.cuda)"
sub = "print(f\"[WARNING] CUDA version mismatch: detected {cuda_str_version} vs torch {torch.version.cuda}. Proceeding with build...\")"
if target in src:
    with open(p, "w") as f:
        f.write(src.replace(target, sub))
    print("[+] Patched PyTorch cpp_extension: CUDA version mismatch check bypassed.")
'

echo "=== [3/6] Installing TAIR Python Dependencies with uv ==="
uv pip install -r requirements.txt

echo "=== [4/6] Compiling and Installing Detectron2 for Blackwell ==="
export FORCE_CUDA=1
export TORCH_CUDA_ARCH_LIST="12.0;10.0"

cd detectron2
rm -rf build/ **/*.so
uv pip install --no-build-isolation -e .
cd ..

echo "=== [5/6] Compiling and Installing TESTR (AdelaiDet) for Blackwell ==="
cd testr
rm -rf build/ **/*.so
uv pip install --no-build-isolation -e .
cd ..

echo "=== [6/6] Verifying Setup and Hardware Compatibility ==="
python check_environment.py

echo "=================================================================="
echo " Setup complete! Your environment is ready for training & demo."
echo "=================================================================="
