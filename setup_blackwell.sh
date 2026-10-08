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

# Check CUDA Toolkit / NVCC
if [ -z "$CUDA_HOME" ]; then
    if [ -d "/usr/local/cuda" ]; then
        export CUDA_HOME="/usr/local/cuda"
    elif command -v nvcc &> /dev/null; then
        export CUDA_HOME="$(dirname $(dirname $(which nvcc)))"
    else
        echo "WARNING: CUDA_HOME is not set and nvcc not found."
        echo "Please install CUDA Toolkit 12.8+ to compile detectron2 and testr C++/CUDA extensions."
    fi
fi

if [ -n "$CUDA_HOME" ]; then
    export PATH="$CUDA_HOME/bin:$PATH"
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
    echo "CUDA_HOME set to: $CUDA_HOME"
fi

echo "=== [2/6] Installing PyTorch with CUDA 12.8 (Blackwell Support) ==="
# Blackwell (RTX PRO 4500 Blackwell / RTX 50-series) requires CUDA 12.8+ wheels for sm_120/sm_100 kernels.
pip install --upgrade pip setuptools wheel ninja

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

echo "=== [3/6] Installing TAIR Python Dependencies ==="
pip install -r requirements.txt

echo "=== [4/6] Compiling and Installing Detectron2 for Blackwell ==="
export FORCE_CUDA=1
# Target Blackwell sm_120 (workstation/client) and sm_100 (datacenter)
export TORCH_CUDA_ARCH_LIST="12.0;10.0"

cd detectron2
rm -rf build/ **/*.so
pip install --no-build-isolation -e .
cd ..

echo "=== [5/6] Compiling and Installing TESTR (AdelaiDet) for Blackwell ==="
cd testr
rm -rf build/ **/*.so
pip install --no-build-isolation -e .
cd ..

echo "=== [6/6] Verifying Setup and Hardware Compatibility ==="
python check_environment.py

echo "=================================================================="
echo " Setup complete! Your environment is ready for training & demo."
echo "=================================================================="
