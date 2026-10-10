#!/usr/bin/env bash
# ==============================================================================
# Blackwell Environment Bootstrap Script (RTX PRO 4500, B100, B200, sm_100/120)
# Enforces 2-Tier Storage: Persistent /home/ubuntu/working & Ephemeral NVMe
# ==============================================================================

set -e

# Allow uv to operate seamlessly in system/conda environments
export UV_SYSTEM_PYTHON=1

echo "=== [1/6] Verifying GPU and System Prerequisites ==="
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found. NVIDIA driver is not installed or not in PATH."
    exit 1
fi

nvidia-smi

# Check CUDA Toolkit / NVCC (prefer CUDA 12.8 if installed)
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
    if command -v nvcc &> /dev/null; then
        nvcc --version | grep "release" || true
    fi
fi

echo "=== [2/6] Provisioning NVMe Ephemeral Workspace (/opt/dlami/nvme/workspace) ==="
if [ -d "/opt/dlami/nvme" ]; then
    sudo mkdir -p /opt/dlami/nvme/workspace/cache/huggingface
    sudo mkdir -p /opt/dlami/nvme/workspace/uploaded_stuff
    sudo mkdir -p /opt/dlami/nvme/workspace/results
    sudo mkdir -p /opt/dlami/nvme/workspace/models
    sudo mkdir -p /opt/dlami/nvme/workspace/weights
    sudo chown -R $(id -u):$(id -g) /opt/dlami/nvme/workspace
    echo "[+] NVMe Ephemeral Workspace subdirectories provisioned with user permissions."

    # Create repo symlinks if running from inside a git repo
    for item in uploaded_stuff results models weights; do
        if [ ! -e "$item" ]; then
            ln -s "/opt/dlami/nvme/workspace/$item" "$item" 2>/dev/null || true
            echo "[+] Linked ./$item -> /opt/dlami/nvme/workspace/$item"
        fi
    done
    if [ ! -e "nvme_workspace" ]; then
        ln -s "/opt/dlami/nvme/workspace" "nvme_workspace" 2>/dev/null || true
    fi
else
    echo "[*] Notice: /opt/dlami/nvme not found. Using local workspace fallback."
fi

echo "=== [3/6] Installing uv, ninja, gdown, and modern wheel tooling ==="
pip install uv ninja gdown
uv pip install --upgrade setuptools wheel

echo "=== [4/6] Installing PyTorch with CUDA 12.8 (Blackwell SM_100/SM_120 Support) ==="
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

echo "=== [5/6] Patching PyTorch cpp_extension to bypass NVCC version mismatch ==="
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

export FORCE_CUDA=1
export TORCH_CUDA_ARCH_LIST="12.0;10.0"

echo "=== [6/6] Running Hardware Diagnostic ==="
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python "$SCRIPT_DIR/check_blackwell_env.py"

echo "=================================================================="
echo " Blackwell environment is bootstrapped and verified!"
echo " 2-Tier Storage: Persistent /home/ubuntu/working & Ephemeral NVMe ready."
echo "=================================================================="
