#!/usr/bin/env python3
"""
Automated Docker & Docker Compose Generator for Blackwell AI Model Runner
Inspects repository architecture and dependencies, then generates:
1. Dockerfile configured for NVIDIA Blackwell GPUs (CUDA 12.8) with WORKDIR /workspace/[repo_name]
2. docker-compose.yml mapping ./:/workspace/[repo_name] with NVMe 2-Tier storage and GPU pass-through
3. CUSTOM_README.md in repo root documenting background Remote Bridge, Docker commands, and CLI inference
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Import repo analyzer from the same scripts directory
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from repo_analyzer import RepoAnalyzer


def generate_dockerfile_content(repo_name: str) -> str:
    return f"""# ==============================================================================
# Blackwell GPU AI Model Runner Dockerfile
# Optimized for NVIDIA Blackwell (RTX PRO 4500, B100, B200 - sm_100/sm_120)
# ==============================================================================

FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV UV_SYSTEM_PYTHON=1
ENV FORCE_CUDA=1
ENV TORCH_CUDA_ARCH_LIST="12.0;10.0"

# Install essential system dependencies and modern Python
RUN apt-get update && apt-get install -y --no-install-recommends \\
    python3.10 \\
    python3.10-dev \\
    python3-pip \\
    git \\
    wget \\
    curl \\
    ffmpeg \\
    libsm6 \\
    libxext6 \\
    libgl1 \\
    libglib2.0-0 \\
    build-essential \\
    ninja-build \\
    ca-certificates \\
    && rm -rf /var/lib/apt/lists/*

# Set python3 as default
RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \\
    ln -sf /usr/bin/python3.10 /usr/bin/python3

# Install uv for fast, deterministic dependency resolution
RUN pip install --no-cache-dir uv ninja gdown wheel setuptools

# Install PyTorch with CUDA 12.8 (Native Blackwell sm_100/sm_120 Support)
RUN uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Bypass NVCC version mismatch check in PyTorch cpp_extension
RUN python -c 'import torch.utils.cpp_extension as ce; p = ce.__file__; s = open(p).read(); t = "raise RuntimeError(CUDA_MISMATCH_MESSAGE, cuda_str_version, torch.version.cuda)"; open(p, "w").write(s.replace(t, "pass")) if t in s else None'

# Set working directory inside container
WORKDIR /workspace/{repo_name}

# Install repo dependencies if requirements.txt exists
COPY requirements.txt* /workspace/{repo_name}/
RUN if [ -f /workspace/{repo_name}/requirements.txt ]; then \\
        sed -i '/xformers/d' /workspace/{repo_name}/requirements.txt && \\
        sed -i '/^torch==/d' /workspace/{repo_name}/requirements.txt && \\
        sed -i '/^torchvision==/d' /workspace/{repo_name}/requirements.txt && \\
        sed -i '/^torchaudio==/d' /workspace/{repo_name}/requirements.txt && \\
        sed -i '/^triton==/d' /workspace/{repo_name}/requirements.txt && \\
        uv pip install --extra-index-url https://download.pytorch.org/whl/cu128 -r /workspace/{repo_name}/requirements.txt || true ; \\
    fi

# Build and install submodules if present (detectron2, testr) into system site-packages
COPY detectron2* /workspace/{repo_name}/detectron2/
COPY testr* /workspace/{repo_name}/testr/
RUN if [ -d /workspace/{repo_name}/detectron2 ]; then \\
        cd /workspace/{repo_name}/detectron2 && (uv pip install --no-build-isolation . || NO_EXT=1 uv pip install --no-build-isolation .) ; \\
    fi && \\
    if [ -d /workspace/{repo_name}/testr ]; then \\
        cd /workspace/{repo_name}/testr && (uv pip install --no-build-isolation . || NO_EXT=1 uv pip install --no-build-isolation .) ; \\
    fi

# Ensure modern transformers & accelerate using cu128 index
RUN uv pip install --extra-index-url https://download.pytorch.org/whl/cu128 \\
    "transformers>=4.37.0" "accelerate>=0.28.0" diffusers peft \\
    opencv-python-headless gradio matplotlib pyyaml einops ftfy sentencepiece lightning prodigyopt \\
    ujson easydict scikit-image Levenshtein pandas pandarallel webcolors av lpips \\
    mmengine modelscope safetensors datasets "numpy<2" tqdm requests "openai-clip>=1.0.1" pyiqa gdown

# Configure PYTHONPATH
ENV PYTHONPATH=/workspace/{repo_name}:/workspace/{repo_name}/testr:$PYTHONPATH

# Default entry command
CMD ["/bin/bash"]
"""


def generate_docker_compose_yaml(service_name: str, repo_name: str, frontend_port: int) -> str:
    ports_section = ""
    if frontend_port:
        ports_section = f"""    ports:
      - "{frontend_port}:{frontend_port}"
"""

    compose_yaml = f"""services:
  {service_name}:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: {service_name}_blackwell_container
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - NVIDIA_DRIVER_CAPABILITIES=compute,utility
      - FORCE_CUDA=1
      - TORCH_CUDA_ARCH_LIST=12.0;10.0
      - UV_SYSTEM_PYTHON=1
      - HF_HOME=/root/.cache/huggingface
      - HF_TOKEN=${{HF_TOKEN:-}}
      - HUGGING_FACE_HUB_TOKEN=${{HF_TOKEN:-}}
{ports_section}    volumes:
      # Mount repository code into /workspace/{repo_name} (from persistent 72 GB storage)
      - ./:/workspace/{repo_name}
      # NVMe Ephemeral Workspace Storage (410 GB) - all heavy/temp data
      - /opt/dlami/nvme/workspace/cache/huggingface:/root/.cache/huggingface
      - /opt/dlami/nvme/workspace/uploaded_stuff:/workspace/{repo_name}/uploaded_stuff
      - /opt/dlami/nvme/workspace/results:/workspace/{repo_name}/results
      - /opt/dlami/nvme/workspace/models:/workspace/{repo_name}/models
      - /opt/dlami/nvme/workspace/weights:/workspace/{repo_name}/weights
    ipc: host
    stdin_open: true
    tty: true
    working_dir: /workspace/{repo_name}
    command: /bin/bash
"""
    return compose_yaml


def generate_custom_readme(repo_name: str, service_name: str, has_frontend: bool, frontend_type: str, frontend_port: int, entrypoint: str) -> str:
    fe_section = ""
    if has_frontend:
        fe_section = f"""### Option B: Launch Interactive Web UI ({frontend_type.upper()})
```bash
python {entrypoint}
```
The web UI binds to port `{frontend_port}` inside the container, which is forwarded to the host and accessible over the public internet via **Confirmed Link 3**!
"""
    else:
        fe_section = """*Note: No inbuilt frontend detected. All execution is performed via CLI batch inference.*
"""

    return f"""# Quickstart Runbook for {repo_name}
> Automated Blackwell GPU Container & 2-Tier Storage Remote Bridge Guide

This repository is configured to run inside a high-performance **NVIDIA Blackwell GPU Docker container**, while enforcing a **strict 2-tier storage architecture**:
1. **Persistent Storage (72 GB)** at `/home/ubuntu/working` (Source code, git history, configs).
2. **Ephemeral NVMe Workspace (410 GB)** at `/opt/dlami/nvme/workspace` (Hugging Face cache, checkpoints, uploaded files, and inference results).

The **Cloudflare Remote Bridge runs strictly on the host system** so web links and file uploads never disconnect during container restarts.

---

## Architecture Overview (2-Tier Storage Layout)

```
Persistent Storage (/home/ubuntu/working/ - 72 GB)
 └── {repo_name}/ (Code, Git history, .env, configs)
      ├── ./uploaded_stuff ➔ Symlinked to NVMe workspace/uploaded_stuff
      ├── ./results        ➔ Symlinked to NVMe workspace/results
      ├── ./models         ➔ Symlinked to NVMe workspace/models
      ├── ./weights        ➔ Symlinked to NVMe workspace/weights
      └── ./nvme_workspace ➔ Symlinked to NVMe /opt/dlami/nvme/workspace

NVMe Ephemeral Storage (/opt/dlami/nvme/workspace/ - 410 GB)
 ├── cache/huggingface/ (Base models & checkpoints)
 ├── uploaded_stuff/    (User uploaded assets & folders via Link 1)
 ├── results/           (Inference outputs & restored images via Link 2)
 ├── models/            (Custom weights & fine-tunes)
 └── weights/           (Foundation weights & TeReDiff checkpoints)

Host System Remote Bridge Daemon (Cloudflare Tunnels)
 ├── [CONFIRMED LINK 1] Upload Portal UI  --> Saves directly to NVMe uploaded_stuff
 ├── [CONFIRMED LINK 2] Repo HTTP Viewer  --> Browses code, results, nvme_workspace
 └── [CONFIRMED LINK 3] Inbuilt Frontend  --> Proxies container port {frontend_port}
```

---

## Step 1: Start the Remote Bridge on the Host (Background Mode)

Run the Remote Bridge in the background on your host machine to get your public Cloudflare URLs:

### Linux / Remote SSH Host:
```bash
mkdir -p .remote_bridge && nohup python3 scripts/remote_bridge_launcher.py --repo . > .remote_bridge/bridge.log 2>&1 &
```

### Windows (PowerShell):
```powershell
Start-Process python -ArgumentList "scripts/remote_bridge_launcher.py --repo ." -WindowStyle Hidden
```

### View Your Active Access Links:
```bash
cat .remote_bridge/bridge_links.md
```
You will get 3 active access links:
- **Confirmed Link 1 (Upload Portal)**: Drag-and-drop test images or entire folders directly into `./uploaded_stuff/`.
- **Confirmed Link 2 (Repo HTTP Viewer)**: Browse code, inspect logs, and view generated outputs in `./results/` live.
- **Confirmed Link 3 (Inbuilt Frontend)**: Live interactive UI ({frontend_type or 'Gradio'}) proxying container port `{frontend_port}`.

---

## Step 2: Download Model Weights (Directly to NVMe Workspace)

Before running restoration for the first time, download the foundation and TeReDiff Stage 3 weights:
```bash
bash download_weights.sh
```
*(This automatically routes weights to `/opt/dlami/nvme/workspace/weights` so your 72 GB root disk remains 100% free.)*

---

## Step 3: Build & Launch the GPU Docker Container

On the host machine, build and start the GPU container:

```bash
docker compose up --build -d
```

Attach a bash shell into the container:
```bash
docker compose exec {service_name} bash
```
> **Note**: Entering the container automatically places you inside `/workspace/{repo_name}`. All relative paths (`./`) refer directly to this repository!

---

## Step 4: Run Inference Inside the Container

Once inside the container (`/workspace/{repo_name}`):

### Option A: Batch Inference on Uploaded Images
Run restoration on images uploaded via Link 1:
```bash
python scripts/batch_inference_runner.py -i ./uploaded_stuff -o ./results
```
Restoration outputs and spotted text visualizations are written to `./results/` and are immediately viewable in your browser on **Confirmed Link 2 (Repo HTTP Viewer)**!

### Option B: Benchmark Evaluation on Demo Dataset
```bash
bash run_script/val_script/run_val_terediff.sh
```

{fe_section}

---

## Step 5: Clean Teardown

When finished:

1. **Stop the Docker Container (on Host):**
   ```bash
   docker compose down
   ```

2. **Stop the Background Remote Bridge (on Host):**
   - **Linux / Remote SSH:**
     ```bash
     pkill -f remote_bridge_launcher.py
     ```
   - **Windows (PowerShell):**
     ```powershell
     Get-Process -Name python, cloudflared -ErrorAction SilentlyContinue | Stop-Process
     ```
"""


class DockerSetupGenerator:
    def __init__(self, repo_dir: str = "."):
        self.repo_dir = Path(repo_dir).resolve()
        self.analyzer = RepoAnalyzer(str(self.repo_dir))

    def generate(self, overwrite: bool = False):
        analysis = self.analyzer.analyze()
        repo_name = analysis.get("repo_name", "TAIR")
        service_name = repo_name.lower().replace(" ", "_").replace("-", "_")
        frontend_port = analysis.get("detected_port", 7860)
        entrypoint = analysis.get("frontend_entrypoint", "app.py")
        has_frontend = analysis.get("has_frontend", False)
        frontend_type = analysis.get("frontend_type", "gradio")

        print("=" * 70)
        print("          DOCKER CONTAINER & RUNBOOK GENERATOR")
        print("=" * 70)
        print(f"[*] Target Repository : {self.repo_dir}")
        print(f"[*] Repository Name   : {repo_name}")
        print(f"[*] Inbuilt Frontend  : {has_frontend} ({frontend_type})")
        print(f"[*] Container Workdir : /workspace/{repo_name}")
        print("=" * 70)

        # 1. Ensure 2-tier storage layout & NVMe workspace exist
        nvme_base = Path("/opt/dlami/nvme")
        if nvme_base.exists():
            workspace_dir = nvme_base / "workspace"
            try:
                workspace_dir.mkdir(parents=True, exist_ok=True)
            except PermissionError:
                import subprocess
                subprocess.run(["sudo", "mkdir", "-p", str(workspace_dir)], check=False)
                uid = os.getuid() if hasattr(os, "getuid") else 1000
                gid = os.getgid() if hasattr(os, "getgid") else 1000
                subprocess.run(["sudo", "chown", "-R", f"{uid}:{gid}", str(workspace_dir)], check=False)

            for sub in ["uploaded_stuff", "results", "models", "weights", "cache/huggingface"]:
                (workspace_dir / sub).mkdir(parents=True, exist_ok=True)

            # Symlink in repo
            for link_name, target in [("uploaded_stuff", workspace_dir / "uploaded_stuff"),
                                      ("results", workspace_dir / "results"),
                                      ("models", workspace_dir / "models"),
                                      ("weights", workspace_dir / "weights"),
                                      ("nvme_workspace", workspace_dir)]:
                lp = self.repo_dir / link_name
                try:
                    if lp.is_symlink():
                        if lp.resolve() == target.resolve():
                            continue
                        lp.unlink()
                    elif lp.is_dir() and not any(lp.iterdir()):
                        lp.rmdir()
                    if not lp.exists():
                        lp.symlink_to(target, target_is_directory=True)
                except Exception:
                    pass
        else:
            (self.repo_dir / "uploaded_stuff").mkdir(parents=True, exist_ok=True)
            (self.repo_dir / "results").mkdir(parents=True, exist_ok=True)
            (self.repo_dir / "models").mkdir(parents=True, exist_ok=True)

        # 2. Write Dockerfile
        dockerfile_path = self.repo_dir / "Dockerfile"
        if dockerfile_path.exists() and not overwrite:
            print(f"[*] Dockerfile already exists at {dockerfile_path}")
        else:
            dockerfile_path.write_text(generate_dockerfile_content(repo_name), encoding="utf-8")
            print(f"[+] Created Blackwell Dockerfile: {dockerfile_path}")

        # 3. Write docker-compose.yml
        compose_path = self.repo_dir / "docker-compose.yml"
        if compose_path.exists() and not overwrite:
            print(f"[*] docker-compose.yml already exists at {compose_path}")
        else:
            compose_content = generate_docker_compose_yaml(
                service_name=service_name,
                repo_name=repo_name,
                frontend_port=frontend_port
            )
            compose_path.write_text(compose_content, encoding="utf-8")
            print(f"[+] Created docker-compose.yml: {compose_path}")

        # 4. Write CUSTOM_README.md in root
        readme_path = self.repo_dir / "CUSTOM_README.md"
        if readme_path.exists() and not overwrite:
            print(f"[*] CUSTOM_README.md already exists at {readme_path}")
        else:
            readme_content = generate_custom_readme(
                repo_name=repo_name,
                service_name=service_name,
                has_frontend=has_frontend,
                frontend_type=frontend_type,
                frontend_port=frontend_port,
                entrypoint=entrypoint
            )
            readme_path.write_text(readme_content, encoding="utf-8")
            print(f"[+] Created root CUSTOM_README.md: {readme_path}")

        # 5. Write .env.example
        env_example_path = self.repo_dir / ".env.example"
        if not env_example_path.exists():
            env_example_content = "# Environment Configurations for TAIR & Remote Bridge\nHF_TOKEN=\nHUGGING_FACE_HUB_TOKEN=\n"
            env_example_path.write_text(env_example_content, encoding="utf-8")
            print(f"[+] Created .env.example: {env_example_path}")

        print("\n[+] All Docker & Runbook assets generated successfully!")


def main():
    parser = argparse.ArgumentParser(description="Docker & Docker Compose Generator for Blackwell AI Repos")
    parser.add_argument("--repo", "-r", type=str, default=".", help="Target repository directory path")
    parser.add_argument("--overwrite", "-w", action="store_true", help="Overwrite existing Docker files")
    args = parser.parse_args()

    generator = DockerSetupGenerator(args.repo)
    generator.generate(overwrite=args.overwrite)


if __name__ == "__main__":
    main()
