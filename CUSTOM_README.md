# Quickstart Runbook for TAIR (Text-Aware Image Restoration)
> Automated NVIDIA Blackwell GPU Docker Container & 2-Tier Storage Remote Bridge Guide

This repository is configured to run inside a high-performance **NVIDIA Blackwell GPU Docker container** (RTX PRO 4500, B100, B200 with compute capability `sm_100` / `sm_120`), while enforcing a **strict 2-tier storage architecture**:
1. **Persistent Storage (72 GB)** at `/home/ubuntu/working/TAIR`: Source code, git history, configurations, and scripts.
2. **Ephemeral NVMe Workspace (410 GB)** at `/opt/dlami/nvme/workspace`: Model checkpoints (8+ GB), Hugging Face cache, user uploads, and batch restoration results.

The **Cloudflare Remote Bridge runs strictly on the host system outside Docker** so web links and file upload sessions never disconnect when containers stop or rebuild.

---

## Architecture Overview (2-Tier Storage & 3-Link Proxy)

```
Persistent Storage (/home/ubuntu/working/ - 72 GB Root)
 └── TAIR/ (Code, Git repository, .env, configs)
      ├── ./uploaded_stuff ➔ Symlinked to NVMe workspace/uploaded_stuff
      ├── ./results        ➔ Symlinked to NVMe workspace/results
      ├── ./models         ➔ Symlinked to NVMe workspace/models
      ├── ./weights        ➔ Symlinked to NVMe workspace/weights
      └── ./nvme_workspace ➔ Symlinked to NVMe /opt/dlami/nvme/workspace

NVMe Ephemeral Storage (/opt/dlami/nvme/workspace/ - 410 GB High-Speed NVMe)
 ├── cache/huggingface/ (Base foundation models & tokenizer cache)
 ├── uploaded_stuff/    (User uploaded degraded images & folders via Link 1)
 ├── results/           (Restored images & spotted text prediction maps via Link 2)
 ├── models/            (Custom weights & fine-tuning checkpoints)
 └── weights/           (TeReDiff Stage 3, DiffBIR v2.1, SwinIR, and SD2.1 weights)

Host System Remote Bridge Daemon (Cloudflare Quick Tunnels)
 ├── [CONFIRMED LINK 1] Upload Portal UI  --> Saves directly to NVMe uploaded_stuff
 ├── [CONFIRMED LINK 2] Repo HTTP Viewer  --> Browses code, results, nvme_workspace live
 └── [CONFIRMED LINK 3] Inbuilt Frontend  --> Proxies container Gradio UI on port 7860
```

---

## Step 1: Start the Remote Bridge on the Host (Background Daemon)

Run the Remote Bridge in the background on your host machine to provision NVMe storage and acquire public Cloudflare URLs:

### Linux / Remote SSH Host:
```bash
mkdir -p .remote_bridge && nohup python3 scripts/remote_bridge_launcher.py --repo . > .remote_bridge/bridge.log 2>&1 &
sleep 4
cat .remote_bridge/bridge_links.md
```

### Windows (PowerShell):
```powershell
Start-Process python -ArgumentList "scripts/remote_bridge_launcher.py --repo ." -WindowStyle Hidden
Start-Sleep -Seconds 4
Get-Content .remote_bridge/bridge_links.md
```
You will get 3 confirmed active access links:
- **Confirmed Link 1 (Upload Portal)**: Drag-and-drop degraded test images or entire folders from your browser directly into `./uploaded_stuff/`.
- **Confirmed Link 2 (Repo HTTP Viewer)**: Browse source code, inspect restoration logs, and download outputs from `./results/` live in your browser.
- **Confirmed Link 3 (Inbuilt Frontend)**: Live interactive **Gradio Web UI** proxying container port `7860`.

---

## Step 2: Download Model Weights to NVMe Storage

Before running restoration for the first time, download the foundation models and the official **TeReDiff Stage 3** checkpoint:

```bash
bash download_weights.sh
```
*Note: This script automatically detects `/opt/dlami/nvme/workspace` and downloads all weights directly into NVMe ephemeral storage, keeping the 72 GB persistent root disk free.*

---

## Step 3: Build & Start the GPU Docker Container

On the host machine, build the Blackwell-optimized Docker image and launch the container:

```bash
docker compose up --build -d
```

Attach an interactive shell into the container:
```bash
docker compose exec tair bash
```
> **Note**: Entering the container automatically places you inside `/workspace/TAIR`. All relative paths (`./`) refer directly to this repository!

---

## Step 4: Run Inference Inside the Container

Once inside the container (`/workspace/TAIR`), choose your preferred workflow:

### Option A: Universal Batch Restoration on Uploaded Images
Super-resolve and restore any images uploaded via **Link 1**:
```bash
python scripts/batch_inference_runner.py -i ./uploaded_stuff -o ./results
```
Restored images (`restored_<id>.png`) and detected text visualization maps (`pred_texts_<id>.png`) are written directly to `./results/` and can be inspected live in your browser on **Confirmed Link 2 (Repo HTTP Viewer)**!

### Option B: Launch the Interactive Gradio Web UI
Launch the interactive web interface:
```bash
python app.py
```
The app binds to port `7860` inside the container and is immediately accessible to anyone over the internet via **Confirmed Link 3**!

### Option C: Official Benchmark Evaluation on Demo Dataset
Run the evaluation script comparing against ground-truth images and computing PSNR, SSIM, and LPIPS metrics:
```bash
bash run_script/val_script/run_val_terediff.sh
```

---

## Step 5: Clean Teardown

When finished:

1. **Stop the GPU Docker Container (on Host):**
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
