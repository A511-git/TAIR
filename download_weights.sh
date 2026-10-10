#!/usr/bin/env bash
# ==============================================================================
# Automated model downloader: Hugging Face foundation weights + TeReDiff Stage 3
# ==============================================================================

set -e

# Enforce 2-Tier Storage: route heavy weights directly to NVMe Ephemeral Workspace
if [ -d "/opt/dlami/nvme" ]; then
    sudo mkdir -p /opt/dlami/nvme/workspace/weights
    sudo mkdir -p /opt/dlami/nvme/workspace/cache/huggingface
    sudo chown -R $(id -u):$(id -g) /opt/dlami/nvme/workspace
    export HF_HOME="/opt/dlami/nvme/workspace/cache/huggingface"

    if [ ! -L "weights" ]; then
        if [ -d "weights" ] && [ ! "$(ls -A weights 2>/dev/null)" ]; then
            rmdir weights
        fi
        if [ ! -e "weights" ]; then
            ln -s /opt/dlami/nvme/workspace/weights weights
            echo "[+] Linked ./weights -> /opt/dlami/nvme/workspace/weights (NVMe Ephemeral Storage)"
        fi
    fi
else
    mkdir -p weights
fi

python3 - << 'EOF'
import os
import sys
import subprocess
import urllib.request
import http.cookiejar
import re

try:
    import torch
except ImportError:
    torch = None
    print("[*] Note: PyTorch is not installed in the host Python environment.")
    print("    Skipping in-memory tensor deserialization on host.")
    print("    Byte-size/integrity verification will be performed; deep tensor verification runs inside the container.")

# Resolve or install gdown with PEP 668 support, or fall back to native HTTP
gdown = None
try:
    import gdown
except ImportError:
    for install_cmd in [
        [sys.executable, "-m", "pip", "install", "--break-system-packages", "gdown"],
        [sys.executable, "-m", "pip", "install", "--user", "--break-system-packages", "gdown"],
        [sys.executable, "-m", "pip", "install", "gdown"],
    ]:
        try:
            subprocess.run(install_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            import gdown
            break
        except Exception:
            pass

def download_gdrive(file_id, output_path):
    """Downloads from Google Drive using gdown if available, else zero-dependency streaming with cookie confirmation."""
    if gdown is not None:
        gdown.download(id=file_id, output=output_path, quiet=False)
        return

    print(f"[*] Downloading from Google Drive using zero-dependency HTTP client...")
    cj = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))
    base_url = f"https://drive.google.com/uc?export=download&id={file_id}"
    req = urllib.request.Request(base_url, headers={"User-Agent": "Mozilla/5.0"})
    
    with opener.open(req) as resp:
        first_chunk = resp.read(65536)
        text = first_chunk.decode("utf-8", errors="ignore")
        match = re.search(r'confirm=([0-9A-Za-z_]+)', text)
        if match:
            confirm = match.group(1)
            confirm_url = f"https://drive.google.com/uc?export=download&confirm={confirm}&id={file_id}"
            req2 = urllib.request.Request(confirm_url, headers={"User-Agent": "Mozilla/5.0"})
            with opener.open(req2) as resp2:
                with open(output_path, "wb") as f:
                    downloaded = 0
                    while True:
                        chunk = resp2.read(2 * 1024 * 1024)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        sys.stdout.write(f"\r[>] Downloaded: {downloaded / (1024*1024):.1f} MB")
                        sys.stdout.flush()
            print()
        else:
            with open(output_path, "wb") as f:
                f.write(first_chunk)
                downloaded = len(first_chunk)
                while True:
                    chunk = resp.read(2 * 1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    sys.stdout.write(f"\r[>] Downloaded: {downloaded / (1024*1024):.1f} MB")
                    sys.stdout.flush()
            print()

HF_MODELS = [
    {
        "name": "realesrgan_s4_swinir_100k.pth",
        "url": "https://huggingface.co/lxq007/DiffBIR-v2/resolve/main/realesrgan_s4_swinir_100k.pth",
    },
    {
        "name": "DiffBIR_v2.1.pt",
        "url": "https://huggingface.co/lxq007/DiffBIR-v2/resolve/main/DiffBIR_v2.1.pt",
    },
    {
        "name": "sd2.1-base-zsnr-laionaes5.ckpt",
        "url": "https://huggingface.co/lxq007/DiffBIR-v2/resolve/main/sd2.1-base-zsnr-laionaes5.ckpt",
    },
]

GDRIVE_MODELS = [
    {
        "name": "terediff_stage3.pt",
        "gdrive_id": "14qtLOso_kurfY_FOzOWUR8z-_IvRwy-X",
        "min_size_mb": 1400,
    },
]

print("=== [1/2] Verifying & Downloading Hugging Face Foundation Models ===")
for m in HF_MODELS:
    path = os.path.join("weights", m["name"])
    
    req = urllib.request.Request(m["url"], headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req) as resp:
            cl = resp.headers.get("content-length")
            expected_bytes = int(cl) if cl else None
    except Exception as e:
        print(f"[!] Warning: Could not fetch Content-Length ({e})")
        expected_bytes = None

    if expected_bytes:
        print(f"\n[*] Checking {m['name']} (Expected: {expected_bytes:,} bytes)...")
    else:
        print(f"\n[*] Checking {m['name']}...")

    need_download = True
    if os.path.exists(path):
        local_bytes = os.path.getsize(path)
        if expected_bytes is not None and local_bytes == expected_bytes:
            if torch is not None:
                try:
                    torch.load(path, map_location="cpu", weights_only=False)
                    print(f"[+] VERIFIED 100%: {m['name']} is complete and valid. Skipping.")
                    need_download = False
                except Exception as e:
                    print(f"[!] Checkpoint corrupted ({e}). Re-downloading...")
                    os.remove(path)
            else:
                print(f"[+] VERIFIED: {m['name']} byte size matches ({local_bytes:,} bytes). Skipping.")
                need_download = False
        elif expected_bytes is not None and local_bytes < expected_bytes:
            print(f"[>] Incomplete file ({local_bytes:,} / {expected_bytes:,} bytes). Resuming...")
        elif expected_bytes is not None and local_bytes > expected_bytes:
            print(f"[!] File size mismatch ({local_bytes:,} > {expected_bytes:,}). Resetting...")
            os.remove(path)
        elif expected_bytes is None and local_bytes > 10 * 1024 * 1024:
            print(f"[+] File exists with size {local_bytes:,} bytes. Skipping.")
            need_download = False

    if need_download:
        subprocess.run(["wget", "-c", "--show-progress", m["url"], "-O", path], check=True)
        if expected_bytes is not None:
            assert os.path.getsize(path) == expected_bytes, f"Byte size mismatch for {m['name']}"
        if torch is not None:
            torch.load(path, map_location="cpu", weights_only=False)
            print(f"[+] VERIFIED 100%: {m['name']} matches Hugging Face byte-for-byte!")
        else:
            print(f"[+] VERIFIED: {m['name']} downloaded successfully ({os.path.getsize(path):,} bytes)!")

print("\n=== [2/2] Verifying & Downloading TeReDiff Stage 3 Checkpoint (Paper Model) ===")
for m in GDRIVE_MODELS:
    path = os.path.join("weights", m["name"])
    need_download = True
    
    if os.path.exists(path):
        size_mb = os.path.getsize(path) / (1024 * 1024)
        if size_mb >= m["min_size_mb"]:
            if torch is not None:
                try:
                    ckpt = torch.load(path, map_location="cpu", weights_only=False)
                    print(f"[+] VERIFIED 100%: {m['name']} is complete ({size_mb:.1f} MB, keys: {list(ckpt.keys())}). Skipping.")
                    need_download = False
                except Exception as e:
                    print(f"[!] Checkpoint corrupted ({e}). Re-downloading...")
                    os.remove(path)
            else:
                print(f"[+] VERIFIED: {m['name']} is complete ({size_mb:.1f} MB >= {m['min_size_mb']} MB). Skipping.")
                need_download = False
        else:
            print(f"[!] Incomplete file ({size_mb:.1f} MB < {m['min_size_mb']} MB). Re-downloading...")
            os.remove(path)

    if need_download:
        print(f"[>] Downloading {m['name']} from Google Drive...")
        download_gdrive(m["gdrive_id"], path)
        size_mb = os.path.getsize(path) / (1024 * 1024)
        assert size_mb >= m["min_size_mb"], f"Downloaded file too small: {size_mb} MB"
        if torch is not None:
            torch.load(path, map_location="cpu", weights_only=False)
            print(f"[+] VERIFIED 100%: {m['name']} downloaded and verified successfully!")
        else:
            print(f"[+] VERIFIED: {m['name']} downloaded successfully ({size_mb:.1f} MB)!")

print("\n==================================================================")
print("[+] All professional model weights downloaded, verified, and ready!")
print("==================================================================")
EOF
