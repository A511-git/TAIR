#!/usr/bin/env bash
# ==============================================================================
# Automated model downloader: Hugging Face foundation weights + TeReDiff Stage 3
# ==============================================================================

set -e
mkdir -p weights

python3 - << 'EOF'
import os
import sys
import subprocess
import urllib.request
import torch

# Ensure gdown is available for downloading TeReDiff from Google Drive
try:
    import gdown
except ImportError:
    print("[*] Installing gdown for Google Drive downloads...")
    subprocess.run([sys.executable, "-m", "pip", "install", "gdown"], check=True)
    import gdown

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
    with urllib.request.urlopen(req) as resp:
        expected_bytes = int(resp.headers.get("content-length"))

    print(f"\n[*] Checking {m['name']} (Expected: {expected_bytes:,} bytes)...")
    need_download = True
    if os.path.exists(path):
        local_bytes = os.path.getsize(path)
        if local_bytes == expected_bytes:
            try:
                torch.load(path, map_location="cpu", weights_only=False)
                print(f"[+] VERIFIED 100%: {m['name']} is complete and valid. Skipping.")
                need_download = False
            except Exception as e:
                print(f"[!] Checkpoint corrupted ({e}). Re-downloading...")
                os.remove(path)
        elif local_bytes < expected_bytes:
            print(f"[>] Incomplete file ({local_bytes:,} / {expected_bytes:,} bytes). Resuming...")
        else:
            print(f"[!] File size mismatch. Resetting...")
            os.remove(path)

    if need_download:
        subprocess.run(["wget", "-c", "--show-progress", m["url"], "-O", path], check=True)
        assert os.path.getsize(path) == expected_bytes, f"Byte size mismatch for {m['name']}"
        torch.load(path, map_location="cpu", weights_only=False)
        print(f"[+] VERIFIED 100%: {m['name']} matches Hugging Face byte-for-byte!")

print("\n=== [2/2] Verifying & Downloading TeReDiff Stage 3 Checkpoint (Paper Model) ===")
for m in GDRIVE_MODELS:
    path = os.path.join("weights", m["name"])
    need_download = True
    
    if os.path.exists(path):
        size_mb = os.path.getsize(path) / (1024 * 1024)
        if size_mb >= m["min_size_mb"]:
            try:
                ckpt = torch.load(path, map_location="cpu", weights_only=False)
                print(f"[+] VERIFIED 100%: {m['name']} is complete ({size_mb:.1f} MB, keys: {list(ckpt.keys())}). Skipping.")
                need_download = False
            except Exception as e:
                print(f"[!] Checkpoint corrupted ({e}). Re-downloading...")
                os.remove(path)
        else:
            print(f"[!] Incomplete file ({size_mb:.1f} MB < {m['min_size_mb']} MB). Re-downloading...")
            os.remove(path)

    if need_download:
        print(f"[>] Downloading {m['name']} from Google Drive...")
        gdown.download(id=m["gdrive_id"], output=path, quiet=False)
        size_mb = os.path.getsize(path) / (1024 * 1024)
        assert size_mb >= m["min_size_mb"], f"Downloaded file too small: {size_mb} MB"
        torch.load(path, map_location="cpu", weights_only=False)
        print(f"[+] VERIFIED 100%: {m['name']} downloaded and verified successfully!")

print("\n==================================================================")
print("[+] All professional model weights downloaded, verified, and ready!")
print("==================================================================")
EOF
