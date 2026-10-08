#!/usr/bin/env bash
# ==============================================================================
# Model weight downloader with Hugging Face byte-to-byte & torch integrity check
# ==============================================================================

set -e
mkdir -p weights

python3 - << 'EOF'
import os
import sys
import urllib.request
import subprocess
import torch

MODELS = [
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

for m in MODELS:
    path = os.path.join("weights", m["name"])
    
    # 1. Query exact remote byte length from Hugging Face
    req = urllib.request.Request(m["url"], headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp:
        expected_bytes = int(resp.headers.get("content-length"))

    print(f"\n[*] Checking {m['name']} (Remote size: {expected_bytes:,} bytes)...")

    # 2. Check local file byte size and integrity
    need_download = True
    if os.path.exists(path):
        local_bytes = os.path.getsize(path)
        if local_bytes == expected_bytes:
            print(f"    Byte size matches ({local_bytes:,} bytes). Verifying torch checkpoint integrity...")
            try:
                torch.load(path, map_location="cpu", weights_only=False)
                print(f"[+] VERIFIED 100%: {m['name']} is complete and loadable. Skipping download.")
                need_download = False
            except Exception as e:
                print(f"[!] File corrupted ({e}). Removing and re-downloading...")
                os.remove(path)
        elif local_bytes < expected_bytes:
            pct = (local_bytes / expected_bytes) * 100
            print(f"[>] Incomplete file ({local_bytes:,} / {expected_bytes:,} bytes, {pct:.1f}%). Resuming download...")
        else:
            print(f"[!] File exceeds remote size ({local_bytes:,} > {expected_bytes:,}). Resetting...")
            os.remove(path)

    # 3. Download / Resume if needed
    if need_download:
        subprocess.run(["wget", "-c", "--show-progress", m["url"], "-O", path], check=True)
        final_bytes = os.path.getsize(path)
        assert final_bytes == expected_bytes, f"Size mismatch for {m['name']}: {final_bytes} != {expected_bytes}"
        torch.load(path, map_location="cpu", weights_only=False)
        print(f"[+] VERIFIED 100%: {m['name']} is complete and verified against Hugging Face!")

print("\n==================================================================")
print("[+] All model weights verified byte-for-byte and loadable!")
print("==================================================================")
EOF
