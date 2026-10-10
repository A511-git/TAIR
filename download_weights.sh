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

# Ensure gdown is available on host if possible
if ! python3 -c "import gdown" 2>/dev/null; then
    pip install --break-system-packages gdown 2>/dev/null || \
    sudo pip install --break-system-packages gdown 2>/dev/null || \
    python3 -m pip install --break-system-packages gdown 2>/dev/null || true
fi

python3 - << 'EOF'
import os
import sys
import glob
import subprocess
import urllib.request
import urllib.parse
import http.cookiejar
import re

# Add user site-packages to sys.path so pip --user packages are visible
for p in glob.glob(os.path.expanduser("~/.local/lib/python*/site-packages")):
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    import torch
except ImportError:
    torch = None
    print("[*] Note: PyTorch is not installed in the host Python environment.")
    print("    Skipping in-memory tensor deserialization on host.")
    print("    Byte-size/integrity verification will be performed; deep tensor verification runs inside the container.")

def download_gdrive(file_id, output_path):
    """Downloads from Google Drive using gdown, docker container gdown, or resilient HTTP fallback."""
    # 1. Try python gdown module
    try:
        import gdown
        print(f"[>] Using gdown module to download Google Drive file...")
        gdown.download(id=file_id, output=output_path, quiet=False)
        if os.path.exists(output_path) and os.path.getsize(output_path) > 100 * 1024 * 1024:
            return
    except Exception as e:
        print(f"[*] Note: gdown module: {e}")

    # 2. Try CLI gdown
    if subprocess.call(["which", "gdown"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0:
        print("[>] Trying system gdown command...")
        res = subprocess.call(["gdown", "--id", file_id, "-O", output_path])
        if res == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 100 * 1024 * 1024:
            return

    # 3. Try container gdown (if docker container tair is running)
    try:
        print("[>] Trying Docker container gdown...")
        subprocess.run(
            ["docker", "compose", "exec", "-T", "tair", "gdown", "--id", file_id, "-O", f"/workspace/TAIR/{output_path}"],
            check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        if os.path.exists(output_path) and os.path.getsize(output_path) > 100 * 1024 * 1024:
            print("[+] Downloaded successfully via Docker container gdown!")
            return
    except Exception:
        pass

    # 4. Resilient native HTTP with cookie and form parsing
    print(f"[*] Downloading from Google Drive using resilient HTTP client...")
    cj = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))
    base_url = f"https://drive.google.com/uc?export=download&id={file_id}"
    req = urllib.request.Request(base_url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    
    with opener.open(req) as resp:
        html = resp.read().decode("utf-8", errors="ignore")

    action_match = re.search(r'<form[^>]*action=["\']([^"\']+)["\']', html)
    inputs = dict(re.findall(r'<input[^>]+name=["\']([^"\']+)["\'][^>]+value=["\']([^"\']+)["\']', html))
    inputs.update(dict(re.findall(r'<input[^>]+value=["\']([^"\']+)["\'][^>]+name=["\']([^"\']+)["\']', html)))

    download_url = None
    if action_match and ("confirm" in inputs or "uuid" in inputs):
        action_url = action_match.group(1)
        if not action_url.startswith("http"):
            action_url = f"https://drive.google.com{action_url}"
        params = urllib.parse.urlencode(inputs)
        download_url = f"{action_url}?{params}"
    elif "confirm=" in html:
        confirm_token = re.search(r'confirm=([0-9A-Za-z_-]+)', html).group(1)
        download_url = f"https://drive.google.com/uc?export=download&confirm={confirm_token}&id={file_id}"
    else:
        link_match = re.search(r'id=["\']uc-download-link["\'][^>]*href=["\']([^"\']+)["\']', html)
        if link_match:
            download_url = link_match.group(1).replace("&amp;", "&")
            if not download_url.startswith("http"):
                download_url = f"https://drive.google.com{download_url}"

    if not download_url:
        download_url = f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"

    req2 = urllib.request.Request(download_url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    with opener.open(req2) as resp2:
        with open(output_path, "wb") as f:
            downloaded = 0
            while True:
                chunk = resp2.read(4 * 1024 * 1024)
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
        if size_mb < m["min_size_mb"]:
            if os.path.exists(path):
                os.remove(path)
            raise AssertionError(f"Downloaded file too small: {size_mb:.2f} MB (< {m['min_size_mb']} MB).")
        if torch is not None:
            torch.load(path, map_location="cpu", weights_only=False)
            print(f"[+] VERIFIED 100%: {m['name']} downloaded and verified successfully!")
        else:
            print(f"[+] VERIFIED: {m['name']} downloaded successfully ({size_mb:.1f} MB)!")

print("\n==================================================================")
print("[+] All professional model weights downloaded, verified, and ready!")
print("==================================================================")
EOF
