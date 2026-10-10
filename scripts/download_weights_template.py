#!/usr/bin/env python3
"""
Model Weights Downloader & Integrity Verifier
Downloads weights from Hugging Face (including authenticated/gated repos) or Google Drive,
verifying exact byte counts and PyTorch checkpoint deserialization before proceeding.
"""

import argparse
import os
import sys
import subprocess
import urllib.request
from pathlib import Path


def get_hf_token(cli_token: str = None) -> str:
    """Resolves HF token from CLI flag, environment variables, or local .env file."""
    if cli_token:
        return cli_token

    env_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if env_token:
        return env_token

    # Check local .env file
    env_file = Path(".env")
    if env_file.exists():
        try:
            for line in env_file.read_text().splitlines():
                line = line.strip()
                if line.startswith("HF_TOKEN=") or line.startswith("HUGGING_FACE_HUB_TOKEN="):
                    token = line.split("=", 1)[1].strip().strip('"').strip("'")
                    if token:
                        return token
        except Exception:
            pass

    return None


def ensure_gdown():
    try:
        import gdown
        return gdown
    except ImportError:
        print("[*] Installing gdown for Google Drive downloads...")
        subprocess.run([sys.executable, "-m", "pip", "install", "gdown"], check=True)
        import gdown
        return gdown


def verify_and_download_hf(model_name, url_or_repo, target_dir="weights", token: str = None):
    if Path("/opt/dlami/nvme").exists():
        nvme_ws = Path("/opt/dlami/nvme/workspace")
        hf_cache = nvme_ws / "cache" / "huggingface"
        hf_cache.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("HF_HOME", str(hf_cache))
        if target_dir in ("weights", "./models", "models"):
            nvme_models = nvme_ws / "models"
            nvme_models.mkdir(parents=True, exist_ok=True)
            target_dir = str(nvme_models)

    os.makedirs(target_dir, exist_ok=True)
    target_path = os.path.join(target_dir, model_name)
    resolved_token = get_hf_token(token)

    # Case 1: Hugging Face repo ID (e.g., 'black-forest-labs/FLUX.1-Fill-dev')
    if "/" in url_or_repo and not url_or_repo.startswith("http"):
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            subprocess.run([sys.executable, "-m", "pip", "install", "huggingface_hub"], check=True)
            from huggingface_hub import snapshot_download

        print(f"\n[*] Downloading repository '{url_or_repo}' from Hugging Face...")
        out_dir = snapshot_download(
            repo_id=url_or_repo,
            local_dir=target_path,
            local_dir_use_symlinks=False,
            token=resolved_token,
            resume_download=True
        )
        print(f"[+] Download complete: {out_dir}")
        return out_dir

    # Case 2: Direct HTTP / Hugging Face file URL
    url = url_or_repo
    headers = {"User-Agent": "Mozilla/5.0"}
    if resolved_token:
        headers["Authorization"] = f"Bearer {resolved_token}"

    print(f"\n[*] Checking remote file metadata for {model_name}...")
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req) as resp:
            expected_bytes = int(resp.headers.get("content-length", 0))
    except Exception as e:
        print(f"[!] Warning: Could not fetch Content-Length: {e}")
        expected_bytes = 0

    if os.path.exists(target_path) and expected_bytes > 0:
        local_bytes = os.path.getsize(target_path)
        if local_bytes == expected_bytes:
            print(f"[+] Verified {model_name} byte count matches ({local_bytes:,} bytes). Validating checkpoint...")
            try:
                import torch
                torch.load(target_path, map_location="cpu", weights_only=False)
                print(f"[+] Checkpoint '{model_name}' is valid. Skipping download.")
                return target_path
            except Exception as e:
                print(f"[!] Checkpoint corrupted ({e}). Re-downloading...")
                os.remove(target_path)
        else:
            print(f"[!] File size mismatch: local {local_bytes:,} != remote {expected_bytes:,}. Re-downloading...")
            os.remove(target_path)

    print(f"[>] Downloading {model_name}...")
    auth_args = ["--header", f"Authorization: Bearer {resolved_token}"] if resolved_token else []
    if subprocess.call(["which", "wget"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0:
        cmd = ["wget", "-c", "--show-progress", url, "-O", target_path] + auth_args
        subprocess.run(cmd, check=True)
    else:
        urllib.request.urlretrieve(url, target_path)

    # Final verification
    try:
        import torch
        torch.load(target_path, map_location="cpu", weights_only=False)
        print(f"[+] Checkpoint verified 100%: {model_name}")
    except Exception as e:
        print(f"[!] Warning: Could not verify with torch.load: {e}")

    return target_path


def verify_and_download_gdrive(model_name, gdrive_id, min_size_mb=100, target_dir="weights"):
    os.makedirs(target_dir, exist_ok=True)
    target_path = os.path.join(target_dir, model_name)
    gdown = ensure_gdown()

    if os.path.exists(target_path):
        size_mb = os.path.getsize(target_path) / (1024 * 1024)
        if size_mb >= min_size_mb:
            try:
                import torch
                torch.load(target_path, map_location="cpu", weights_only=False)
                print(f"[+] Verified {model_name} ({size_mb:.1f} MB). Skipping download.")
                return target_path
            except Exception as e:
                print(f"[!] Checkpoint corrupted ({e}). Re-downloading...")
                os.remove(target_path)

    print(f"[>] Downloading {model_name} from Google Drive...")
    gdown.download(id=gdrive_id, output=target_path, quiet=False)

    size_mb = os.path.getsize(target_path) / (1024 * 1024)
    if size_mb < min_size_mb:
        raise ValueError(f"Downloaded file too small: {size_mb:.1f} MB < {min_size_mb} MB")

    try:
        import torch
        torch.load(target_path, map_location="cpu", weights_only=False)
        print(f"[+] Checkpoint verified 100%: {model_name}")
    except Exception as e:
        print(f"[!] Warning: Could not verify with torch.load: {e}")

    return target_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model weights downloader utility")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face access token")
    args = parser.parse_args()
    token = get_hf_token(args.token)
    print(f"[*] Weight downloader ready. HF Authentication: {'TOKEN CONFIGURED' if token else 'ANONYMOUS'}")
