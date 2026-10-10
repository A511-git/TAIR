#!/usr/bin/env python3
"""
Universal Batch Inference Runner for TAIR (Text-Aware Image Restoration)
Processes folders of degraded images (e.g. from ./uploaded_stuff) and outputs
high-quality restored images and text spotting predictions to ./results.
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

VALID_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff")

REQUIRED_WEIGHTS = [
    "realesrgan_s4_swinir_100k.pth",
    "DiffBIR_v2.1.pt",
    "sd2.1-base-zsnr-laionaes5.ckpt",
    "terediff_stage3.pt"
]


def find_valid_images(folder_path: Path, recursive: bool = False):
    if not folder_path.exists():
        raise FileNotFoundError(
            f"Input directory does not exist: {folder_path}. "
            "Please upload test images via Link 1 (Upload Portal) into ./uploaded_stuff first."
        )

    pattern = "**/*" if recursive else "*"
    images = [
        str(p) for p in folder_path.glob(pattern)
        if p.is_file() and p.suffix.lower() in VALID_IMAGE_EXTENSIONS
    ]
    return sorted(images)


def verify_weights(repo_dir: Path) -> bool:
    weights_dir = repo_dir / "weights"
    missing = []
    if not weights_dir.exists():
        missing = REQUIRED_WEIGHTS
    else:
        for w in REQUIRED_WEIGHTS:
            if not (weights_dir / w).exists():
                missing.append(w)
    
    if missing:
        print("\n" + "!" * 70)
        print(" [!] MISSING TAIR MODEL WEIGHTS IN ./weights/")
        print("!" * 70)
        for m in missing:
            print(f"    - Missing: ./weights/{m}")
        print("\n[*] To download and verify all required model weights automatically, run:")
        print("    bash download_weights.sh")
        print("!" * 70 + "\n")
        return False
    return True


def run_batch_inference(input_dir: str, output_dir: str, config: str = "configs/val/val_terediff.yaml",
                        config_testr: str = "testr/configs/TESTR/TESTR_R_50_Polygon.yaml",
                        recursive: bool = False):
    repo_dir = Path(__file__).resolve().parent.parent
    in_path = Path(input_dir).resolve()
    out_path = Path(output_dir).resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("      TAIR BATCH RESTORATION INFERENCE RUNNER")
    print("=" * 72)
    print(f"[*] Input Images Directory  : {in_path}")
    print(f"[*] Output Results Directory: {out_path}")
    print(f"[*] Model Configuration     : {config}")
    print("=" * 72)

    images = find_valid_images(in_path, recursive=recursive)
    print(f"[*] Found {len(images)} valid images to process.")
    if not images:
        print("[!] No images found to restore. Upload test images into ./uploaded_stuff first!")
        return False

    weights_ok = verify_weights(repo_dir)
    if not weights_ok:
        sys.exit(1)

    # Determine execution method: accelerate launch or python
    cmd = []
    if shutil.which("accelerate"):
        cmd = [
            "accelerate", "launch",
            str(repo_dir / "val.py"),
            "--config", str(repo_dir / config),
            "--config_testr", str(repo_dir / config_testr),
            "--input_dir", str(in_path),
            "--output_dir", str(out_path)
        ]
    else:
        cmd = [
            sys.executable,
            str(repo_dir / "val.py"),
            "--config", str(repo_dir / config),
            "--config_testr", str(repo_dir / config_testr),
            "--input_dir", str(in_path),
            "--output_dir", str(out_path)
        ]

    print(f"[*] Executing Command:\n    {' '.join(cmd)}\n")
    ret = subprocess.run(cmd, cwd=str(repo_dir))
    if ret.returncode == 0:
        print("\n" + "=" * 72)
        print(f"[+] Batch restoration complete! Outputs saved to:\n    {out_path}")
        print("=" * 72)
        return True
    else:
        print(f"[!] Batch inference finished with exit code {ret.returncode}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Universal Batch Inference Runner for TAIR")
    parser.add_argument("--input_dir", "-i", type=str, default="./uploaded_stuff", help="Input directory of degraded images (default: ./uploaded_stuff)")
    parser.add_argument("--output_dir", "-o", type=str, default="./results", help="Directory to save restored images (default: ./results)")
    parser.add_argument("--config", "-c", type=str, default="configs/val/val_terediff.yaml", help="Path to TAIR YAML configuration")
    parser.add_argument("--config_testr", type=str, default="testr/configs/TESTR/TESTR_R_50_Polygon.yaml", help="Path to TESTR config")
    parser.add_argument("--recursive", "-r", action="store_true", help="Search input directory recursively")
    args = parser.parse_args()

    run_batch_inference(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        config=args.config,
        config_testr=args.config_testr,
        recursive=args.recursive
    )


if __name__ == "__main__":
    main()
