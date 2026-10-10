#!/usr/bin/env python3
"""
Hardware & CUDA Kernel Diagnostic for NVIDIA Blackwell GPUs (RTX PRO 4500, B100, B200, sm_100/sm_120)
Verifies Python, PyTorch, CUDA Toolkit, device capability, SDP attention, and TAIR extensions.
"""

import sys
import os

def check_env():
    print("=" * 72)
    print("         BLACKWELL GPU & RUNTIME ENVIRONMENT DIAGNOSTIC REPORT")
    print("=" * 72)

    # 1. Python info
    print(f"[*] Python version: {sys.version.split()[0]}")

    # 2. PyTorch & CUDA availability
    try:
        import torch
        print(f"[*] PyTorch version: {torch.__version__}")
        print(f"[*] PyTorch compiled CUDA: {torch.version.cuda}")
        cuda_avail = torch.cuda.is_available()
        print(f"[*] torch.cuda.is_available(): {cuda_avail}")

        if not cuda_avail:
            print("[!] WARNING: PyTorch does NOT detect CUDA. Check driver and CUDA installation.")
            return False

        device_count = torch.cuda.device_count()
        print(f"[*] Available GPU count: {device_count}")

        for i in range(device_count):
            name = torch.cuda.get_device_name(i)
            cap = torch.cuda.get_device_capability(i)
            total_mem_gb = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
            print(f"    - Device {i}: {name} (Compute Capability: sm_{cap[0]}{cap[1]}, VRAM: {total_mem_gb:.2f} GB)")

        arch_list = torch.cuda.get_arch_list() if hasattr(torch.cuda, "get_arch_list") else []
        print(f"[*] Supported PyTorch Arch List: {arch_list}")

        # Active CUDA tensor computation test: catches "no kernel image is available for execution"
        try:
            x = torch.randn(128, 128, device="cuda")
            y = torch.matmul(x, x)
            torch.cuda.synchronize()
            print("[+] CUDA tensor computation test: SUCCESS (Kernels match GPU architecture)")
        except Exception as e:
            print(f"[!] CUDA execution test FAILED: {e}")
            if "no kernel image" in str(e).lower():
                print("[!] CRITICAL: Your PyTorch build lacks kernels for this GPU architecture (Blackwell sm_100/sm_120).")
                print("    Fix: Install PyTorch with CUDA 12.8: uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128")
            return False

        # Scaled Dot-Product Attention (SDP) test
        try:
            import torch.nn.functional as F
            q = torch.randn(1, 4, 32, 64, device="cuda")
            k = torch.randn(1, 4, 32, 64, device="cuda")
            v = torch.randn(1, 4, 32, 64, device="cuda")
            out = F.scaled_dot_product_attention(q, k, v)
            torch.cuda.synchronize()
            print("[+] Native Scaled Dot-Product Attention (SDP): OPERATIONAL")
        except Exception as e:
            print(f"[!] Native SDP check failed: {e}")

    except ImportError:
        print("[!] ERROR: PyTorch is not installed in the active environment.")
        return False

    # 3. Detectron2 check
    try:
        import detectron2
        ver = getattr(detectron2, "__version__", "dev")
        try:
            from detectron2 import _C
            print(f"[+] Detectron2 imported successfully (version: {ver}) [Compiled C++ mode]")
        except ImportError:
            print(f"[+] Detectron2 imported successfully (version: {ver}) [Pure Python mode]")
    except ImportError as e:
        print(f"[*] Detectron2 not currently imported: {e}")

    # 4. TESTR C++ / CUDA extension check
    try:
        try:
            import adet
        except ImportError:
            sys.path.append(os.path.join(os.getcwd(), "testr"))
            import adet
        try:
            from adet import _C
            print("[+] TESTR / AdelaiDet imported successfully [Compiled CUDA mode]")
        except ImportError:
            print("[+] TESTR / AdelaiDet imported successfully [Pure PyTorch fallback mode]")
    except ImportError as e:
        print(f"[*] TESTR / AdelaiDet not currently imported: {e}")

    # 5. TAIR Attention Mode Check
    try:
        from terediff.model.config import Config, AttnMode
        print(f"[+] TAIR attention mode initialized to: {Config.attn_mode.name}")
    except Exception:
        pass

    print("=" * 72)
    print("Environment diagnostic completed successfully!")
    print("=" * 72)
    return True

if __name__ == "__main__":
    success = check_env()
    sys.exit(0 if success else 1)
