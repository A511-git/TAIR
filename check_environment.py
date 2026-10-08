import sys
import os

def check_env():
    print("=" * 70)
    print("           TAIR CUDA & HARDWARE DIAGNOSTIC REPORT")
    print("=" * 70)
    
    # 1. Python info
    print(f"[*] Python version: {sys.version.split()[0]}")
    
    # 2. PyTorch & CUDA availability
    try:
        import torch
        print(f"[*] PyTorch version: {torch.__version__}")
        print(f"[*] PyTorch CUDA compiled version: {torch.version.cuda}")
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
            print(f"    - Device {i}: {name} (Compute Capability {cap[0]}.{cap[1]}, Memory: {total_mem_gb:.2f} GB)")
            
        arch_list = torch.cuda.get_arch_list() if hasattr(torch.cuda, "get_arch_list") else []
        print(f"[*] Supported PyTorch Arch List: {arch_list}")
        
        # Test CUDA kernel execution (catches "no kernel image is available for execution" on Blackwell)
        try:
            x = torch.randn(100, 100, device="cuda")
            y = torch.matmul(x, x)
            torch.cuda.synchronize()
            print("[+] Basic CUDA tensor computation: SUCCESS (kernels match GPU architecture)")
        except Exception as e:
            print(f"[!] CUDA execution test FAILED: {e}")
            if "no kernel image" in str(e).lower():
                print("[!] CRITICAL: Your PyTorch build lacks kernels for this GPU architecture (Blackwell sm_120/sm_100).")
                print("    Please install PyTorch with CUDA 12.8+ wheel (https://download.pytorch.org/whl/cu128).")
            return False

    except ImportError:
        print("[!] ERROR: PyTorch is not installed in the active environment.")
        return False

    # 3. Detectron2 C++ / CUDA extension check
    try:
        import detectron2
        from detectron2 import _C
        print(f"[+] Detectron2 imported successfully (version: {detectron2.__version__})")
        print("[+] detectron2._C compiled extension: AVAILABLE")
    except ImportError as e:
        print(f"[!] Detectron2 or detectron2._C not available: {e}")
        print("    Run: cd detectron2 && pip install -e . --no-build-isolation")

    # 4. TESTR C++ / CUDA extension check
    try:
        import adet
        from adet import _C
        print("[+] TESTR / AdelaiDet imported successfully")
        print("[+] adet._C compiled extension: AVAILABLE")
    except ImportError as e:
        print(f"[!] TESTR / AdelaiDet or adet._C not available: {e}")
        print("    Run: cd testr && pip install -e . --no-build-isolation")

    # 5. Attention Mechanism Check
    try:
        from terediff.model.config import Config, AttnMode
        print(f"[+] TAIR attention mode initialized to: {Config.attn_mode.name}")
        if Config.attn_mode == AttnMode.SDP:
            print("    (Using PyTorch Native Scaled Dot-Product Attention: Optimal & hardware-accelerated)")
        elif Config.attn_mode == AttnMode.XFORMERS:
            print("    (Using xFormers attention)")
    except Exception as e:
        print(f"[!] Could not check TAIR attention config: {e}")

    print("=" * 70)
    print("Environment check finished!")
    print("=" * 70)
    return True

if __name__ == "__main__":
    check_env()
