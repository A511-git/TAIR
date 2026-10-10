#!/usr/bin/env python3
"""
Architecture-Aware Repository & Frontend Analyzer
Non-destructively inspects AI model repositories to identify:
1. Inbuilt frontends (Gradio, ComfyUI, Streamlit, Flask/FastAPI) and their launch ports/configs
2. Key entrypoints, batch inference scripts, and dataset/eval modules
3. Dependency signals and framework architectures
"""

import ast
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


class RepoAnalyzer:
    def __init__(self, repo_dir: str):
        self.repo_dir = Path(repo_dir).resolve()
        if not self.repo_dir.exists():
            raise FileNotFoundError(f"Repository directory does not exist: {self.repo_dir}")

    def analyze(self) -> Dict[str, Any]:
        result = {
            "repo_path": str(self.repo_dir),
            "repo_name": self.repo_dir.name,
            "has_frontend": False,
            "frontend_type": None,
            "frontend_entrypoint": None,
            "detected_port": None,
            "native_share_enabled": False,
            "frontend_details": {},
            "comfyui_detected": False,
            "comfyui_details": {},
            "inference_scripts": [],
            "requirements_present": (self.repo_dir / "requirements.txt").exists(),
            "architectures_detected": [],
            "gated_models_detected": [],
            "requires_hf_token": False,
            "fragile_code_patterns": []
        }

        # Scan for Gated Models and Fragile Patterns
        gated_models = self._detect_gated_models()
        if gated_models:
            result["gated_models_detected"] = gated_models
            result["requires_hf_token"] = True

        fragile_patterns = self._detect_fragile_patterns()
        result["fragile_code_patterns"] = fragile_patterns

        # 1. Scan for Frontends (Gradio, Streamlit, ComfyUI, Flask/FastAPI)
        gradio_info = self._detect_gradio()
        comfy_info = self._detect_comfyui()
        streamlit_info = self._detect_streamlit()
        web_server_info = self._detect_custom_web()

        if gradio_info:
            result["has_frontend"] = True
            result["frontend_type"] = "gradio"
            result["frontend_entrypoint"] = gradio_info.get("file")
            result["detected_port"] = gradio_info.get("port", 7860)
            result["native_share_enabled"] = gradio_info.get("share", False)
            result["frontend_details"] = gradio_info
        elif comfy_info and comfy_info.get("is_standalone_comfy"):
            result["has_frontend"] = True
            result["frontend_type"] = "comfyui"
            result["frontend_entrypoint"] = comfy_info.get("entrypoint")
            result["detected_port"] = comfy_info.get("port", 8188)
            result["frontend_details"] = comfy_info
        elif streamlit_info:
            result["has_frontend"] = True
            result["frontend_type"] = "streamlit"
            result["frontend_entrypoint"] = streamlit_info.get("file")
            result["detected_port"] = streamlit_info.get("port", 8501)
            result["frontend_details"] = streamlit_info
        elif web_server_info:
            result["has_frontend"] = True
            result["frontend_type"] = web_server_info.get("type", "web_server")
            result["frontend_entrypoint"] = web_server_info.get("file")
            result["detected_port"] = web_server_info.get("port", 8000)
            result["frontend_details"] = web_server_info

        if comfy_info:
            result["comfyui_detected"] = True
            result["comfyui_details"] = comfy_info

        # 2. Detect Inference / Evaluation scripts
        result["inference_scripts"] = self._detect_inference_scripts()

        return result

    def _detect_gradio(self) -> Optional[Dict[str, Any]]:
        # Candidate UI script names
        candidates = [
            "app.py", "app_low_VRAM.py", "webui.py", "demo.py", "gui.py",
            "ui.py", "gradio_app.py", "main.py", "run_gradio.py"
        ]

        # First check candidates in repo root
        found_files = []
        for name in candidates:
            p = self.repo_dir / name
            if p.is_file():
                found_files.append(p)

        # Also search top-level .py files if not found
        if not found_files:
            for p in self.repo_dir.glob("*.py"):
                found_files.append(p)

        for py_file in found_files:
            try:
                content = py_file.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue

            if "gradio" in content or "gr." in content:
                # Extract port and share flags non-destructively
                port = 7860
                share = False

                # Regex patterns for server_port=... or port=...
                port_match = re.search(r'(?:server_port|port)\s*=\s*(\d+)', content)
                if port_match:
                    try:
                        port = int(port_match.group(1))
                    except ValueError:
                        pass

                # Regex patterns for share=True/False
                share_match = re.search(r'share\s*=\s*(True|False)', content)
                if share_match:
                    share = (share_match.group(1) == "True")

                return {
                    "file": py_file.name,
                    "rel_path": str(py_file.relative_to(self.repo_dir)),
                    "port": port,
                    "share": share,
                    "uses_blocks": "gr.Blocks" in content,
                    "uses_interface": "gr.Interface" in content
                }

        return None

    def _detect_comfyui(self) -> Optional[Dict[str, Any]]:
        is_standalone = False
        entrypoint = None
        workflows = []

        # Check for standalone ComfyUI main.py
        comfy_main = self.repo_dir / "main.py"
        if comfy_main.is_file():
            try:
                content = comfy_main.read_text(encoding="utf-8", errors="ignore")
                if "comfy" in content or "ComfyUI" in content or "PromptServer" in content:
                    is_standalone = True
                    entrypoint = "main.py"
            except Exception:
                pass

        # Check for ComfyUI custom node directories or naming
        custom_node_dirs = []
        for item in self.repo_dir.iterdir():
            if item.is_dir() and "comfyui" in item.name.lower():
                custom_node_dirs.append(item.name)

        # Check for workflow JSONs
        workflow_dirs = [self.repo_dir / "workflow", self.repo_dir / "workflows"]
        for wdir in workflow_dirs:
            if wdir.is_dir():
                for j in wdir.glob("*.json"):
                    workflows.append(str(j.relative_to(self.repo_dir)))

        if is_standalone or custom_node_dirs or workflows:
            return {
                "is_standalone_comfy": is_standalone,
                "entrypoint": entrypoint,
                "port": 8188,
                "custom_node_dirs": custom_node_dirs,
                "workflows": workflows
            }

        return None

    def _detect_streamlit(self) -> Optional[Dict[str, Any]]:
        for py_file in self.repo_dir.glob("*.py"):
            try:
                content = py_file.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue

            if "import streamlit" in content or "from streamlit" in content:
                port = 8501
                port_match = re.search(r'--server\.port\s+(\d+)', content)
                if port_match:
                    port = int(port_match.group(1))

                return {
                    "file": py_file.name,
                    "rel_path": str(py_file.relative_to(self.repo_dir)),
                    "port": port
                }
        return None

    def _detect_custom_web(self) -> Optional[Dict[str, Any]]:
        for py_file in self.repo_dir.glob("*.py"):
            try:
                content = py_file.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue

            if "from fastapi" in content or "import fastapi" in content:
                port = 8000
                port_m = re.search(r'port\s*=\s*(\d+)', content)
                if port_m:
                    port = int(port_m.group(1))
                return {"type": "fastapi", "file": py_file.name, "port": port}

            if "from flask" in content or "import flask" in content:
                port = 5000
                port_m = re.search(r'port\s*=\s*(\d+)', content)
                if port_m:
                    port = int(port_m.group(1))
                return {"type": "flask", "file": py_file.name, "port": port}

        return None

    def _detect_inference_scripts(self) -> List[str]:
        patterns = ["test*.py", "eval*.py", "val*.py", "inference*.py", "demo*.py", "generate*.py"]
        found = set()
        for pat in patterns:
            for p in self.repo_dir.glob(pat):
                if p.is_file():
                    found.add(str(p.relative_to(self.repo_dir)))
            # Check 1 level deep (e.g. eval/t3_dataset.py)
            for p in self.repo_dir.glob(f"*/{pat}"):
                if p.is_file():
                    found.add(str(p.relative_to(self.repo_dir)))

        return sorted(list(found))

    def _detect_gated_models(self) -> List[str]:
        gated_signatures = [
            "black-forest-labs/FLUX",
            "meta-llama/Llama",
            "meta-llama/Meta-Llama",
            "google/gemma",
            "stabilityai/stable-diffusion-3",
            "mistralai/Mistral",
        ]
        found = set()
        for p in self.repo_dir.rglob("*.py"):
            if any(x in str(p) for x in [".git", "__pycache__", "venv", ".remote_bridge", "uploaded_stuff"]):
                continue
            try:
                content = p.read_text(encoding="utf-8", errors="ignore")
                for sig in gated_signatures:
                    if sig.lower() in content.lower():
                        found.add(sig)
            except Exception:
                pass
        for p in self.repo_dir.rglob("*.yaml"):
            try:
                content = p.read_text(encoding="utf-8", errors="ignore")
                for sig in gated_signatures:
                    if sig.lower() in content.lower():
                        found.add(sig)
            except Exception:
                pass
        return sorted(list(found))

    def _detect_fragile_patterns(self) -> List[Dict[str, str]]:
        issues = []
        for p in self.repo_dir.rglob("*.py"):
            if any(x in str(p) for x in [".git", "__pycache__", "venv", ".remote_bridge", "uploaded_stuff"]):
                continue
            rel = str(p.relative_to(self.repo_dir))
            try:
                content = p.read_text(encoding="utf-8", errors="ignore")
                if "np.int0" in content:
                    issues.append({"file": rel, "pattern": "np.int0", "fix": "Replace with np.int32 (removed in NumPy 2.x)"})
                if ".getsize(" in content and "ImageFont" in content:
                    issues.append({"file": rel, "pattern": ".getsize()", "fix": "Replace with .getbbox() (removed in Pillow 10+)"})
                if ".getoffset(" in content and "ImageFont" in content:
                    issues.append({"file": rel, "pattern": ".getoffset()", "fix": "Replace with .getbbox() (removed in Pillow 10+)"})
                if "USE_PEFT_BACKEND" in content and "from diffusers.models.transformers" in content:
                    issues.append({"file": rel, "pattern": "USE_PEFT_BACKEND in diffusers.models", "fix": "Import from diffusers.utils instead"})
                if "import ujson" in content and "try:" not in content:
                    issues.append({"file": rel, "pattern": "bare import ujson", "fix": "Add try/except fallback to import json as ujson"})
            except Exception:
                pass
        return issues


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Architecture-Aware Repository & Frontend Analyzer")
    parser.add_argument("--repo", "-r", type=str, default=".", help="Target repository directory path")
    parser.add_argument("--json", "-j", action="store_true", help="Output in JSON format")
    args = parser.parse_args()

    analyzer = RepoAnalyzer(args.repo)
    data = analyzer.analyze()

    if args.json:
        print(json.dumps(data, indent=2))
    else:
        print("=" * 64)
        print(f" REPO ARCHITECTURE ANALYSIS: {data['repo_name']}")
        print("=" * 64)
        print(f"[*] Inbuilt Frontend Detected : {data['has_frontend']}")
        if data['has_frontend']:
            print(f"    - Type                    : {data['frontend_type'].upper()}")
            print(f"    - Entrypoint              : {data['frontend_entrypoint']}")
            print(f"    - Detected Port           : {data['detected_port']}")
            print(f"    - Native Share Support    : {data['native_share_enabled']}")
        if data['comfyui_detected']:
            print(f"[*] ComfyUI Elements Detected :")
            if data['comfyui_details'].get('custom_node_dirs'):
                print(f"    - Custom Nodes            : {data['comfyui_details']['custom_node_dirs']}")
            if data['comfyui_details'].get('workflows'):
                print(f"    - Workflows               : {data['comfyui_details']['workflows']}")
        print(f"[*] Inference / Eval Scripts   : {len(data['inference_scripts'])} found")
        for s in data['inference_scripts'][:5]:
            print(f"    - {s}")
        if data.get('gated_models_detected'):
            print(f"[!] Gated Models Detected     : {data['gated_models_detected']}")
            print(f"    - Set HF_TOKEN in .env or environment for authentication!")
        if data.get('fragile_code_patterns'):
            print(f"[!] Fragile Patterns Detected : {len(data['fragile_code_patterns'])} items")
            for item in data['fragile_code_patterns'][:3]:
                print(f"    - {item['file']}: {item['pattern']} -> {item['fix']}")
        print("=" * 64)


if __name__ == "__main__":
    main()
