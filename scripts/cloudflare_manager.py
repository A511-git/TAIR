#!/usr/bin/env python3
"""
Cloudflare Quick Tunnel Manager
Manages automated ephemeral reverse proxies via cloudflared without requiring
accounts, tokens, or custom domains.
"""

import os
import platform
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path
from typing import Dict, Optional, Tuple


class CloudflareTunnelManager:
    OFFICIAL_DOWNLOAD_URLS = {
        "Linux_x86_64": "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64",
        "Linux_aarch64": "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-arm64",
        "Windows_AMD64": "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe",
        "Windows_x86_64": "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe"
    }

    def __init__(self, bin_dir: Optional[str] = None):
        if bin_dir is None:
            self.bin_dir = Path(".remote_bridge/bin").resolve()
        else:
            self.bin_dir = Path(bin_dir).resolve()
        
        self.bin_dir.mkdir(parents=True, exist_ok=True)
        self.cloudflared_path = self._resolve_or_download_binary()
        self.active_tunnels: Dict[int, Dict] = {}  # port -> {process, url, label}

    def _resolve_or_download_binary(self) -> str:
        # 1. Check if cloudflared is already in PATH
        found_in_path = shutil.which("cloudflared")
        if found_in_path:
            return found_in_path

        # 2. Check standard installation directories
        candidates = [
            r"C:\Program Files (x86)\cloudflared\cloudflared.exe",
            r"C:\Program Files\cloudflared\cloudflared.exe",
            "/usr/local/bin/cloudflared",
            "/usr/bin/cloudflared"
        ]
        for c in candidates:
            if os.path.exists(c):
                return c

        # 3. Check local bin_dir
        binary_name = "cloudflared.exe" if platform.system() == "Windows" else "cloudflared"
        local_binary = self.bin_dir / binary_name
        if local_binary.exists():
            return str(local_binary)

        # 4. Attempt automatic download
        system = platform.system()
        machine = platform.machine()
        key = f"{system}_{machine}"

        download_url = self.OFFICIAL_DOWNLOAD_URLS.get(key)
        if not download_url and system == "Linux" and ("x86_64" in machine or "amd64" in machine.lower()):
            download_url = self.OFFICIAL_DOWNLOAD_URLS["Linux_x86_64"]
        elif not download_url and system == "Windows":
            download_url = self.OFFICIAL_DOWNLOAD_URLS["Windows_AMD64"]

        if not download_url:
            raise RuntimeError(
                f"No automatic cloudflared download found for {system} ({machine}). "
                "Please install cloudflared manually: https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/"
            )

        print(f"[*] Downloading cloudflared binary for {system} ({machine})...")
        try:
            urllib.request.urlretrieve(download_url, str(local_binary))
            if system != "Windows":
                os.chmod(str(local_binary), 0o755)
            print(f"[+] Downloaded cloudflared to {local_binary}")
            return str(local_binary)
        except Exception as e:
            raise RuntimeError(f"Failed to download cloudflared: {e}")

    def start_tunnel(self, local_port: int, label: str = "service", timeout_seconds: int = 25) -> Optional[str]:
        """
        Starts an ephemeral Cloudflare Quick Tunnel forwarding to http://127.0.0.1:<local_port>.
        Extracts and returns the public https://*.trycloudflare.com URL.
        """
        if local_port in self.active_tunnels:
            return self.active_tunnels[local_port].get("url")

        cmd = [
            self.cloudflared_path,
            "tunnel",
            "--url", f"http://127.0.0.1:{local_port}",
            "--no-autoupdate"
        ]

        # Spawn subprocess
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )

        tunnel_url = None
        url_event = threading.Event()

        def stream_reader(pipe):
            nonlocal tunnel_url
            url_pattern = re.compile(r'https://[a-zA-Z0-9\-]+\.trycloudflare\.com')
            try:
                for line in iter(pipe.readline, ''):
                    if not line:
                        break
                    match = url_pattern.search(line)
                    if match:
                        tunnel_url = match.group(0)
                        url_event.set()
            except Exception:
                pass

        # cloudflared logs the tunnel URL to stderr
        t_err = threading.Thread(target=stream_reader, args=(proc.stderr,), daemon=True)
        t_out = threading.Thread(target=stream_reader, args=(proc.stdout,), daemon=True)
        t_err.start()
        t_out.start()

        # Wait for the URL to be parsed
        url_event.wait(timeout=timeout_seconds)

        if tunnel_url:
            self.active_tunnels[local_port] = {
                "process": proc,
                "url": tunnel_url,
                "label": label,
                "port": local_port
            }
            return tunnel_url
        else:
            # Check if process died
            retcode = proc.poll()
            if retcode is not None:
                print(f"[!] cloudflared process terminated prematurely with exit code {retcode}")
            else:
                proc.terminate()
            return None

    def stop_tunnel(self, local_port: int):
        if local_port in self.active_tunnels:
            info = self.active_tunnels[local_port]
            proc = info.get("process")
            if proc and proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    proc.kill()
            del self.active_tunnels[local_port]

    def stop_all(self):
        ports = list(self.active_tunnels.keys())
        for p in ports:
            self.stop_tunnel(p)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Cloudflare Quick Tunnel Launcher")
    parser.add_argument("--port", "-p", type=int, required=True, help="Local port to expose")
    parser.add_argument("--label", "-l", type=str, default="service", help="Service label")
    args = parser.parse_args()

    mgr = CloudflareTunnelManager()
    print(f"[*] Starting tunnel for port {args.port} ({args.label})...")
    url = mgr.start_tunnel(args.port, args.label)
    if url:
        print(f"[+] Tunnel Established: {url}")
        print("[*] Press Ctrl+C to close tunnel.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n[*] Stopping tunnel...")
            mgr.stop_all()
    else:
        print("[!] Failed to establish Cloudflare tunnel.")
        sys.exit(1)


if __name__ == "__main__":
    main()
