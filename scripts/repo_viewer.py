#!/usr/bin/env python3
"""
Custom Multi-Path Repository & NVMe Workspace HTTP File Viewer
Seamlessly serves repository code, results, and uploaded_stuff from NVMe storage.
Eliminates symlink restrictions and provides rich file navigation with image previews.
"""

import argparse
import html
import mimetypes
import os
import sys
import time
import urllib.parse
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path


class WorkspaceHTTPHandler(SimpleHTTPRequestHandler):
    repo_dir = Path(".").resolve()
    workspace_dir = Path("/opt/dlami/nvme/workspace").resolve()

    def translate_path(self, path):
        """Translates URL path to actual filesystem path, resolving NVMe redirects."""
        # Unquote and normalize
        path = urllib.parse.unquote(path.split("?", 1)[0].split("#", 1)[0])
        clean_path = path.strip("/")

        # Direct shortcuts to NVMe Workspace subdirectories
        if clean_path == "uploaded_stuff" or clean_path.startswith("uploaded_stuff/"):
            rel = clean_path[len("uploaded_stuff"):].lstrip("/")
            target = self.workspace_dir / "uploaded_stuff" / rel
            return str(target.resolve()) if target.exists() else str(target)

        if clean_path == "results" or clean_path.startswith("results/"):
            rel = clean_path[len("results"):].lstrip("/")
            target = self.workspace_dir / "results" / rel
            return str(target.resolve()) if target.exists() else str(target)

        if clean_path == "weights" or clean_path.startswith("weights/"):
            rel = clean_path[len("weights"):].lstrip("/")
            target = self.workspace_dir / "weights" / rel
            return str(target.resolve()) if target.exists() else str(target)

        if clean_path == "models" or clean_path.startswith("models/"):
            rel = clean_path[len("models"):].lstrip("/")
            target = self.workspace_dir / "models" / rel
            return str(target.resolve()) if target.exists() else str(target)

        if clean_path == "nvme_workspace" or clean_path.startswith("nvme_workspace/"):
            rel = clean_path[len("nvme_workspace"):].lstrip("/")
            target = self.workspace_dir / rel
            return str(target.resolve()) if target.exists() else str(target)

        # Default: serve from repo directory, resolving symlinks explicitly
        target = (self.repo_dir / clean_path).resolve()
        return str(target)

    def list_directory(self, path):
        """Generates a modern, dark-themed directory listing with breadcrumbs & image previews."""
        try:
            entries = os.listdir(path)
        except OSError:
            self.send_error(404, "Directory not found")
            return None

        entries.sort(key=lambda a: a.lower())
        r = []
        display_path = html.escape(urllib.parse.unquote(self.path))
        
        # Build breadcrumbs
        parts = [p for p in display_path.strip("/").split("/") if p]
        breadcrumb_html = '<a href="/" style="color:#38bdf8;text-decoration:none;">repo_root</a>'
        acc = ""
        for p in parts:
            acc += "/" + p
            breadcrumb_html += f' <span style="color:#64748b;">/</span> <a href="{acc}/" style="color:#38bdf8;text-decoration:none;">{p}</a>'

        r.append('<!DOCTYPE html>')
        r.append('<html lang="en">')
        r.append('<head>')
        r.append('<meta charset="utf-8">')
        r.append('<meta name="viewport" content="width=device-width, initial-scale=1">')
        r.append(f'<title>Viewer: {display_path}</title>')
        r.append('<style>')
        r.append('''
            :root {
                --bg: #0b0f19;
                --surface: #131c2e;
                --border: #1e293b;
                --text: #f1f5f9;
                --muted: #94a3b8;
                --accent: #38bdf8;
                --accent-bg: rgba(56, 189, 248, 0.12);
            }
            * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
            body { background: var(--bg); color: var(--text); padding: 2rem 1.5rem; }
            .container { max-width: 1100px; margin: 0 auto; }
            header { margin-bottom: 2rem; border-bottom: 1px solid var(--border); padding-bottom: 1.5rem; }
            h1 { font-size: 1.5rem; margin-bottom: 0.5rem; display: flex; align-items: center; gap: 0.5rem; }
            .breadcrumbs { font-size: 0.95rem; margin-top: 0.5rem; }
            .quick-nav { display: flex; gap: 0.75rem; flex-wrap: wrap; margin-top: 1.25rem; }
            .badge {
                display: inline-flex; align-items: center; gap: 0.4rem; padding: 0.4rem 0.85rem;
                border-radius: 8px; font-size: 0.85rem; font-weight: 600; text-decoration: none;
                background: var(--surface); color: var(--text); border: 1px solid var(--border);
                transition: all 0.15s ease;
            }
            .badge:hover { border-color: var(--accent); background: var(--accent-bg); color: var(--accent); }
            .badge-primary { background: var(--accent-bg); border-color: var(--accent); color: var(--accent); }
            .file-table { width: 100%; border-collapse: collapse; background: var(--surface); border-radius: 12px; overflow: hidden; border: 1px solid var(--border); }
            .file-table th { background: #0f172a; text-align: left; padding: 0.85rem 1rem; font-size: 0.8rem; text-transform: uppercase; color: var(--muted); border-bottom: 1px solid var(--border); }
            .file-table td { padding: 0.85rem 1rem; border-bottom: 1px solid var(--border); font-size: 0.92rem; }
            .file-table tr:last-child td { border-bottom: none; }
            .file-table tr:hover { background: rgba(255, 255, 255, 0.02); }
            .item-link { color: var(--text); text-decoration: none; display: inline-flex; align-items: center; gap: 0.5rem; }
            .item-link:hover { color: var(--accent); text-decoration: underline; }
            .is-dir { font-weight: 600; color: #38bdf8; }
            .meta { color: var(--muted); font-size: 0.85rem; }
            .thumb { max-height: 48px; max-width: 48px; border-radius: 4px; object-fit: cover; vertical-align: middle; border: 1px solid var(--border); }
        ''')
        r.append('</style>')
        r.append('</head>')
        r.append('<body>')
        r.append('<div class="container">')
        r.append('<header>')
        r.append(f'<h1>📁 TAIR Repository & NVMe Workspace Explorer</h1>')
        r.append(f'<div class="breadcrumbs">Location: {breadcrumb_html}</div>')
        r.append('<div class="quick-nav">')
        r.append('  <a href="/uploaded_stuff/" class="badge badge-primary">📂 NVMe Uploads (uploaded_stuff)</a>')
        r.append('  <a href="/results/" class="badge badge-primary">✨ NVMe Restored Results (results)</a>')
        r.append('  <a href="/weights/" class="badge">⚖️ NVMe Model Weights (weights)</a>')
        r.append('  <a href="/nvme_workspace/" class="badge">💾 Raw NVMe Disk Workspace</a>')
        r.append('  <a href="/" class="badge">🏠 Repo Code Root</a>')
        r.append('</div>')
        r.append('</header>')

        r.append('<table class="file-table">')
        r.append('<thead><tr><th>Name</th><th>Size</th><th>Last Modified</th></tr></thead>')
        r.append('<tbody>')

        # Parent directory row
        if display_path != "/" and display_path != "":
            parent = "/".join(display_path.rstrip("/").split("/")[:-1]) or "/"
            r.append(f'<tr><td><a href="{parent}" class="item-link is-dir">⤴ .. (Parent Directory)</a></td><td>-</td><td>-</td></tr>')

        for name in entries:
            # Skip hidden git or system files
            if name.startswith(".") and name != ".":
                continue

            fullname = os.path.join(path, name)
            display_name = name
            link_name = urllib.parse.quote(name)

            is_dir = os.path.isdir(fullname)
            if is_dir:
                display_name = name + "/"
                link_name += "/"
                size_str = "Folder"
                icon = "📁"
            else:
                try:
                    size = os.path.getsize(fullname)
                    if size < 1024:
                        size_str = f"{size} B"
                    elif size < 1024 * 1024:
                        size_str = f"{size / 1024:.1f} KB"
                    elif size < 1024 * 1024 * 1024:
                        size_str = f"{size / (1024 * 1024):.1f} MB"
                    else:
                        size_str = f"{size / (1024 * 1024 * 1024):.2f} GB"
                except Exception:
                    size_str = "-"

                ext = os.path.splitext(name)[1].lower()
                if ext in (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"):
                    icon = "🖼️"
                elif ext in (".pt", ".pth", ".ckpt", ".bin", ".safetensors"):
                    icon = "⚖️"
                elif ext in (".py", ".sh", ".yaml", ".yml", ".json", ".md", ".txt"):
                    icon = "📄"
                else:
                    icon = "📦"

            try:
                mtime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(os.path.getmtime(fullname)))
            except Exception:
                mtime = "-"

            r.append(f'<tr>')
            cls = "is-dir" if is_dir else ""
            r.append(f'<td><a href="{link_name}" class="item-link {cls}">{icon} {html.escape(display_name)}</a></td>')
            r.append(f'<td class="meta">{size_str}</td>')
            r.append(f'<td class="meta">{mtime}</td>')
            r.append(f'</tr>')

        r.append('</tbody></table></div></body></html>')
        encoded = '\n'.join(r).encode('utf-8', 'surrogateescape')
        self.send_response(200)
        self.send_header("Content-type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)
        return None


def run_viewer(repo_dir: str, port: int, workspace_dir: str = None):
    repo_path = Path(repo_dir).resolve()
    if workspace_dir:
        ws_path = Path(workspace_dir).resolve()
    elif Path("/opt/dlami/nvme/workspace").exists():
        ws_path = Path("/opt/dlami/nvme/workspace").resolve()
    else:
        ws_path = (repo_path / ".workspace").resolve()

    ws_path.mkdir(parents=True, exist_ok=True)
    (ws_path / "uploaded_stuff").mkdir(parents=True, exist_ok=True)
    (ws_path / "results").mkdir(parents=True, exist_ok=True)
    (ws_path / "weights").mkdir(parents=True, exist_ok=True)

    WorkspaceHTTPHandler.repo_dir = repo_path
    WorkspaceHTTPHandler.workspace_dir = ws_path

    server_address = ("", port)
    httpd = HTTPServer(server_address, WorkspaceHTTPHandler)
    print(f"[*] TAIR Multi-Path Repo & NVMe Viewer running on 0.0.0.0:{port}")
    print(f"    • Repo root      : {repo_path}")
    print(f"    • NVMe Workspace : {ws_path}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[*] Shutting down viewer...")
        httpd.server_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TAIR Multi-Path Repo & NVMe Workspace HTTP File Viewer")
    parser.add_argument("--repo", default=".", help="Repository root directory")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--workspace", default=None, help="NVMe workspace directory path")
    args = parser.parse_args()

    run_viewer(args.repo, args.port, args.workspace)
