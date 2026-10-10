#!/usr/bin/env python3
"""
Universal File & Folder Upload Portal Server
Zero-dependency Python HTTP server that accepts files and whole folders from a web UI,
saving them into [repo]/uploaded_stuff with directory hierarchy preserved.
"""

import argparse
import html
import json
import os
import sys
import urllib.parse
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Upload Assets & Folders &bull; uploaded_stuff</title>
<style>
  :root {
    --bg: #0f172a;
    --card: #1e293b;
    --border: #334155;
    --accent: #38bdf8;
    --accent-hover: #0284c7;
    --text: #f8fafc;
    --muted: #94a3b8;
    --success: #22c55e;
    --danger: #ef4444;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
  body { background: var(--bg); color: var(--text); padding: 2rem 1rem; display: flex; justify-content: center; }
  .container { width: 100%; max-width: 900px; }
  header { margin-bottom: 2rem; border-bottom: 1px solid var(--border); padding-bottom: 1.25rem; }
  h1 { font-size: 1.75rem; font-weight: 700; color: #fff; display: flex; align-items: center; gap: 0.5rem; }
  p.subtitle { color: var(--muted); margin-top: 0.35rem; font-size: 0.95rem; }
  .tag { background: #0369a1; color: #e0f2fe; padding: 0.2rem 0.5rem; border-radius: 4px; font-family: monospace; font-size: 0.85rem; }
  
  .dropzone {
    border: 2px dashed var(--border);
    border-radius: 12px;
    background: var(--card);
    padding: 3rem 2rem;
    text-align: center;
    cursor: pointer;
    transition: all 0.2s ease;
    margin-bottom: 2rem;
  }
  .dropzone.dragover { border-color: var(--accent); background: rgba(56, 189, 248, 0.08); transform: scale(1.01); }
  .dropzone svg { width: 48px; height: 48px; fill: none; stroke: var(--accent); stroke-width: 1.5; margin-bottom: 1rem; }
  .dropzone h3 { font-size: 1.15rem; margin-bottom: 0.5rem; }
  .dropzone p { color: var(--muted); font-size: 0.9rem; margin-bottom: 1.5rem; }
  .btn-group { display: flex; justify-content: center; gap: 1rem; flex-wrap: wrap; }
  .btn {
    background: var(--accent);
    color: #0f172a;
    font-weight: 600;
    border: none;
    padding: 0.65rem 1.25rem;
    border-radius: 8px;
    cursor: pointer;
    font-size: 0.9rem;
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    transition: background 0.15s;
  }
  .btn:hover { background: var(--accent-hover); color: #fff; }
  .btn-outline {
    background: transparent;
    color: var(--text);
    border: 1px solid var(--border);
  }
  .btn-outline:hover { background: var(--border); }
  
  /* Progress Section */
  .progress-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1.25rem;
    margin-bottom: 2rem;
    display: none;
  }
  .progress-header { display: flex; justify-content: space-between; margin-bottom: 0.5rem; font-size: 0.9rem; }
  .progress-bar-bg { background: #334155; border-radius: 9999px; height: 8px; overflow: hidden; width: 100%; }
  .progress-bar { background: var(--accent); height: 100%; width: 0%; transition: width 0.15s; }
  .progress-file { font-size: 0.8rem; color: var(--muted); margin-top: 0.5rem; font-family: monospace; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }

  /* Table */
  .table-card { background: var(--card); border: 1px solid var(--border); border-radius: 8px; overflow: hidden; }
  .table-header { padding: 1rem 1.25rem; border-bottom: 1px solid var(--border); display: flex; justify-content: space-between; align-items: center; }
  .table-header h2 { font-size: 1.05rem; }
  table { width: 100%; border-collapse: collapse; text-align: left; font-size: 0.9rem; }
  th { padding: 0.75rem 1.25rem; background: rgba(0, 0, 0, 0.2); color: var(--muted); font-weight: 600; }
  td { padding: 0.75rem 1.25rem; border-top: 1px solid var(--border); }
  tr:hover td { background: rgba(255, 255, 255, 0.02); }
  .file-path { font-family: monospace; color: #38bdf8; }
  .del-btn { color: var(--danger); background: transparent; border: none; cursor: pointer; padding: 0.2rem 0.5rem; font-size: 0.8rem; border-radius: 4px; }
  .del-btn:hover { background: rgba(239, 68, 68, 0.15); }
  
  /* Helper Code Box */
  .helper-box {
    margin-top: 2rem;
    background: #020617;
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1rem;
    font-size: 0.85rem;
  }
  .helper-box code { color: #38bdf8; font-family: monospace; }
</style>
</head>
<body>
<div class="container">
  <header>
    <h1>Direct Upload Portal</h1>
    <p class="subtitle">Upload test images, LoRAs, or entire folders into <span class="tag">uploaded_stuff (NVMe Ephemeral Storage)</span></p>
  </header>

  <div class="dropzone" id="dropzone">
    <svg viewBox="0 0 24 24"><path d="M12 16.5V3m0 0l-4 4m4-4l4 4M4 14v4a2 2 0 002 2h12a2 2 0 002-2v-4"/></svg>
    <h3>Drag and drop files or folders here</h3>
    <p>Supports individual images/weights or full directory structures</p>
    <div class="btn-group" onclick="event.stopPropagation()">
      <button class="btn" onclick="document.getElementById('folderInput').click()">Upload Folder</button>
      <button class="btn btn-outline" onclick="document.getElementById('fileInput').click()">Select Files</button>
    </div>
    <input type="file" id="folderInput" webkitdirectory directory multiple style="display:none">
    <input type="file" id="fileInput" multiple style="display:none">
  </div>

  <div class="progress-card" id="progressCard">
    <div class="progress-header">
      <span id="progressText">Uploading 0/0 items...</span>
      <span id="progressPercent">0%</span>
    </div>
    <div class="progress-bar-bg"><div class="progress-bar" id="progressBar"></div></div>
    <div class="progress-file" id="currentFile">Ready</div>
  </div>

  <div class="table-card">
    <div class="table-header">
      <h2>Uploaded Items (<span id="itemCount">0</span>)</h2>
      <button class="btn btn-outline" style="padding: 0.35rem 0.75rem; font-size: 0.8rem;" onclick="refreshFiles()">Refresh</button>
    </div>
    <table>
      <thead>
        <tr>
          <th>Relative Path</th>
          <th>Size</th>
          <th>Modified</th>
          <th>Action</th>
        </tr>
      </thead>
      <tbody id="filesTableBody">
        <tr><td colspan="4" style="text-align:center; color: var(--muted);">No files uploaded yet.</td></tr>
      </tbody>
    </table>
  </div>

  <div class="helper-box">
    <strong>💡 Uploaded files are routed to NVMe Ephemeral Workspace to preserve root disk:</strong><br>
    Universal Batch Runner: <code>python scripts/batch_inference_runner.py -i ./uploaded_stuff -o ./results</code><br>
    Or subfolder: <code>python scripts/batch_inference_runner.py -i ./uploaded_stuff/my_folder -o ./results</code>
  </div>
</div>

<script>
const dropzone = document.getElementById('dropzone');
const folderInput = document.getElementById('folderInput');
const fileInput = document.getElementById('fileInput');
const progressCard = document.getElementById('progressCard');
const progressBar = document.getElementById('progressBar');
const progressText = document.getElementById('progressText');
const progressPercent = document.getElementById('progressPercent');
const currentFile = document.getElementById('currentFile');
const filesTableBody = document.getElementById('filesTableBody');
const itemCount = document.getElementById('itemCount');

// Drag and drop event listeners
dropzone.addEventListener('dragover', (e) => { e.preventDefault(); dropzone.classList.add('dragover'); });
dropzone.addEventListener('dragleave', () => { dropzone.classList.remove('dragover'); });
dropzone.addEventListener('drop', async (e) => {
  e.preventDefault();
  dropzone.classList.remove('dragover');
  const items = e.dataTransfer.items;
  if (!items) return;
  const filesToUpload = [];
  for (let i = 0; i < items.length; i++) {
    const entry = items[i].webkitGetAsEntry ? items[i].webkitGetAsEntry() : null;
    if (entry) {
      await traverseFileTree(entry, '', filesToUpload);
    } else {
      const f = items[i].getAsFile();
      if (f) filesToUpload.push({ file: f, path: f.name });
    }
  }
  uploadBatch(filesToUpload);
});

async function traverseFileTree(item, path, list) {
  path = path || "";
  if (item.isFile) {
    return new Promise((resolve) => {
      item.file((file) => {
        list.push({ file: file, path: path + file.name });
        resolve();
      });
    });
  } else if (item.isDirectory) {
    const dirReader = item.createReader();
    const entries = await new Promise((resolve) => {
      dirReader.readEntries((entries) => resolve(entries));
    });
    for (let i = 0; i < entries.length; i++) {
      await traverseFileTree(entries[i], path + item.name + "/", list);
    }
  }
}

folderInput.addEventListener('change', (e) => {
  const files = Array.from(e.target.files).map(f => ({
    file: f,
    path: f.webkitRelativePath || f.name
  }));
  uploadBatch(files);
});

fileInput.addEventListener('change', (e) => {
  const files = Array.from(e.target.files).map(f => ({
    file: f,
    path: f.name
  }));
  uploadBatch(files);
});

async function uploadBatch(items) {
  if (!items || items.length === 0) return;
  progressCard.style.display = 'block';
  let completed = 0;

  for (let i = 0; i < items.length; i++) {
    const item = items[i];
    currentFile.textContent = item.path;
    const pct = Math.round((completed / items.length) * 100);
    progressBar.style.width = pct + '%';
    progressPercent.textContent = pct + '%';
    progressText.textContent = `Uploading ${completed + 1}/${items.length} items...`;

    try {
      await uploadSingleFile(item.file, item.path);
    } catch (err) {
      console.error("Upload error:", err);
    }
    completed++;
  }

  progressBar.style.width = '100%';
  progressPercent.textContent = '100%';
  progressText.textContent = `Completed ${items.length} items!`;
  currentFile.textContent = 'Upload complete.';
  setTimeout(() => { progressCard.style.display = 'none'; }, 3000);
  refreshFiles();
}

function uploadSingleFile(file, relPath) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    const encodedPath = encodeURIComponent(relPath);
    xhr.open('POST', `/api/upload?path=${encodedPath}`, true);
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) resolve();
      else reject(new Error(xhr.responseText));
    };
    xhr.onerror = () => reject(new Error('Network error'));
    xhr.send(file);
  });
}

async function refreshFiles() {
  try {
    const res = await fetch('/api/files');
    const data = await res.json();
    itemCount.textContent = data.length;
    if (data.length === 0) {
      filesTableBody.innerHTML = '<tr><td colspan="4" style="text-align:center; color: var(--muted);">No files uploaded yet.</td></tr>';
      return;
    }
    filesTableBody.innerHTML = data.map(item => `
      <tr>
        <td class="file-path">${item.path}</td>
        <td>${formatBytes(item.size)}</td>
        <td>${item.mtime}</td>
        <td><button class="del-btn" onclick="deleteFile('${encodeURIComponent(item.path)}')">Delete</button></td>
      </tr>
    `).join('');
  } catch (e) {
    console.error("Failed to load files:", e);
  }
}

async function deleteFile(encodedPath) {
  if (!confirm("Are you sure you want to delete this file?")) return;
  await fetch(`/api/delete?path=${encodedPath}`, { method: 'DELETE' });
  refreshFiles();
}

function formatBytes(bytes) {
  if (bytes === 0) return '0 B';
  const k = 1024, dm = 2;
  const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
}

// Initial fetch
refreshFiles();
</script>
</body>
</html>
"""


class UploadHandler(BaseHTTPRequestHandler):
    target_dir: Path = Path("uploaded_stuff")

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/" or parsed.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_TEMPLATE.encode("utf-8"))
        elif parsed.path == "/api/files":
            files = self._list_uploaded_files()
            data = json.dumps(files).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        else:
            self.send_error(404, "Not Found")

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/upload":
            query = urllib.parse.parse_qs(parsed.query)
            rel_path_param = query.get("path", [None])[0]
            if not rel_path_param:
                self.send_error(400, "Missing 'path' query parameter")
                return

            rel_path = urllib.parse.unquote(rel_path_param).strip("/\\")
            # Path traversal check
            safe_target = (self.target_dir / rel_path).resolve()
            if not str(safe_target).startswith(str(self.target_dir.resolve())):
                self.send_error(403, "Forbidden path")
                return

            safe_target.parent.mkdir(parents=True, exist_ok=True)

            # Read raw body directly to avoid RAM explosion
            content_length = int(self.headers.get("Content-Length", 0))
            bytes_read = 0
            chunk_size = 64 * 1024

            with open(safe_target, "wb") as f:
                while bytes_read < content_length:
                    to_read = min(chunk_size, content_length - bytes_read)
                    chunk = self.rfile.read(to_read)
                    if not chunk:
                        break
                    f.write(chunk)
                    bytes_read += len(chunk)

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ok", "saved_path": rel_path}).encode("utf-8"))
        else:
            self.send_error(404, "Not Found")

    def do_DELETE(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/delete":
            query = urllib.parse.parse_qs(parsed.query)
            rel_path_param = query.get("path", [None])[0]
            if not rel_path_param:
                self.send_error(400, "Missing path")
                return

            rel_path = urllib.parse.unquote(rel_path_param).strip("/\\")
            safe_target = (self.target_dir / rel_path).resolve()
            if not str(safe_target).startswith(str(self.target_dir.resolve())):
                self.send_error(403, "Forbidden path")
                return

            if safe_target.exists():
                if safe_target.is_file():
                    safe_target.unlink()
                elif safe_target.is_dir():
                    import shutil
                    shutil.rmtree(safe_target)

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status":"deleted"}')
        else:
            self.send_error(404, "Not Found")

    def _list_uploaded_files(self):
        res = []
        if not self.target_dir.exists():
            return res

        for p in self.target_dir.glob("**/*"):
            if p.is_file():
                try:
                    stat = p.stat()
                    import datetime
                    mtime = datetime.datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
                    rel = str(p.relative_to(self.target_dir)).replace("\\", "/")
                    res.append({
                        "path": rel,
                        "size": stat.st_size,
                        "mtime": mtime
                    })
                except Exception:
                    pass
        res.sort(key=lambda x: x["path"])
        return res

    def log_message(self, format, *args):
        # Quiet down default stdout logs
        pass


def run_upload_server(repo_dir: str, port: int = 7865, upload_dir: str = None):
    repo_path = Path(repo_dir).resolve()
    if upload_dir:
        upload_target = Path(upload_dir).resolve()
    elif Path("/opt/dlami/nvme/workspace/uploaded_stuff").exists():
        upload_target = Path("/opt/dlami/nvme/workspace/uploaded_stuff").resolve()
    else:
        upload_target = repo_path / "uploaded_stuff"

    try:
        upload_target.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        import subprocess
        try:
            subprocess.run(["sudo", "mkdir", "-p", str(upload_target)], check=True)
            uid = os.getuid() if hasattr(os, "getuid") else 1000
            gid = os.getgid() if hasattr(os, "getgid") else 1000
            subprocess.run(["sudo", "chown", "-R", f"{uid}:{gid}", str(upload_target)], check=True)
        except Exception:
            pass

    UploadHandler.target_dir = upload_target

    server = HTTPServer(("0.0.0.0", port), UploadHandler)
    print(f"[+] Upload Portal Server running at http://127.0.0.1:{port}")
    print(f"[*] Target Upload Directory: {upload_target}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[*] Stopping Upload Portal Server...")
        server.server_close()


def main():
    parser = argparse.ArgumentParser(description="Direct File & Folder Upload Server")
    parser.add_argument("--repo", "-r", type=str, default=".", help="Repository root path")
    parser.add_argument("--upload-dir", "-u", type=str, default=None, help="Target upload directory path (overrides [repo]/uploaded_stuff)")
    parser.add_argument("--port", "-p", type=int, default=7865, help="Port to bind (default 7865)")
    args = parser.parse_args()

    run_upload_server(args.repo, args.port, upload_dir=args.upload_dir)


if __name__ == "__main__":
    main()
