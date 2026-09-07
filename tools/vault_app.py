#!/usr/bin/env python3
"""Local app for turning a vault video into an editable stick figure.

Start it, drop a video on the page, and it splits the clip into frames, runs
pose detection on each, and hands you the result in the pose editor with the
skeleton and pole drawn on every frame, ready to drag.

    python tools/vault_app.py

Then use the page it opens. Everything stays on this machine; nothing is
uploaded anywhere. Processed clips land in data/ and can be reopened later
from the dropdown at the top of the page.
"""
import argparse
import http.server
import secrets
import socket
import json
import os
import re
import shutil
import socketserver
import subprocess
import sys
import threading
import time
import urllib.parse
import uuid
import webbrowser

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(ROOT, "tools")
DATA = os.path.join(ROOT, "data")
UPLOADS = os.path.join(ROOT, "uploads")
FRAMES = os.path.join(DATA, "frames")
MAX_UPLOAD = 600 * 1024 * 1024

JOBS = {}
JOBS_LOCK = threading.Lock()


def can_process():
    """True when the Python pipeline can run here. Without it the page falls
    back to detecting in the browser, which needs nothing installed."""
    # find_spec locates the packages without importing them, which keeps
    # startup instant; importing mediapipe takes seconds.
    try:
        import importlib.util as u
        return all(u.find_spec(m) is not None for m in ("mediapipe", "cv2", "scipy"))
    except Exception:
        return False


CAN_PROCESS = False
# Set when serving beyond this machine; then /api/ and /data/ need ?k=TOKEN.
TOKEN = ""


def lan_ip():
    """Best guess at this machine's address on the local network."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))       # no packets are sent, just picks a route
        return s.getsockname()[0]
    except OSError:
        try:
            return socket.gethostbyname(socket.gethostname())
        except OSError:
            return "127.0.0.1"
    finally:
        s.close()


def slugify(name):
    base = os.path.splitext(os.path.basename(name))[0]
    slug = re.sub(r"[^A-Za-z0-9_-]+", "-", base).strip("-").lower()
    return slug or "vault"


def unique_name(slug):
    name, n = slug, 2
    while os.path.exists(os.path.join(DATA, name + ".json")):
        name, n = f"{slug}-{n}", n + 1
    return name


def set_job(job_id, **kw):
    with JOBS_LOCK:
        JOBS.setdefault(job_id, {}).update(kw)


def get_job(job_id):
    with JOBS_LOCK:
        return dict(JOBS.get(job_id, {}))


def run_step(cmd, job_id, on_line):
    """Run a pipeline step, feeding each stdout line to on_line for progress."""
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1, cwd=ROOT)
    tail = []
    for line in proc.stdout:
        line = line.rstrip()
        tail.append(line)
        del tail[:-40]
        on_line(line)
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError("\n".join(tail[-12:]) or f"step failed: {cmd[1]}")
    return tail


def pipeline(job_id, video_path, name, bar_px):
    py = sys.executable
    raw = os.path.join(DATA, name + "_raw.json")
    out = os.path.join(DATA, name + ".json")
    frames_dir = os.path.join(FRAMES, name)
    total = [0]
    seen = [0]

    def on_extract(line):
        m = re.match(r"^(\d+) frames,", line)
        if m:
            total[0] = int(m.group(1))
            set_job(job_id, stage="Splitting the clip into frames", pct=4,
                    message=f"{total[0]} frames")
            return
        m = re.match(r"^scan (\d+)/(\d+)", line)
        if m and total[0]:
            done, n = int(m.group(1)), int(m.group(2))
            set_job(job_id, stage="Looking for the athlete", pct=5 + 35 * done / n,
                    message=f"frame {done} of {n}")
            return
        # Tracking runs forward from the anchor frame and then backward, so
        # count the frames reported rather than trusting their numbering.
        if re.match(r"^frame \d+:", line) and total[0]:
            seen[0] += 1
            done = min(seen[0], total[0])
            set_job(job_id, stage="Tracking the body through the vault",
                    pct=40 + 50 * done / total[0], message=f"frame {done} of {total[0]}")

    try:
        set_job(job_id, stage="Starting", pct=1, message="")
        run_step([py, os.path.join(TOOLS, "extract_pose.py"), video_path, raw,
                  "--frames-dir", frames_dir], job_id, on_extract)
        set_job(job_id, stage="Smoothing and building the vault", pct=92, message="")
        cmd = [py, os.path.join(TOOLS, "process_pose.py"), raw, out, "--name", name]
        if bar_px:
            cmd += ["--bar", str(bar_px)]
        run_step(cmd, job_id, lambda l: None)
        data = json.load(open(out))
        n_ok = len(data["frames"])
        set_job(job_id, stage="Ready", pct=100, done=True, name=name,
                message=f"{n_ok} frames tracked")
    except Exception as exc:
        set_job(job_id, stage="Failed", pct=100, done=True, error=str(exc))


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=ROOT, **kw)

    def log_message(self, fmt, *args):
        if "/api/" in (self.path or ""):
            sys.stderr.write("  %s\n" % (fmt % args))

    def _authorised(self, parts):
        if not TOKEN:
            return True
        if not (parts.path.startswith("/api/") or parts.path.startswith("/data/")):
            return True
        given = urllib.parse.parse_qs(parts.query).get("k", [""])[0]
        return secrets.compare_digest(given, TOKEN)

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parts = urllib.parse.urlsplit(self.path)
        if not self._authorised(parts):
            return self._json({"error": "bad or missing key"}, 403)
        q = urllib.parse.parse_qs(parts.query)
        if parts.path == "/api/job":
            return self._json(get_job(q.get("id", [""])[0]))
        if parts.path == "/api/vaults":
            found = []
            if os.path.isdir(DATA):
                for f in os.listdir(DATA):
                    if f.endswith(".json") and not f.endswith(("_keys.json", "_raw.json")):
                        full = os.path.join(DATA, f)
                        found.append((os.path.getmtime(full), f[:-5]))
            found.sort(reverse=True)          # newest first
            return self._json({"vaults": [n for _, n in found], "canProcess": CAN_PROCESS})
        if parts.path == "/":
            self.send_response(302)
            self.send_header("Location", "/pose-editor.html" + (f"?k={TOKEN}" if TOKEN else ""))
            self.end_headers()
            return
        self.path = parts.path          # the static handler chokes on ?k=...
        return super().do_GET()

    def do_POST(self):
        parts = urllib.parse.urlsplit(self.path)
        if not self._authorised(parts):
            return self._json({"error": "bad or missing key"}, 403)
        if parts.path != "/api/upload":
            return self._json({"error": "unknown endpoint"}, 404)
        if not CAN_PROCESS:
            return self._json({"error": "this copy has no pose packages installed"}, 503)
        q = urllib.parse.parse_qs(parts.query)
        filename = q.get("filename", ["clip.mp4"])[0]
        bar_px = q.get("bar", [""])[0]
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return self._json({"error": "empty upload"}, 400)
        if length > MAX_UPLOAD:
            return self._json({"error": "file larger than 600 MB"}, 413)
        os.makedirs(UPLOADS, exist_ok=True)
        os.makedirs(DATA, exist_ok=True)
        name = unique_name(slugify(filename))
        ext = os.path.splitext(filename)[1] or ".mp4"
        video_path = os.path.join(UPLOADS, name + ext)
        remaining = length
        with open(video_path, "wb") as fh:
            while remaining > 0:
                chunk = self.rfile.read(min(1 << 20, remaining))
                if not chunk:
                    break
                fh.write(chunk)
                remaining -= len(chunk)
        if remaining > 0:
            os.remove(video_path)
            return self._json({"error": "upload cut short"}, 400)
        job_id = uuid.uuid4().hex
        set_job(job_id, stage="Queued", pct=0, done=False, name=name, message="")
        threading.Thread(target=pipeline,
                         args=(job_id, video_path, name, bar_px), daemon=True).start()
        return self._json({"job": job_id, "name": name})


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default=None,
                    help="address to bind (default 127.0.0.1, or 0.0.0.0 with --lan)")
    ap.add_argument("--lan", action="store_true",
                    help="also serve to phones and tablets on the same wifi")
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()
    global TOKEN, CAN_PROCESS
    host = args.host or ("0.0.0.0" if args.lan else "127.0.0.1")
    open_to_network = args.lan or host not in ("127.0.0.1", "localhost")
    if open_to_network:
        TOKEN = secrets.token_urlsafe(9)
    for d in (DATA, UPLOADS, FRAMES):
        os.makedirs(d, exist_ok=True)
    suffix = f"?k={TOKEN}" if TOKEN else ""
    url = f"http://127.0.0.1:{args.port}/pose-editor.html{suffix}"
    CAN_PROCESS = can_process()
    srv = Server((host, args.port), Handler)
    print(f"Pole vault editor running at {url}")
    if not CAN_PROCESS:
        print("(clips will be processed in the browser; nothing to install)")
    if open_to_network:
        print()
        print("On your phone, on the same wifi, open:")
        print(f"    http://{lan_ip()}:{args.port}/pose-editor.html{suffix}")
        print()
        print("The key in that link is what keeps other people on the network out,")
        print("so treat the link as private. Restarting issues a new key.")
    print("Drop a video on the page to start. Ctrl+C here to stop.")
    if not args.no_browser:
        def open_browser():
            try:
                webbrowser.open(url)
            except Exception:
                pass          # no desktop to open one on, e.g. Termux; the URL is printed above
        threading.Timer(0.6, open_browser).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
