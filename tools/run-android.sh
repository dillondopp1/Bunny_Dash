#!/bin/bash
# Start the pole vault editor on an Android phone (inside proot-distro Linux).
cd "$(dirname "$0")/.."
PY=./.venv/bin/python
[ -x "$PY" ] || PY=python3
echo "Open Chrome on this phone at http://127.0.0.1:8765/pose-editor.html"
echo "Leave this window open while you use it. Press Ctrl+C to stop."
exec "$PY" tools/vault_app.py --no-browser "$@"
