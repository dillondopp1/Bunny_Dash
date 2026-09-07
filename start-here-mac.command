#!/bin/bash
# Double-click this file to start the pole vault editor on a Mac.
# (Linux works too: run ./start-here-mac.command from a terminal.)
cd "$(dirname "$0")" || exit 1

PY=""
for c in python3 python; do
  if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(0 if sys.version_info[:2] >= (3,9) else 1)' 2>/dev/null; then
    PY="$c"; break
  fi
done
if [ -z "$PY" ]; then
  echo "Python 3.9 or newer is not installed."
  echo "Get it from https://www.python.org/downloads/ then double-click this file again."
  read -r -p "Press return to close. " _
  exit 1
fi

if ! "$PY" -c 'import mediapipe, cv2, scipy' >/dev/null 2>&1; then
  echo "Setting up for the first time. This takes a few minutes and only happens once."
  "$PY" -m pip install --quiet --upgrade pip
  if ! "$PY" -m pip install --quiet -r requirements.txt; then
    echo
    echo "Setup failed. The message above says why."
    read -r -p "Press return to close. " _
    exit 1
  fi
  echo "Setup done."
fi

echo
echo "Where do you want to use the editor?"
echo "  1) Just this computer"
echo "  2) This computer and my phone on the same wifi"
printf "Choose 1 or 2 [1]: "
read -r choice
echo
if [ "$choice" = "2" ]; then
  exec "$PY" tools/vault_app.py --lan
else
  exec "$PY" tools/vault_app.py
fi
