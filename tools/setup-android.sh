#!/bin/bash
# Set up the pole vault editor to run on an Android phone.
#
# Run this INSIDE a proot-distro Ubuntu or Debian session, not in plain Termux.
# Termux itself does not use glibc, and the mediapipe, opencv, numpy and scipy
# builds on PyPI are glibc builds, so they will not install there.
#
#   proot-distro login ubuntu        # then, in the folder holding this project:
#   bash tools/setup-android.sh
set -e
cd "$(dirname "$0")/.."

echo "==> Checking where this is running"
if [ -n "$PREFIX" ] && [ -d "$PREFIX/bin" ] && [ ! -f /etc/os-release ]; then
  echo "This looks like plain Termux. Start a Linux session first:"
  echo "    pkg install -y proot-distro && proot-distro install ubuntu"
  echo "    proot-distro login ubuntu"
  exit 1
fi
ARCH="$(uname -m)"
echo "    architecture: $ARCH"
if [ "$ARCH" != "aarch64" ] && [ "$ARCH" != "x86_64" ]; then
  echo "    warning: no prebuilt packages for $ARCH, this will probably fail"
fi

echo "==> Installing system packages"
APT="apt-get"
[ "$(id -u)" -ne 0 ] && APT="sudo apt-get"
$APT update -qq
$APT install -y -qq python3 python3-venv python3-pip unzip curl
# opencv and mediapipe load these at import time. Package names moved around
# between Ubuntu releases, so try each and keep going.
for lib in libgl1 libgl1-mesa-glx libglib2.0-0t64 libglib2.0-0 libsm6 libxext6; do
  $APT install -y -qq "$lib" 2>/dev/null || true
done

echo "==> Creating a private Python environment (.venv)"
# Recent Ubuntu refuses pip installs into the system Python, so use a venv.
python3 -m venv .venv
./.venv/bin/pip install --quiet --upgrade pip wheel

echo "==> Installing Python packages (several minutes, downloads about 200 MB)"
./.venv/bin/pip install --quiet -r requirements.txt

echo "==> Checking the install"
./.venv/bin/python - <<'PY'
import cv2, numpy, scipy
import mediapipe as mp
# solutions is exposed as an attribute, not an importable submodule path
mp.solutions.pose.Pose(model_complexity=1)
mediapipe = mp
print("    opencv", cv2.__version__, "| numpy", numpy.__version__,
      "| scipy", scipy.__version__, "| mediapipe", mediapipe.__version__)
print("    pose model loaded")
PY

cat <<'MSG'

Setup finished. To start the editor:

    bash tools/run-android.sh

Then open Chrome on this phone at:

    http://127.0.0.1:8765/pose-editor.html

MSG
