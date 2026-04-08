#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# install_deps.sh — Install all dependencies for the
# Parallel Image Processing System on Ubuntu / Debian
#
# Run ONCE before building:  bash scripts/install_deps.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

echo "===================================================="
echo " Parallel Image Processing System — Dependency Setup"
echo "===================================================="

# C++ toolchain + OpenMP (comes with GCC)
echo "[1/5] Installing C++ toolchain ..."
sudo apt-get update -qq
sudo apt-get install -y build-essential

# OpenCV
echo "[2/5] Installing OpenCV development libraries ..."
sudo apt-get install -y libopencv-dev

# MPI (Open MPI)
echo "[3/5] Installing Open MPI ..."
sudo apt-get install -y libopenmpi-dev openmpi-bin

# OpenCL headers + ICD loader + CPU runtime fallback
echo "[4/5] Installing OpenCL development/runtime packages ..."
sudo apt-get install -y ocl-icd-opencl-dev opencl-headers clinfo pocl-opencl-icd

# Python plotting dependencies
echo "[5/5] Installing Python 3 libraries ..."
sudo apt-get install -y python3-pip python3-numpy python3-pandas python3-matplotlib
pip3 install --user opencv-python 2>/dev/null || true   # Python OpenCV for test image gen

echo ""
echo "All dependencies installed."
echo ""
echo "Verify:"
g++     --version | head -1
mpic++  --version | head -1
pkg-config --modversion opencv4 2>/dev/null || pkg-config --modversion opencv 2>/dev/null || echo "  opencv: check 'pkg-config --modversion opencv4'"
clinfo >/dev/null 2>&1 && echo "  opencl: clinfo ok" || echo "  opencl: runtime not available (check GPU driver / ICD)"
python3 -c "import cv2; print('  python cv2:', cv2.__version__)"
echo ""
echo "Now run:  make all"
