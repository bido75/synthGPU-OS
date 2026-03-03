#!/usr/bin/env bash
# ============================================================
# SynthGPU CUDA Shim — Linux/macOS install script
# Usage:  bash install.sh [--prefix /usr/local] [--dev]
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHIM_DIR="$SCRIPT_DIR"
BUILD_DIR="$SHIM_DIR/build"
PREFIX="${PREFIX:-/usr/local}"
DEV_MODE=0

# ── Argument parsing ─────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --prefix) PREFIX="$2"; shift 2 ;;
        --dev)    DEV_MODE=1; shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║  SynthGPU CUDA Shim — Installer                  ║"
echo "╚══════════════════════════════════════════════════╝"
echo "  Shim dir : $SHIM_DIR"
echo "  Build dir: $BUILD_DIR"
echo "  Prefix   : $PREFIX"
echo ""

# ── Dependency check ─────────────────────────────────────────
echo "==> Checking dependencies..."
MISSING=()
command -v cmake   >/dev/null 2>&1 || MISSING+=(cmake)
command -v gcc     >/dev/null 2>&1 || MISSING+=(gcc)
command -v python3 >/dev/null 2>&1 || MISSING+=(python3)
command -v pip3    >/dev/null 2>&1 || MISSING+=(pip3)

if [[ ${#MISSING[@]} -gt 0 ]]; then
    echo "ERROR: Missing required tools: ${MISSING[*]}"
    echo "  Ubuntu/Debian: sudo apt install cmake gcc python3-dev python3-pip libopenblas-dev"
    echo "  Fedora/RHEL:   sudo dnf install cmake gcc python3-devel python3-pip openblas-devel"
    exit 1
fi

# Check for OpenBLAS headers
if ! pkg-config --exists openblas 2>/dev/null; then
    echo "WARNING: OpenBLAS not found via pkg-config. Install libopenblas-dev for full matrix support."
fi

echo "  All required tools present."

# ── Python package install ───────────────────────────────────
echo ""
echo "==> Installing Python kernel layer..."
if [[ $DEV_MODE -eq 1 ]]; then
    pip3 install -e "$SHIM_DIR" --quiet
    echo "  Installed in editable (dev) mode."
else
    pip3 install "$SHIM_DIR" --quiet
    echo "  Installed."
fi

# ── C library build ──────────────────────────────────────────
echo ""
echo "==> Building C shared library..."
mkdir -p "$BUILD_DIR"
cmake -S "$SHIM_DIR" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$PREFIX" \
    -DCMAKE_C_FLAGS="-O3 -march=native" \
    2>&1 | tail -20

cmake --build "$BUILD_DIR" --parallel "$(nproc 2>/dev/null || echo 4)"

echo ""
echo "==> Installing C library to $PREFIX ..."
cmake --install "$BUILD_DIR"

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║  Install complete!                               ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""
echo "  To use (LD_PRELOAD intercept):"
echo "    export LD_PRELOAD=$PREFIX/lib/libsynthgpu_cuda.so"
echo "    python your_script.py"
echo ""
echo "  To run tests:"
echo "    pytest $SHIM_DIR/../tests/ -v"
echo ""
