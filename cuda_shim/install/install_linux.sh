#!/usr/bin/env bash
# =================================================================
# SynthGPU CUDA Shim — Linux Installer
# =================================================================
# Usage:
#   bash cuda_shim/install/install_linux.sh
#   bash cuda_shim/install/install_linux.sh --prefix /opt/synthgpu
# =================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHIM_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJ_DIR="$(cd "$SHIM_DIR/.." && pwd)"
PREFIX="${PREFIX:-/usr/local/lib/synthgpu}"
BUILD_DIR="$SHIM_DIR/build"

echo "==================================================="
echo "  SynthGPU CUDA Shim — Linux Installer"
echo "  Replaces NVIDIA CUDA for CPU-only machines"
echo "==================================================="
echo "  Project: $PROJ_DIR"
echo "  Prefix:  $PREFIX"
echo ""

# ── Check prerequisites ────────────────────────────────────────
echo "[0/5] Checking prerequisites..."
MISSING=()
for cmd in cmake gcc python3 pip3; do
    command -v "$cmd" &>/dev/null || MISSING+=("$cmd")
done
if [[ ${#MISSING[@]} -gt 0 ]]; then
    echo "ERROR: Missing tools: ${MISSING[*]}"
    echo "  Ubuntu: sudo apt install cmake gcc python3-dev python3-pip libopenblas-dev"
    exit 1
fi
echo "  All tools present."

# ── Build C library ────────────────────────────────────────────
echo ""
echo "[1/5] Building libsynthgpu_cuda.so..."
mkdir -p "$BUILD_DIR"
cmake -S "$SHIM_DIR" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$PREFIX" \
    2>&1 | tail -5
cmake --build "$BUILD_DIR" --parallel "$(nproc 2>/dev/null || echo 4)"

# ── Install library ────────────────────────────────────────────
echo ""
echo "[2/5] Installing to $PREFIX ..."
mkdir -p "$PREFIX"
sudo cp "$BUILD_DIR/libsynthgpu_cuda.so" "$PREFIX/"
sudo ldconfig

# ── Install Python kernels ─────────────────────────────────────
echo ""
echo "[3/5] Installing Python kernel layer..."
pip3 install -e "$SHIM_DIR" --quiet
echo "  Python kernels installed."

# ── Create activation script ───────────────────────────────────
echo ""
echo "[4/5] Creating activation script..."
cat > "$PREFIX/activate.sh" << ACTIVATE
#!/usr/bin/env bash
# Source this file to activate SynthGPU for the current session:
#   source $PREFIX/activate.sh
export LD_PRELOAD=$PREFIX/libsynthgpu_cuda.so
export CUDA_VISIBLE_DEVICES=synthgpu0
export SYNTHGPU_ROOT=$PROJ_DIR
export SYNTHGPU_VRAM_MB=\${SYNTHGPU_VRAM_MB:-4096}
echo "[SynthGPU] CUDA shim active — virtual GPU ready"
ACTIVATE
chmod +x "$PREFIX/activate.sh"

# ── Quick smoke test ───────────────────────────────────────────
echo ""
echo "[5/5] Running Python smoke test..."
python3 -c "
from cuda_shim.kernels.bridge_api import get_telemetry
t = get_telemetry()
print('  Python bridge: OK — shim_active =', t['shim_active'])
"

echo ""
echo "==================================================="
echo "  Installation complete!"
echo ""
echo "  Activate for a session:"
echo "    source $PREFIX/activate.sh"
echo ""
echo "  One-shot usage:"
echo "    LD_PRELOAD=$PREFIX/libsynthgpu_cuda.so python script.py"
echo ""
echo "  Verify (requires PyTorch):"
echo "    source $PREFIX/activate.sh"
echo "    python -c \"import torch; print(torch.cuda.is_available())\""
echo "==================================================="
