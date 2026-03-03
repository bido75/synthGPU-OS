#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/../build"
LIB_DIR="/usr/local/lib/synthgpu"
ICD_D="/etc/vulkan/icd.d"

echo "[SynthGPU] Installing Vulkan ICD for Linux..."

if [ ! -f "$BUILD_DIR/libsynthgpu_vulkan_icd.so" ]; then
    echo "[ERROR] .so not found at $BUILD_DIR/libsynthgpu_vulkan_icd.so"
    echo "[ERROR] Build first: cd build && cmake .. && make -j\$(nproc)"
    exit 1
fi

sudo mkdir -p "$LIB_DIR"
sudo cp "$BUILD_DIR/libsynthgpu_vulkan_icd.so" "$LIB_DIR/"
sudo ldconfig

sudo mkdir -p "$ICD_D"
sudo cp "$SCRIPT_DIR/../manifests/synthgpu_icd_linux.json" "$ICD_D/synthgpu_icd.json"

echo "[SynthGPU] Installed:"
echo "  Library:  $LIB_DIR/libsynthgpu_vulkan_icd.so"
echo "  Manifest: $ICD_D/synthgpu_icd.json"
echo ""
echo "[SynthGPU] Verifying..."
vulkaninfo --summary 2>/dev/null | grep -i "SynthGPU" && \
    echo "[SynthGPU] SUCCESS — SynthGPU Virtual Accelerator detected!" || \
    echo "[SynthGPU] Run 'vulkaninfo --summary | grep SynthGPU' to verify."
