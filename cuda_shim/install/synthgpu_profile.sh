#!/usr/bin/env bash
# =================================================================
# SynthGPU CUDA Shim — Shell Profile Additions (synthgpu_profile.sh)
# =================================================================
# Add SynthGPU to your shell permanently.
#
# Usage (one-time setup):
#   echo 'source /usr/local/lib/synthgpu/synthgpu_profile.sh' >> ~/.bashrc
#   source ~/.bashrc
#
# Or to activate for a single session only:
#   source cuda_shim/install/synthgpu_profile.sh
# =================================================================

# Location of the installed shared library
SYNTHGPU_LIB_DIR="${SYNTHGPU_LIB_DIR:-/usr/local/lib/synthgpu}"
SYNTHGPU_LIB="$SYNTHGPU_LIB_DIR/libsynthgpu_cuda.so"

# Only activate if the library is actually installed
if [[ -f "$SYNTHGPU_LIB" ]]; then
    export LD_PRELOAD="${LD_PRELOAD:+$LD_PRELOAD:}$SYNTHGPU_LIB"
    export CUDA_VISIBLE_DEVICES=synthgpu0
    export SYNTHGPU_ACTIVE=1
    export SYNTHGPU_VRAM_MB="${SYNTHGPU_VRAM_MB:-4096}"

    # Shell function: run any command with SynthGPU active
    synthgpu() {
        LD_PRELOAD="$SYNTHGPU_LIB" SYNTHGPU_ACTIVE=1 "$@"
    }
    export -f synthgpu

    echo "[SynthGPU] CUDA shim loaded — Virtual GPU: SynthGPU Virtual Accelerator"
    echo "[SynthGPU] Virtual VRAM: ${SYNTHGPU_VRAM_MB} MB   Library: $SYNTHGPU_LIB"
else
    echo "[SynthGPU] Library not found at $SYNTHGPU_LIB"
    echo "[SynthGPU] Run: bash cuda_shim/install/install_linux.sh"

    # Still provide the helper function (no-op without the library)
    synthgpu() {
        echo "[SynthGPU] Shim not installed. Run install_linux.sh first."
        return 1
    }
    export -f synthgpu
fi

# Convenience alias
alias synthgpu-info='python3 -c "from cuda_shim.kernels.bridge_api import get_telemetry; import json; print(json.dumps(get_telemetry(), indent=2))"'
