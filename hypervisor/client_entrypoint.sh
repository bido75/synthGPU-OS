#!/bin/bash
# =============================================================================
# SynthGPU AI Client Entrypoint — Socket Wait + LD_PRELOAD Injection
# =============================================================================
# Waits for the hypervisor's Unix Domain Socket to appear, then
# execs the user's command with LD_PRELOAD set so every CUDA call
# is intercepted by libsynthgpu.so.
#
# Usage:
#   docker run <image> /usr/local/bin/client_entrypoint.sh python -c "import torch; ..."
# =============================================================================

set -euo pipefail

SHIM_PATH="${LD_PRELOAD:-/usr/local/lib/synthgpu/libsynthgpu.so}"
UDS_PATH="${SYNTHGPU_UDS_PATH:-/tmp/vgpu/control.sock}"
TIMEOUT_SEC="${SYNTHGPU_WAIT_TIMEOUT:-120}"
INTERVAL_SEC="${SYNTHGPU_WAIT_INTERVAL:-1}"

echo "[entrypoint] Waiting for hypervisor socket at ${UDS_PATH} ..."

elapsed=0
while [ ! -S "${UDS_PATH}" ]; do
    if [ "${elapsed}" -ge "${TIMEOUT_SEC}" ]; then
        echo "[entrypoint] ERROR: Hypervisor socket not found after ${TIMEOUT_SEC}s"
        echo "[entrypoint] Ensure synthgpu-hypervisor container is running"
        exit 1
    fi
    sleep "${INTERVAL_SEC}"
    elapsed=$((elapsed + INTERVAL_SEC))
done

echo "[entrypoint] Hypervisor socket detected after ~${elapsed}s"

# Inject the CUDA shim via LD_PRELOAD
if [ -f "${SHIM_PATH}" ]; then
    export LD_PRELOAD="${SHIM_PATH}"
    echo "[entrypoint] LD_PRELOAD set to ${SHIM_PATH}"
else
    echo "[entrypoint] WARNING: ${SHIM_PATH} not found — CUDA will NOT be intercepted"
fi

export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:/usr/local/lib/synthgpu"

echo "[entrypoint] Executing: $*"
exec "$@"
