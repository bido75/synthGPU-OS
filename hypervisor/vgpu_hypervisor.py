"""
SynthGPU Hyper-Virtualized Control Plane — Lazy VRAM Allocation
===============================================================
Boots instantly with a 0-byte /dev/shm file. Only grows the backing
file incrementally when the C CUDA shim intercepts a cudaMalloc call.

Protocol:
  Client sends:  8-byte uint64 (requested bytes, little-endian)
  Server replies: 8-byte uint64 (byte offset into /dev/shm file,
                   or 0xFFFFFFFFFFFFFFFF on OOM)

The hypervisor is the absolute authority on memory limits.
"""

import os
import sys
import socket
import struct
import signal
import logging

logging.basicConfig(
    level=logging.INFO,
    format="[Hypervisor] %(asctime)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("vgpu")

# ── Configuration (override via environment) ─────────────────────
VRAM_SIZE_GB = int(os.environ.get("SYNTHGPU_VRAM_GB", "2"))
VRAM_BYTES   = VRAM_SIZE_GB * 1024 * 1024 * 1024
SHM_PATH     = os.environ.get("SYNTHGPU_SHM_PATH", "/dev/shm/synth_vgpu_vram")
UDS_PATH     = os.environ.get("SYNTHGPU_UDS_PATH", "/tmp/vgpu/control.sock")
UDS_DIR      = os.path.dirname(UDS_PATH)

OOM_SENTINEL = 0xFFFFFFFFFFFFFFFF


class VGPUHypervisor:
    """Control-plane memory manager — lazy allocation, bump allocator."""

    def __init__(self):
        self._offset = 0          # bump allocator cursor (bytes in shm)
        self._server = None

    # ── Public lifecycle ──────────────────────────────────────────

    def start(self):
        self._ensure_uds_dir()
        self._init_shm_zero()
        self._bind_uds()
        log.info("Control Plane ready on %s", UDS_PATH)
        log.info("Lazy VRAM: max %s GB, backing file at %s (0 bytes now)",
                 VRAM_SIZE_GB, SHM_PATH)

    def run_forever(self):
        signal.signal(signal.SIGINT, self._shutdown)
        signal.signal(signal.SIGTERM, self._shutdown)

        while True:
            try:
                conn, _ = self._server.accept()
            except (KeyboardInterrupt, SystemExit):
                break
            except OSError:
                break

            with conn:
                try:
                    self._handle_client(conn)
                except Exception as exc:
                    log.warning("client error: %s", exc)

        self._cleanup()

    # ── Internal: lazy shared memory pool ─────────────────────────

    def _ensure_uds_dir(self):
        os.makedirs(UDS_DIR, exist_ok=True)

    def _init_shm_zero(self):
        """Create a 0-byte backing file — instant, no I/O hammer."""
        if os.path.exists(SHM_PATH):
            os.remove(SHM_PATH)
        with open(SHM_PATH, "wb") as f:
            pass  # 0-byte file — Docker Desktop can handle this
        log.info("VRAM backing file created (0 bytes) at %s", SHM_PATH)

    def _grow_shm(self, new_size: int):
        """
        Grow the backing file to at least new_size bytes.
        Only expands — never shrinks. Uses page-aligned truncate,
        which is near-instant on tmpfs for incremental growth.
        """
        page_size = 4096
        aligned = (new_size + page_size - 1) & ~(page_size - 1)
        with open(SHM_PATH, "r+b") as f:
            f.truncate(aligned)

    # ── Internal: UDS server ──────────────────────────────────────

    def _bind_uds(self):
        if os.path.exists(UDS_PATH):
            os.remove(UDS_PATH)

        self._server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._server.bind(UDS_PATH)
        self._server.listen(128)
        os.chmod(UDS_PATH, 0o777)
        log.debug("UDS bound to %s", UDS_PATH)

    def _handle_client(self, conn):
        data = conn.recv(8)
        if not data or len(data) < 8:
            return

        requested_size = struct.unpack("<Q", data[:8])[0]

        if requested_size == 0:
            conn.sendall(struct.pack("<Q", 0))
            return

        # Page-align the request
        page_size = 4096
        aligned_size = (requested_size + page_size - 1) & ~(page_size - 1)

        # OOM check
        if self._offset + aligned_size > VRAM_BYTES:
            log.warning("OOM: requested %d bytes (%d aligned), "
                        "%d of %d used",
                        requested_size, aligned_size,
                        self._offset, VRAM_BYTES)
            conn.sendall(struct.pack("<Q", OOM_SENTINEL))
            return

        # Grow the backing file incrementally
        new_size = self._offset + aligned_size
        self._grow_shm(new_size)

        # Grant the offset
        granted_offset = self._offset
        self._offset += aligned_size

        conn.sendall(struct.pack("<Q", granted_offset))
        log.info("granted %d bytes (aligned %d) at offset %d — "
                 "shm now %d bytes",
                 requested_size, aligned_size,
                 granted_offset, new_size)

    # ── Shutdown ──────────────────────────────────────────────────

    def _shutdown(self, signum, frame):
        log.info("received signal %d, shutting down ...", signum)
        self._cleanup()
        sys.exit(0)

    def _cleanup(self):
        if self._server:
            try:
                self._server.close()
            except Exception:
                pass
        if os.path.exists(UDS_PATH):
            try:
                os.remove(UDS_PATH)
            except Exception:
                pass
        log.info("cleanup done")


if __name__ == "__main__":
    hv = VGPUHypervisor()
    hv.start()
    hv.run_forever()
