"""
SynthGPU - Virtual VRAM Manager v0.3 (Hardware-Aware, mmap-Backed)
=================================================================
Replaces the pure accounting system with real mmap-backed memory allocations.
Includes a Constrained Hardware Profile for systems with <16GB RAM.
"""

import threading
import time
import os
import mmap
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
import psutil
import numpy as np

# ── Constrained Hardware Profile Thresholds ──────────────────────────
CONSTRAINED_RAM_THRESHOLD_MB = 16 * 1024   # 16 GB
CONSTRAINED_POOL_MAX_MB      = 256          # hard cap in constrained mode
CONSTRAINED_POOL_SAFE_MB     = 128          # safer baseline
STANDARD_RAM_FRACTION        = 0.40         # 40% of host RAM in standard mode
DEGRADED_MATRIX_MAX_DIM      = 64           # max matrix dimension in degraded mode

# PCIe bandwidth simulation constant (GB/s)
PCIE_BANDWIDTH_GBPS = 32.0

# ── Sentinel byte for uninitialized mmap pages ──────────────────────
_UNINIT_BYTE = 0x00


def _probe_system_memory() -> dict:
    """Returns a snapshot of host memory state, never raises."""
    try:
        mem = psutil.virtual_memory()
        return {
            "total_bytes":  mem.total,
            "total_mb":     mem.total / (1024 * 1024),
            "available_mb": mem.available / (1024 * 1024),
            "percent_used": mem.percent,
        }
    except Exception:
        return {"total_bytes": 8 * 1024**3, "total_mb": 8192,
                "available_mb": 4096, "percent_used": 50.0}


def _is_constrained(system: dict = None) -> bool:
    """Returns True when total system RAM is below the 16GB threshold."""
    if system is None:
        system = _probe_system_memory()
    return system["total_mb"] < CONSTRAINED_RAM_THRESHOLD_MB


def _compute_vram_budget_mb(override_mb: Optional[int] = None) -> Tuple[int, bool, bool]:
    """
    Returns (pool_size_mb, is_constrained, is_degraded).

    - >=16GB total RAM: 40% of host RAM, full mmap backing.
    - <16GB total RAM:  hard-cap at 128-256MB, degraded mode active.
    - ENV override `SYNTHGPU_VRAM_MB` always respected (with safety clamp).
    """
    system = _probe_system_memory()
    constrained = _is_constrained(system)
    degraded = constrained  # degraded mode always follows constrained mode

    if override_mb is not None:
        pool_mb = int(override_mb)
    else:
        env_val = os.environ.get("SYNTHGPU_VRAM_MB")
        if env_val is not None and env_val.strip():
            pool_mb = int(env_val)
        else:
            pool_mb = None

    if pool_mb is not None:
        if constrained:
            safe_max = int(system["available_mb"] * 0.40)
            pool_mb = min(pool_mb, safe_max, CONSTRAINED_POOL_MAX_MB)
        else:
            safe_max = int(system["available_mb"] * 0.80)
            pool_mb = min(pool_mb, safe_max)
        pool_mb = max(64, pool_mb)
    elif constrained:
        pool_mb = min(CONSTRAINED_POOL_SAFE_MB,
                      int(system["available_mb"] * 0.25))
        pool_mb = max(64, pool_mb)
    else:
        pool_mb = int(system["total_mb"] * STANDARD_RAM_FRACTION)
        avail_cap = int(system["available_mb"] * 0.75)
        pool_mb = min(pool_mb, avail_cap)
        pool_mb = max(256, pool_mb)

    pool_mb = (pool_mb // 64) * 64  # round down to 64 MB boundary
    return pool_mb, constrained, degraded


# ── Allocation tracking dataclass (preserved from v0.2) ─────────────
@dataclass
class VRAMAllocation:
    handle:      int
    name:        str
    size_bytes:  int
    shape:       Tuple[int, ...] = ()
    dtype_str:   str = "float32"
    allocated_at: float = field(default_factory=time.time)


class VirtualMemoryManager:
    """
    Hardware-aware Virtual VRAM Manager with mmap-backed storage.

    Two operating modes:
      1. Standard (>=16GB RAM):   40% of host RAM, real mmap allocations.
      2. Constrained (<16GB RAM):  128-256MB hard cap, degraded flag active.

    All existing telemetry and accounting surfaces are preserved.
    Never raises unhandled MemoryError.
    """

    def __init__(self, vram_size_mb: Optional[int] = None):
        pool_mb, self._constrained, self._degraded = \
            _compute_vram_budget_mb(vram_size_mb)

        system = _probe_system_memory()
        self.vram_size_mb = pool_mb
        self.vram_size_bytes = pool_mb * 1024 * 1024

        # ── Memory-mapped backing store ─────────────────────────────
        self._mmap: Optional[mmap.mmap] = None
        self._backed = False
        self._alloc_mmap()

        # ── Accounting state (preserved from v0.2) ───────────────────
        self._allocations: Dict[int, VRAMAllocation] = {}
        self._next_handle = 1
        self._used_bytes = 0
        self._lock = threading.Lock()
        self._transfer_stats = {
            "h2d_bytes": 0, "d2h_bytes": 0,
            "h2d_ms": 0.0,  "d2h_ms": 0.0,
        }
        self._allocation_history: deque = deque(maxlen=50)

        # ── Startup report ───────────────────────────────────────────
        mode = "CONSTRAINED (degraded)" if self._degraded else \
               "CONSTRAINED" if self._constrained else "STANDARD"
        backed_str = "mmap-backed" if self._backed else "accounting-only"
        print(f"[SynthGPU] vRAM pool: {self.vram_size_mb}MB [{mode}] "
              f"({backed_str}, {system['available_mb']:.0f}MB available)")
        if self._degraded:
            print(f"[SynthGPU] Degraded Simulation Mode ACTIVE — "
                  f"matrix ops capped at {DEGRADED_MATRIX_MAX_DIM}x{DEGRADED_MATRIX_MAX_DIM}")
        print(f"[SynthGPU] System RAM: {system['total_mb']:.0f}MB total, "
              f"{system['available_mb']:.0f}MB available")

    # ── Internal: mmap pool management ───────────────────────────────

    def _alloc_mmap(self) -> None:
        """Allocate the mmap backing store. Falls back silently."""
        if self.vram_size_bytes <= 0:
            self._backed = False
            return
        try:
            self._mmap = mmap.mmap(-1, self.vram_size_bytes,
                                   access=mmap.ACCESS_WRITE)
            self._backed = True
        except (mmap.error, OSError, MemoryError) as exc:
            self._mmap = None
            self._backed = False
            print(f"[SynthGPU] mmap allocation failed ({exc}) — "
                  f"falling back to accounting-only mode")

    def _ensure_mmap(self) -> bool:
        """Returns True if mmap is usable."""
        return self._backed and self._mmap is not None

    def _mmap_write(self, offset: int, data: bytes) -> bool:
        """Write bytes into the mmap pool at offset. Returns success."""
        if not self._ensure_mmap():
            return False
        end = offset + len(data)
        if end > self.vram_size_bytes:
            return False
        try:
            self._mmap.seek(offset)
            self._mmap.write(data)
            return True
        except Exception:
            return False

    def _mmap_read(self, offset: int, size: int) -> Optional[bytes]:
        """Read bytes from the mmap pool at offset. Returns None on failure."""
        if not self._ensure_mmap():
            return None
        if offset + size > self.vram_size_bytes:
            return None
        try:
            self._mmap.seek(offset)
            return self._mmap.read(size)
        except Exception:
            return None

    def _calculate_offset(self, handle: int) -> int:
        """
        Deterministic offset within the mmap pool for a given handle.
        Uses a simple linear layout: allocations are placed consecutively
        in allocation order. Returns 0 if handle unknown.
        """
        offset = 0
        for h in sorted(self._allocations.keys()):
            if h == handle:
                return offset
            offset += self._allocations[h].size_bytes
        return 0

    # ── Public API ───────────────────────────────────────────────────

    @property
    def is_constrained(self) -> bool:
        return self._constrained

    @property
    def is_degraded(self) -> bool:
        return self._degraded

    @property
    def degraded_matrix_max_dim(self) -> int:
        return DEGRADED_MATRIX_MAX_DIM if self._degraded else 0

    @staticmethod
    def get_degraded_matrix_max_dim() -> int:
        """Class-level accessor for ops that need to downscale."""
        return DEGRADED_MATRIX_MAX_DIM

    def allocate(self, shape: tuple, dtype=None, name: str = "tensor") -> int:
        """
        Allocate a block in virtual VRAM.

        Returns a negative handle on failure (never raises).
        """
        import numpy as np
        try:
            np_dtype = np.dtype(dtype) if dtype is not None else np.dtype(np.float32)
            size = int(math.prod(shape)) * np_dtype.itemsize
        except Exception:
            size = int(math.prod(shape)) if shape else 1024

        with self._lock:
            if self._used_bytes + size > self.vram_size_bytes:
                err_msg = (f"[SynthGPU] VRAM OOM: requested {size/1e6:.1f}MB, "
                           f"available {(self.vram_size_bytes-self._used_bytes)/1e6:.1f}MB")
                print(f"[SynthGPU] {err_msg}")
                return -1

            handle = self._next_handle
            self._next_handle += 1
            dtype_str = str(np_dtype) if 'np_dtype' in dir() else "float32"
            self._allocations[handle] = VRAMAllocation(
                handle=handle, name=name, size_bytes=size,
                shape=tuple(shape), dtype_str=dtype_str,
            )
            self._used_bytes += size
            self._allocation_history.append({
                "t": round(time.time(), 3), "event": "alloc",
                "name": name, "size_mb": round(size / 1e6, 3),
                "used_mb": round(self._used_bytes / 1e6, 2),
            })
        return handle

    def free(self, handle: int) -> None:
        """Free a previously allocated VRAM handle. Safe to call on invalid handles."""
        with self._lock:
            if handle not in self._allocations:
                return
            alloc = self._allocations.pop(handle)
            self._used_bytes -= alloc.size_bytes
            self._allocation_history.append({
                "t": round(time.time(), 3), "event": "free",
                "name": alloc.name, "size_mb": round(alloc.size_bytes / 1e6, 3),
                "used_mb": round(self._used_bytes / 1e6, 2),
            })

    def free_all(self) -> None:
        """Release all tracked allocations. Does not release mmap pool."""
        with self._lock:
            self._allocations.clear()
            self._used_bytes = 0
            self._allocation_history.append({
                "t": round(time.time(), 3), "event": "free_all",
                "name": "all", "size_mb": 0,
                "used_mb": 0,
            })

    def write(self, handle: int, data: np.ndarray, offset: int = 0) -> bool:
        """
        Write a numpy array into the backing mmap pool at the handle's offset.

        Args:
            handle: Allocation handle from allocate().
            data:   numpy array whose bytes will be written.
            offset: Byte offset within the allocation (not mmap pool).

        Returns True on success, False on failure (logs, never raises).
        """
        if handle < 0:
            return False
        with self._lock:
            alloc = self._allocations.get(handle)
            if alloc is None:
                return False
            data_bytes = data.tobytes() if hasattr(data, 'tobytes') else b''
            if not data_bytes:
                return True
            write_offset = self._calculate_offset(handle) + offset
            if offset + len(data_bytes) > alloc.size_bytes:
                return False
        return self._mmap_write(write_offset, data_bytes)

    def read(self, handle: int,
             shape: Optional[tuple] = None,
             dtype: object = None,
             offset: int = 0) -> np.ndarray:
        """
        Read a numpy array from the backing mmap pool.

        Args:
            handle: Allocation handle from allocate().
            shape:  Target shape (defaults to allocation shape).
            dtype:  Target dtype (defaults to float32).
            offset: Byte offset within the allocation.

        Returns a numpy array (possibly empty on failure). Never raises.
        """
        if handle < 0:
            return np.array([], dtype=np.float32)
        with self._lock:
            alloc = self._allocations.get(handle)
            if alloc is None:
                return np.array([], dtype=np.float32)

            read_shape = shape if shape is not None else alloc.shape
            np_dtype = np.dtype(dtype) if dtype is not None else np.dtype(np.float32)
            read_size = int(math.prod(read_shape)) * np_dtype.itemsize

            if offset + read_size > alloc.size_bytes:
                return np.array([], dtype=np.float32)

            mmap_offset = self._calculate_offset(handle) + offset

        raw = self._mmap_read(mmap_offset, read_size)
        if raw is None:
            return np.zeros(read_shape, dtype=np_dtype)
        try:
            arr = np.frombuffer(raw, dtype=np_dtype).reshape(read_shape).copy()
            return arr
        except Exception:
            return np.zeros(read_shape, dtype=np_dtype)

    def host_to_device(self, handle: int, host_data) -> None:
        """
        Simulate PCIe transfer from host to device VRAM.
        Also performs a real mmap write if backing is available.
        """
        import numpy as np
        nbytes = host_data.nbytes if hasattr(host_data, 'nbytes') else 0
        size_gb = nbytes / 1e9
        sim_ms = (size_gb / PCIE_BANDWIDTH_GBPS) * 1000

        with self._lock:
            self._transfer_stats["h2d_bytes"] += nbytes
            self._transfer_stats["h2d_ms"] += sim_ms

        if nbytes > 0 and isinstance(host_data, np.ndarray):
            self.write(handle, host_data)

    def device_to_host(self, handle: int) -> np.ndarray:
        """
        Simulate PCIe transfer from device VRAM to host.
        Returns the backed data if available, else an empty array.
        """
        import numpy as np
        with self._lock:
            alloc = self._allocations.get(handle)
            if alloc is None:
                return np.array([], dtype=np.float32)
            size_gb = alloc.size_bytes / 1e9
            sim_ms = (size_gb / PCIE_BANDWIDTH_GBPS) * 1000
            self._transfer_stats["d2h_bytes"] += alloc.size_bytes
            self._transfer_stats["d2h_ms"] += sim_ms
            read_shape = alloc.shape
            read_dtype = alloc.dtype_str

        if read_shape:
            try:
                return self.read(handle, shape=read_shape, dtype=read_dtype)
            except Exception:
                pass
        return np.array([], dtype=np.float32)

    def get_tensor(self, handle: int) -> np.ndarray:
        """Convenience alias for device_to_host."""
        return self.device_to_host(handle)

    # ── Properties (preserved from v0.2) ─────────────────────────────

    @property
    def used_mb(self) -> float:
        return self._used_bytes / (1024 * 1024)

    @property
    def free_mb(self) -> float:
        return (self.vram_size_bytes - self._used_bytes) / (1024 * 1024)

    @property
    def total_mb(self) -> float:
        return self.vram_size_bytes / (1024 * 1024)

    # ── Telemetry (preserved and extended from v0.2) ─────────────────

    def get_telemetry(self) -> dict:
        with self._lock:
            return {
                "vram_total_mb":      round(self.total_mb, 1),
                "vram_used_mb":       round(self.used_mb, 2),
                "vram_free_mb":       round(self.free_mb, 2),
                "utilization_pct":    round(100 * self._used_bytes / self.vram_size_bytes, 1),
                "num_allocations":    len(self._allocations),
                "h2d_transferred_mb": round(self._transfer_stats["h2d_bytes"] / 1e6, 2),
                "d2h_transferred_mb": round(self._transfer_stats["d2h_bytes"] / 1e6, 2),
                "allocation_history": list(self._allocation_history),
                "backed":             self._backed,
                "constrained_mode":   self._constrained,
                "degraded_mode":      self._degraded,
                "mmap_active":        self._ensure_mmap(),
            }

    def get_stats(self) -> dict:
        return self.get_telemetry()

    def __del__(self):
        """Clean up mmap on destruction."""
        if getattr(self, "_mmap", None) is not None:
            try:
                self._mmap.close()
            except Exception:
                pass
