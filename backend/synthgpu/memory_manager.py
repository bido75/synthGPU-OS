"""
SynthGPU - Virtual VRAM Manager v0.3 (8GB-optimized)
Pure accounting system — tracks allocation sizes without backing numpy arrays.
RAM consumption: ~1MB regardless of vram_size_mb setting.
"""

import threading
import time
import os
from collections import deque
from dataclasses import dataclass, field
from typing import Dict
import psutil


@dataclass
class VRAMAllocation:
    handle:     int
    name:       str
    size_bytes: int
    allocated_at: float = field(default_factory=time.time)


def _calculate_vram_budget_mb() -> int:
    total_mb = psutil.virtual_memory().total    / (1024 * 1024)
    avail_mb = psutil.virtual_memory().available / (1024 * 1024)
    from_total     = int(total_mb * 0.10)
    from_available = int(avail_mb * 0.15)
    safe_mb = min(from_total, from_available)
    safe_mb = max(128, min(2048, safe_mb))
    safe_mb = (safe_mb // 64) * 64
    env = os.environ.get("SYNTHGPU_VRAM_MB")
    if env:
        override = int(env)
        max_allowed = int(avail_mb * 0.40)
        safe_mb = min(override, max_allowed)
    return safe_mb


class VirtualMemoryManager:
    """
    Tracks virtual VRAM allocations by name and size.
    Does NOT allocate a backing memory pool — pure accounting.
    RAM consumption: ~1MB regardless of vram_size_mb setting.
    """

    PCIE_BANDWIDTH_GBps = 32.0

    @staticmethod
    def _calculate_vram_pool() -> int:
        """
        Returns safe vRAM pool size in MB.
        On constrained machines (<2GB free), uses conservative allocation.
        """
        mem = psutil.virtual_memory()
        available_mb = mem.available // (1024 * 1024)
        total_mb     = mem.total     // (1024 * 1024)

        if available_mb < 2048:
            # Constrained: cap at 128MB to leave room for inference
            pool_mb = min(128, available_mb // 8)
        else:
            # Normal: use 10% of available, cap at 512MB
            pool_mb = min(512, available_mb // 10)

        return max(64, pool_mb)

    def __init__(self, vram_size_mb: int = None):
        if vram_size_mb is None:
            vram_size_mb = self._calculate_vram_pool()
        avail_mb = psutil.virtual_memory().available / (1024 * 1024)
        mode = "constrained" if avail_mb < 2048 else "normal"
        print(f"[SynthGPU] vRAM pool: {vram_size_mb}MB ({mode} mode, {avail_mb:.0f}MB available)")
        self.vram_size_mb   = vram_size_mb
        self.vram_size_bytes = vram_size_mb * 1024 * 1024
        self._allocations: Dict[int, VRAMAllocation] = {}
        self._next_handle   = 1
        self._used_bytes    = 0
        self._lock          = threading.Lock()
        self._transfer_stats = {"h2d_bytes": 0, "d2h_bytes": 0,
                                "h2d_ms": 0.0, "d2h_ms": 0.0}
        self._allocation_history: deque = deque(maxlen=50)
        avail_mb = psutil.virtual_memory().available / (1024 * 1024)
        print(f"[SynthGPU] Virtual VRAM initialized: {vram_size_mb} MB (accounting only)")
        print(f"[SynthGPU] System RAM available: {avail_mb:.0f} MB")

    def allocate(self, shape: tuple, dtype=None, name: str = "tensor") -> int:
        import numpy as np
        np_dtype = np.dtype(dtype) if dtype is not None else np.dtype(np.float32)
        size = int(__import__('math').prod(shape)) * np_dtype.itemsize
        with self._lock:
            if self._used_bytes + size > self.vram_size_bytes:
                raise MemoryError(
                    f"[SynthGPU] VRAM OOM: requested {size/1e6:.1f}MB, "
                    f"available {(self.vram_size_bytes-self._used_bytes)/1e6:.1f}MB"
                )
            handle = self._next_handle
            self._next_handle += 1
            self._allocations[handle] = VRAMAllocation(
                handle=handle, name=name, size_bytes=size
            )
            self._used_bytes += size
            self._allocation_history.append({
                "t": round(time.time(), 3), "event": "alloc",
                "name": name, "size_mb": round(size / 1e6, 3),
                "used_mb": round(self._used_bytes / 1e6, 2),
            })
        return handle

    def host_to_device(self, handle: int, host_data) -> None:
        import numpy as np
        nbytes = host_data.nbytes if hasattr(host_data, 'nbytes') else 0
        size_gb = nbytes / 1e9
        sim_ms = (size_gb / self.PCIE_BANDWIDTH_GBps) * 1000
        with self._lock:
            self._transfer_stats["h2d_bytes"] += nbytes
            self._transfer_stats["h2d_ms"] += sim_ms

    def device_to_host(self, handle: int):
        import numpy as np
        with self._lock:
            alloc = self._allocations.get(handle)
            if alloc is None:
                return np.array([], dtype=np.float32)
            size_gb = alloc.size_bytes / 1e9
            sim_ms = (size_gb / self.PCIE_BANDWIDTH_GBps) * 1000
            self._transfer_stats["d2h_bytes"] += alloc.size_bytes
            self._transfer_stats["d2h_ms"] += sim_ms
        return np.array([], dtype=np.float32)

    def get_tensor(self, handle: int):
        import numpy as np
        return np.array([], dtype=np.float32)

    def free(self, handle: int):
        with self._lock:
            if handle in self._allocations:
                size = self._allocations[handle].size_bytes
                name = self._allocations[handle].name
                self._used_bytes -= size
                del self._allocations[handle]
                self._allocation_history.append({
                    "t": round(time.time(), 3), "event": "free",
                    "name": name, "size_mb": round(size / 1e6, 3),
                    "used_mb": round(self._used_bytes / 1e6, 2),
                })

    def free_all(self):
        with self._lock:
            self._allocations.clear()
            self._used_bytes = 0

    @property
    def used_mb(self) -> float:
        return self._used_bytes / (1024 * 1024)

    @property
    def free_mb(self) -> float:
        return (self.vram_size_bytes - self._used_bytes) / (1024 * 1024)

    @property
    def total_mb(self) -> float:
        return self.vram_size_bytes / (1024 * 1024)

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
            }

    def get_stats(self) -> dict:
        return self.get_telemetry()
