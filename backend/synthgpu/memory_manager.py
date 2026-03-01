"""
SynthGPU - Virtual VRAM Manager v0.2
Manages a dedicated region of system RAM acting as virtual GPU VRAM.
"""

import numpy as np
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict
import psutil


@dataclass
class VRAMAllocation:
    handle: int
    name: str
    size_bytes: int
    dtype: np.dtype
    shape: tuple
    data: np.ndarray
    allocated_at: float = field(default_factory=time.time)


class VirtualMemoryManager:
    PCIE_BANDWIDTH_GBps = 32.0

    def __init__(self, vram_size_mb: int = 4096):
        self.vram_size_bytes = vram_size_mb * 1024 * 1024
        self._allocations: Dict[int, VRAMAllocation] = {}
        self._next_handle = 1
        self._used_bytes = 0
        self._lock = threading.Lock()
        self._transfer_stats = {"h2d_bytes": 0, "d2h_bytes": 0,
                                "h2d_ms": 0.0, "d2h_ms": 0.0}
        self._allocation_history: deque = deque(maxlen=50)

        available_ram = psutil.virtual_memory().available
        if self.vram_size_bytes > available_ram * 0.6:
            self.vram_size_bytes = int(available_ram * 0.4)
            vram_size_mb = self.vram_size_bytes // (1024 * 1024)

        print(f"[SynthGPU] Virtual VRAM initialized: {vram_size_mb} MB")
        print(f"[SynthGPU] System RAM available: {available_ram // (1024 * 1024)} MB")

    def allocate(self, shape: tuple, dtype=np.float32, name: str = "tensor") -> int:
        arr = np.zeros(shape, dtype=dtype)
        size = arr.nbytes
        with self._lock:
            if self._used_bytes + size > self.vram_size_bytes:
                raise MemoryError(
                    f"[SynthGPU] VRAM OOM: requested {size / 1e6:.1f}MB, "
                    f"available {(self.vram_size_bytes - self._used_bytes) / 1e6:.1f}MB"
                )
            handle = self._next_handle
            self._next_handle += 1
            self._allocations[handle] = VRAMAllocation(
                handle=handle, name=name, size_bytes=size,
                dtype=np.dtype(dtype), shape=shape, data=arr
            )
            self._used_bytes += size
            self._allocation_history.append({
                "t": round(time.time(), 3),
                "event": "alloc",
                "name": name,
                "size_mb": round(size / 1e6, 3),
                "used_mb": round(self._used_bytes / 1e6, 2),
            })
        return handle

    def host_to_device(self, handle: int, host_data: np.ndarray):
        with self._lock:
            alloc = self._allocations[handle]
            t0 = time.perf_counter()
            np.copyto(alloc.data, host_data.reshape(alloc.shape).astype(alloc.dtype))
            elapsed_ms = (time.perf_counter() - t0) * 1000
            self._transfer_stats["h2d_bytes"] += host_data.nbytes
            self._transfer_stats["h2d_ms"] += elapsed_ms

    def device_to_host(self, handle: int) -> np.ndarray:
        with self._lock:
            alloc = self._allocations[handle]
            t0 = time.perf_counter()
            result = alloc.data.copy()
            elapsed_ms = (time.perf_counter() - t0) * 1000
            self._transfer_stats["d2h_bytes"] += result.nbytes
            self._transfer_stats["d2h_ms"] += elapsed_ms
        return result

    def get_tensor(self, handle: int) -> np.ndarray:
        return self._allocations[handle].data

    def free(self, handle: int):
        with self._lock:
            if handle in self._allocations:
                size = self._allocations[handle].size_bytes
                name = self._allocations[handle].name
                self._used_bytes -= size
                del self._allocations[handle]
                self._allocation_history.append({
                    "t": round(time.time(), 3),
                    "event": "free",
                    "name": name,
                    "size_mb": round(size / 1e6, 3),
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
                "vram_total_mb": round(self.total_mb, 1),
                "vram_used_mb": round(self.used_mb, 2),
                "vram_free_mb": round(self.free_mb, 2),
                "utilization_pct": round(100 * self._used_bytes / self.vram_size_bytes, 1),
                "num_allocations": len(self._allocations),
                "h2d_transferred_mb": round(self._transfer_stats["h2d_bytes"] / 1e6, 2),
                "d2h_transferred_mb": round(self._transfer_stats["d2h_bytes"] / 1e6, 2),
                "allocation_history": list(self._allocation_history),
            }

    def get_stats(self) -> dict:
        return self.get_telemetry()
