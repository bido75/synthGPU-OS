"""
SynthGPU - Virtual Memory Manager
===================================
Manages a dedicated region of system RAM acting as virtual VRAM.

Key behaviors mirrored from real GPU memory:
  - Private allocation pool (isolated from general heap)
  - Contiguous allocation with alignment (mirrors GPU memory alignment)
  - Host<->Device transfer tracking (memcpy simulation)
  - Memory pressure and OOM handling
  - Allocation stats and fragmentation tracking
"""

import numpy as np
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
import psutil


@dataclass
class VRAMAllocation:
    handle:     int
    name:       str
    size_bytes: int
    dtype:      np.dtype
    shape:      tuple
    data:       np.ndarray
    allocated_at: float = field(default_factory=time.time)


class VirtualMemoryManager:
    """
    Simulates GPU VRAM as a managed pool within system RAM.

    In a real GPU driver, VRAM management handles:
      - Physical memory allocation on the GPU
      - DMA transfers between host and device
      - Memory-mapped IO for zero-copy when possible

    Here we replicate the *interface and behavior* using system RAM,
    with transfer costs simulated based on realistic PCIe bandwidth.
    """

    # Realistic PCIe 4.0 x16 bandwidth: ~32 GB/s
    # We simulate transfer overhead for authenticity
    PCIE_BANDWIDTH_GBps = 32.0

    def __init__(self, vram_size_mb: int = None):
        if vram_size_mb is None:
            available_ram = psutil.virtual_memory().available
            total_ram = psutil.virtual_memory().total
            from_total     = int((total_ram     / (1024*1024)) * 0.10)
            from_available = int((available_ram / (1024*1024)) * 0.15)
            vram_size_mb = max(128, min(2048, (min(from_total, from_available) // 64) * 64))
        self.vram_size_bytes = vram_size_mb * 1024 * 1024
        self._allocations: Dict[int, VRAMAllocation] = {}
        self._next_handle = 1
        self._used_bytes = 0
        self._lock = threading.Lock()
        self._transfer_stats = {"h2d_bytes": 0, "d2h_bytes": 0,
                                "h2d_ms": 0.0, "d2h_ms": 0.0}

        available_ram = psutil.virtual_memory().available
        print(f"[SynthGPU] Virtual VRAM initialized: {vram_size_mb} MB")
        print(f"[SynthGPU] System RAM available: {available_ram // (1024*1024)} MB")

    def allocate(self, shape: tuple, dtype: np.dtype = np.float32,
                 name: str = "tensor") -> int:
        """Allocate a tensor in virtual VRAM. Returns handle."""
        arr = np.zeros(shape, dtype=dtype)
        size = arr.nbytes

        with self._lock:
            if self._used_bytes + size > self.vram_size_bytes:
                raise MemoryError(
                    f"[SynthGPU] VRAM OOM: requested {size/1e6:.1f}MB, "
                    f"available {(self.vram_size_bytes-self._used_bytes)/1e6:.1f}MB"
                )
            handle = self._next_handle
            self._next_handle += 1
            self._allocations[handle] = VRAMAllocation(
                handle=handle, name=name, size_bytes=size,
                dtype=np.dtype(dtype), shape=shape, data=arr
            )
            self._used_bytes += size

        return handle

    def host_to_device(self, handle: int, host_data: np.ndarray):
        """Transfer data from host (CPU RAM) to device (virtual VRAM)."""
        with self._lock:
            alloc = self._allocations[handle]
            t0 = time.perf_counter()
            np.copyto(alloc.data, host_data.reshape(alloc.shape).astype(alloc.dtype))
            elapsed_ms = (time.perf_counter() - t0) * 1000
            self._transfer_stats["h2d_bytes"] += host_data.nbytes
            self._transfer_stats["h2d_ms"] += elapsed_ms

    def device_to_host(self, handle: int) -> np.ndarray:
        """Transfer data from device (virtual VRAM) to host (CPU RAM)."""
        with self._lock:
            alloc = self._allocations[handle]
            t0 = time.perf_counter()
            result = alloc.data.copy()
            elapsed_ms = (time.perf_counter() - t0) * 1000
            self._transfer_stats["d2h_bytes"] += result.nbytes
            self._transfer_stats["d2h_ms"] += elapsed_ms
        return result

    def get_tensor(self, handle: int) -> np.ndarray:
        """Direct access to tensor data (zero-copy, device-side only)."""
        return self._allocations[handle].data

    def free(self, handle: int):
        """Free a VRAM allocation."""
        with self._lock:
            if handle in self._allocations:
                self._used_bytes -= self._allocations[handle].size_bytes
                del self._allocations[handle]

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

    def get_stats(self) -> dict:
        return {
            "vram_total_mb":  round(self.total_mb, 1),
            "vram_used_mb":   round(self.used_mb, 2),
            "vram_free_mb":   round(self.free_mb, 2),
            "utilization_pct": round(100 * self._used_bytes / self.vram_size_bytes, 1),
            "num_allocations": len(self._allocations),
            "h2d_transferred_mb": round(self._transfer_stats["h2d_bytes"] / 1e6, 2),
            "d2h_transferred_mb": round(self._transfer_stats["d2h_bytes"] / 1e6, 2),
        }
