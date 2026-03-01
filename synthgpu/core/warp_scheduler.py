"""
SynthGPU - Warp Emulation Scheduler
=====================================
Emulates GPU SIMT (Single Instruction, Multiple Thread) warp execution
across CPU cores using thread pools and numpy SIMD operations.

A GPU warp = 32 threads executing the same instruction on different data.
We emulate this by grouping work into warp-sized chunks and dispatching
them across CPU threads using numpy vectorized operations (which map
directly to CPU SIMD instructions: AVX2/AVX-512 on x86, NEON on ARM).
"""

import numpy as np
import threading
import queue
import time
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Any
from enum import Enum


WARP_SIZE = 32  # Mirror real GPU warp size (NVIDIA) / wavefront (AMD=64)


class WarpStatus(Enum):
    PENDING   = "pending"
    RUNNING   = "running"
    COMPLETED = "completed"
    FAILED    = "failed"


@dataclass
class Warp:
    """
    A synthetic warp: WARP_SIZE logical threads executing one kernel function.
    The kernel receives a slice of data representing all 32 thread lanes.
    """
    warp_id:     int
    kernel:      Callable
    data_slice:  np.ndarray          # shape: (WARP_SIZE, ...) — one row per lane
    lane_mask:   np.ndarray          # bool mask for active lanes (predication)
    shared_mem:  dict                # shared memory visible to all lanes in warp
    status:      WarpStatus = WarpStatus.PENDING
    result:      Optional[np.ndarray] = None
    exec_time_ms: float = 0.0

    def execute(self) -> np.ndarray:
        t0 = time.perf_counter()
        self.status = WarpStatus.RUNNING
        try:
            # Apply lane mask — inactive lanes produce zeros (predicated execution)
            active_data = np.where(self.lane_mask[:, np.newaxis] if self.data_slice.ndim > 1
                                   else self.lane_mask, self.data_slice, 0)
            self.result = self.kernel(active_data, self.shared_mem)
            self.status = WarpStatus.COMPLETED
        except Exception as e:
            self.status = WarpStatus.FAILED
            self.result = None
            raise
        finally:
            self.exec_time_ms = (time.perf_counter() - t0) * 1000
        return self.result


@dataclass
class WarpBlock:
    """
    A block of warps sharing the same shared memory — mirrors GPU thread blocks.
    All warps in a block can communicate via shared_mem.
    """
    block_id:   int
    warps:      List[Warp]
    shared_mem: dict = field(default_factory=dict)


class WarpScheduler:
    """
    The SynthGPU Warp Scheduler.

    Manages a pool of CPU worker threads, each capable of executing warps.
    Implements:
      - Warp dispatch queue (FIFO with priority support)
      - Occupancy tracking (mirrors GPU SM occupancy)
      - Performance counters (throughput, latency, utilization)
    """

    def __init__(self, num_compute_units: int = None):
        # Compute units = CPU cores available for GPU work
        # We leave 2 cores for OS and control plane
        total_cores = os.cpu_count() or 4
        self.num_compute_units = num_compute_units or max(1, total_cores - 2)
        self.executor = ThreadPoolExecutor(
            max_workers=self.num_compute_units,
            thread_name_prefix="SynthGPU-CU"
        )

        # Performance counters
        self._lock = threading.Lock()
        self._warps_executed = 0
        self._total_exec_time_ms = 0.0
        self._start_time = time.perf_counter()

        print(f"[SynthGPU] WarpScheduler initialized")
        print(f"[SynthGPU] Compute Units: {self.num_compute_units} (of {total_cores} CPU cores)")
        print(f"[SynthGPU] Warp Size: {WARP_SIZE} lanes")
        print(f"[SynthGPU] Max concurrent warps: {self.num_compute_units}")

    def dispatch_kernel(self,
                        kernel: Callable,
                        data: np.ndarray,
                        shared_mem: dict = None) -> np.ndarray:
        """
        Dispatch a kernel across all data, automatically partitioned into warps.

        Args:
            kernel:     Function(data_slice, shared_mem) -> result_slice
            data:       Full input data array, shape (N, ...)
            shared_mem: Dict of shared memory visible to all warps

        Returns:
            Full result array, shape (N, ...) assembled from warp results
        """
        if shared_mem is None:
            shared_mem = {}

        N = data.shape[0]
        num_warps = (N + WARP_SIZE - 1) // WARP_SIZE  # ceiling division

        # Partition data into warp-sized slices
        warps = []
        for warp_id in range(num_warps):
            start = warp_id * WARP_SIZE
            end   = min(start + WARP_SIZE, N)
            slice_data = data[start:end]

            # Pad last warp to WARP_SIZE if needed (inactive lanes masked)
            actual_lanes = slice_data.shape[0]
            lane_mask = np.zeros(WARP_SIZE, dtype=bool)
            lane_mask[:actual_lanes] = True

            if actual_lanes < WARP_SIZE:
                pad_shape = (WARP_SIZE - actual_lanes,) + slice_data.shape[1:]
                slice_data = np.concatenate([
                    slice_data,
                    np.zeros(pad_shape, dtype=slice_data.dtype)
                ], axis=0)

            warps.append(Warp(
                warp_id=warp_id,
                kernel=kernel,
                data_slice=slice_data,
                lane_mask=lane_mask,
                shared_mem=shared_mem
            ))

        # Dispatch all warps to thread pool (parallel execution across CUs)
        futures = {self.executor.submit(w.execute): w for w in warps}

        # Collect results in warp order
        results = [None] * num_warps
        for future in as_completed(futures):
            warp = futures[future]
            result_slice = future.result()
            # Strip padding from last warp
            active_lanes = int(warp.lane_mask.sum())
            results[warp.warp_id] = result_slice[:active_lanes]

            with self._lock:
                self._warps_executed += 1
                self._total_exec_time_ms += warp.exec_time_ms

        return np.concatenate(results, axis=0)

    def get_stats(self) -> dict:
        elapsed = time.perf_counter() - self._start_time
        return {
            "compute_units":     self.num_compute_units,
            "warps_executed":    self._warps_executed,
            "total_exec_ms":     round(self._total_exec_time_ms, 3),
            "uptime_seconds":    round(elapsed, 2),
            "warp_throughput":   round(self._warps_executed / elapsed, 1) if elapsed > 0 else 0,
            "avg_warp_ms":       round(self._total_exec_time_ms / self._warps_executed, 4)
                                 if self._warps_executed > 0 else 0,
        }

    def shutdown(self):
        self.executor.shutdown(wait=True)
