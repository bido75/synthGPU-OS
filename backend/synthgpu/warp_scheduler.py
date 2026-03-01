"""
SynthGPU - Warp Emulation Scheduler v0.2
Emulates GPU SIMT warp execution across CPU thread pool with live telemetry.
"""

import numpy as np
import threading
import time
import os
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Callable, List, Optional
from enum import Enum

WARP_SIZE = 32


class WarpStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Warp:
    warp_id: int
    kernel: Callable
    data_slice: np.ndarray
    lane_mask: np.ndarray
    shared_mem: dict
    status: WarpStatus = WarpStatus.PENDING
    result: Optional[np.ndarray] = None
    exec_time_ms: float = 0.0

    def execute(self) -> np.ndarray:
        t0 = time.perf_counter()
        self.status = WarpStatus.RUNNING
        try:
            active_data = np.where(
                self.lane_mask[:, np.newaxis] if self.data_slice.ndim > 1 else self.lane_mask,
                self.data_slice, 0
            )
            self.result = self.kernel(active_data, self.shared_mem)
            self.status = WarpStatus.COMPLETED
        except Exception:
            self.status = WarpStatus.FAILED
            self.result = None
            raise
        finally:
            self.exec_time_ms = (time.perf_counter() - t0) * 1000
        return self.result


class WarpScheduler:
    def __init__(self, num_compute_units: int = None):
        total_cores = os.cpu_count() or 4
        self.num_compute_units = num_compute_units or max(1, total_cores - 2)
        self.executor = ThreadPoolExecutor(
            max_workers=self.num_compute_units,
            thread_name_prefix="SynthGPU-CU"
        )
        self._lock = threading.Lock()
        self._warps_executed = 0
        self._warps_in_flight = 0
        self._total_exec_time_ms = 0.0
        self._start_time = time.perf_counter()
        self._warp_history: deque = deque(maxlen=100)
        self._last_throughput_check = time.perf_counter()
        self._warps_since_last_check = 0

        print(f"[SynthGPU] WarpScheduler initialized")
        print(f"[SynthGPU] Compute Units: {self.num_compute_units} (of {total_cores} CPU cores)")
        print(f"[SynthGPU] Warp Size: {WARP_SIZE} lanes")

    def dispatch_kernel(self, kernel: Callable, data: np.ndarray,
                        shared_mem: dict = None) -> np.ndarray:
        if shared_mem is None:
            shared_mem = {}
        N = data.shape[0]
        num_warps = (N + WARP_SIZE - 1) // WARP_SIZE
        warps = []
        for warp_id in range(num_warps):
            start = warp_id * WARP_SIZE
            end = min(start + WARP_SIZE, N)
            slice_data = data[start:end]
            actual_lanes = slice_data.shape[0]
            lane_mask = np.zeros(WARP_SIZE, dtype=bool)
            lane_mask[:actual_lanes] = True
            if actual_lanes < WARP_SIZE:
                pad_shape = (WARP_SIZE - actual_lanes,) + slice_data.shape[1:]
                slice_data = np.concatenate([
                    slice_data, np.zeros(pad_shape, dtype=slice_data.dtype)
                ], axis=0)
            warps.append(Warp(
                warp_id=warp_id, kernel=kernel,
                data_slice=slice_data, lane_mask=lane_mask,
                shared_mem=shared_mem
            ))

        with self._lock:
            self._warps_in_flight += len(warps)

        futures = {self.executor.submit(w.execute): w for w in warps}
        results = [None] * num_warps
        for future in as_completed(futures):
            warp = futures[future]
            result_slice = future.result()
            active_lanes = int(warp.lane_mask.sum())
            results[warp.warp_id] = result_slice[:active_lanes]
            with self._lock:
                self._warps_executed += 1
                self._warps_in_flight -= 1
                self._warps_since_last_check += 1
                self._total_exec_time_ms += warp.exec_time_ms
                now = time.perf_counter()
                elapsed = now - self._last_throughput_check
                if elapsed >= 0.2:
                    throughput = self._warps_since_last_check / elapsed
                    self._warp_history.append({
                        "t": round(now - self._start_time, 2),
                        "throughput": round(throughput, 2),
                        "utilization": min(100, round(
                            (self._warps_in_flight / max(1, self.num_compute_units)) * 100, 1
                        ))
                    })
                    self._warps_since_last_check = 0
                    self._last_throughput_check = now

        return np.concatenate(results, axis=0)

    def get_telemetry(self) -> dict:
        with self._lock:
            elapsed = time.perf_counter() - self._start_time
            throughput = self._warps_executed / elapsed if elapsed > 0 else 0
            avg_warp_ms = (self._total_exec_time_ms / self._warps_executed
                           if self._warps_executed > 0 else 0)
            utilization = min(100, round(
                (self._warps_in_flight / max(1, self.num_compute_units)) * 100, 1
            ))
            return {
                "compute_units": self.num_compute_units,
                "warp_size": WARP_SIZE,
                "warps_executed": self._warps_executed,
                "warps_in_flight": self._warps_in_flight,
                "warp_throughput_per_sec": round(throughput, 2),
                "avg_warp_ms": round(avg_warp_ms, 4),
                "utilization_pct": utilization,
                "uptime_seconds": round(elapsed, 2),
                "warp_history": list(self._warp_history),
            }

    def get_stats(self) -> dict:
        return self.get_telemetry()

    def shutdown(self):
        self.executor.shutdown(wait=True)
