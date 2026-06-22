"""
SynthGPU - Main Device Interface v0.2
The top-level SynthGPU device — applications talk to this.
"""

import numpy as np
import time
import os
import platform
from typing import Optional, Dict, List, Generator

from synthgpu.warp_scheduler import WarpScheduler, WARP_SIZE
from synthgpu.memory_manager import VirtualMemoryManager
from synthgpu.ops import gpu_ops
from synthgpu._version import __version__


class SynthGPU:
    VERSION = __version__
    DEVICE_NAME = "SynthGPU Virtual Accelerator"

    def __init__(self, vram_mb: int = 4096, compute_units: int = None):
        print("=" * 60)
        print(f"  SynthGPU v{self.VERSION} — Initializing Virtual Device")
        print("=" * 60)
        print(f"  Platform: {platform.processor() or platform.machine()}")
        print(f"  OS:       {platform.system()} {platform.release()}")
        print(f"  Python:   {platform.python_version()}")
        print()

        self.memory = VirtualMemoryManager(vram_size_mb=vram_mb)
        print()
        self.scheduler = WarpScheduler(num_compute_units=compute_units)
        print()

        self._op_count = 0
        self._total_compute_ms = 0.0
        self._initialized_at = time.time()
        self._platform = platform.processor() or platform.machine()
        self._os = f"{platform.system()} {platform.release()}"

        # Pre-allocated buffers for repeated benchmark matmuls (avoids GC pressure)
        bench_dim = 16 if self.memory.is_degraded else 64
        self._bench_A = np.random.randn(bench_dim, bench_dim).astype(np.float32)
        self._bench_B = np.random.randn(bench_dim, bench_dim).astype(np.float32)

        # Self-test
        self._self_test()

        if self.memory.is_degraded:
            print(f"  [SynthGPU] [!] DEGRADED MODE — System RAM <16GB. "
                  f"Matrix ops capped at {self.memory.degraded_matrix_max_dim}x{self.memory.degraded_matrix_max_dim}.")
        print(f"  [SynthGPU] Device ready. "
              f"{self.scheduler.num_compute_units} compute units. "
              f"{vram_mb}MB vRAM allocated.")
        print("=" * 60)
        print()

    def _self_test(self):
        A = self._bench_A
        B = self._bench_B
        result = self.matmul(A, B)
        ref = np.matmul(A, B)
        max_diff = np.max(np.abs(result - ref))
        assert max_diff < 1e-4, f"Self-test FAILED: max_diff={max_diff}"
        print(f"[SynthGPU] Self-test passed (GEMM 64x64, max_diff={max_diff:.2e})")

    def matmul(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        t0 = time.perf_counter()
        if A.shape[0] >= WARP_SIZE:
            def row_kernel(data_slice, shared):
                return gpu_ops.gemm(data_slice, shared['B'])
            result = self.scheduler.dispatch_kernel(
                kernel=row_kernel, data=A, shared_mem={'B': B}
            )
        else:
            result = gpu_ops.gemm(A, B)
        self._record_op(time.perf_counter() - t0)
        return result

    def linear(self, x: np.ndarray, weight: np.ndarray,
               bias: Optional[np.ndarray] = None) -> np.ndarray:
        out = self.matmul(x, weight.T)
        if bias is not None:
            out = out + bias
        return out

    def relu(self, x: np.ndarray) -> np.ndarray:
        t0 = time.perf_counter()
        result = gpu_ops.relu(x)
        self._record_op(time.perf_counter() - t0)
        return result

    def gelu(self, x: np.ndarray) -> np.ndarray:
        t0 = time.perf_counter()
        result = gpu_ops.gelu(x)
        self._record_op(time.perf_counter() - t0)
        return result

    def softmax(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        return gpu_ops.softmax(x, axis)

    def layer_norm(self, x: np.ndarray, gamma: np.ndarray,
                   beta: np.ndarray) -> np.ndarray:
        return gpu_ops.layer_norm(x, gamma, beta)

    def attention(self, Q: np.ndarray, K: np.ndarray,
                  V: np.ndarray, mask=None) -> np.ndarray:
        t0 = time.perf_counter()
        result = gpu_ops.scaled_dot_product_attention(Q, K, V, mask)
        self._record_op(time.perf_counter() - t0)
        return result

    def embedding(self, indices: np.ndarray, weight_table: np.ndarray) -> np.ndarray:
        return gpu_ops.embedding_lookup(indices, weight_table)

    def run_mlp(self, x: np.ndarray, layers: List[Dict]) -> np.ndarray:
        h = x
        for layer in layers:
            h = self.linear(h, layer['weight'], layer.get('bias'))
            act = layer.get('activation', 'relu')
            if act == 'relu':
                h = self.relu(h)
            elif act == 'gelu':
                h = self.gelu(h)
            elif act == 'softmax':
                h = self.softmax(h)
        return h

    def run_transformer_block(self, x: np.ndarray, config: Dict) -> np.ndarray:
        batch, seq_len, d_model = x.shape
        num_heads = config['num_heads']
        d_k = d_model // num_heads

        h = self.layer_norm(x, config['gamma1'], config['beta1'])

        Q = self.linear(h, config['Wq']).reshape(batch, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)
        K = self.linear(h, config['Wk']).reshape(batch, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)
        V = self.linear(h, config['Wv']).reshape(batch, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)

        attn_out = self.attention(Q, K, V)
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(batch, seq_len, d_model)
        attn_out = self.linear(attn_out, config['Wo'])

        h = x + attn_out
        h2 = self.layer_norm(h, config['gamma2'], config['beta2'])

        ffn = self.linear(h2, config['W1'], config.get('b1'))
        ffn = self.gelu(ffn)
        ffn = self.linear(ffn, config['W2'], config.get('b2'))

        return h + ffn

    def run_inference(self, x: np.ndarray, model_config: Dict) -> np.ndarray:
        h = x
        num_layers = model_config.get('num_layers', 1)
        for i in range(num_layers):
            h = self.run_transformer_block(h, model_config['layers'][i])
        if 'lm_head' in model_config:
            h = self.linear(h, model_config['lm_head'])
            h = self.softmax(h, axis=-1)
        return h

    def generate_tokens(self, model_config: Dict, max_tokens: int = 20) -> Generator:
        d_model = model_config['layers'][0]['Wq'].shape[0]
        # In degraded mode, cap max_tokens to prevent memory pressure
        if self.memory.is_degraded:
            max_tokens = min(max_tokens, 5)
        for step in range(max_tokens):
            seq_len = step + 1
            x = np.random.randn(1, seq_len, d_model).astype(np.float32) * 0.1
            t0 = time.perf_counter()
            logits = self.run_inference(x, model_config)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            token_id = int(np.argmax(logits[0, -1, :]))
            yield {
                "step": step,
                "token_id": token_id,
                "ms": round(elapsed_ms, 2),
                "tokens_per_sec": round(1000 / elapsed_ms, 2) if elapsed_ms > 0 else 0,
            }

    def _record_op(self, elapsed_s: float):
        self._op_count += 1
        self._total_compute_ms += elapsed_s * 1000

    def get_telemetry(self) -> dict:
        """
        Returns current device telemetry. Never raises — uses getattr fallbacks.
        Includes scheduler+memory sub-dicts for the cuda_shim/status endpoint.
        """
        try:
            uptime     = round(time.time() - self._initialized_at, 1)
            sched_tele = self.scheduler.get_telemetry()
            mem_tele   = self.memory.get_telemetry()

            total_warps        = getattr(self.scheduler, 'total_warps_executed',
                                         sched_tele.get('warps_executed', 0))
            external_warps     = getattr(self.scheduler, 'external_warp_count', 0)
            throughput_samples = getattr(self.scheduler, '_throughput_samples', [])
            avg_throughput     = (sum(throughput_samples) / len(throughput_samples)
                                  if throughput_samples
                                  else sched_tele.get('warp_throughput_per_sec', 0.0))
            kernels            = getattr(self.scheduler, 'kernels_dispatched',
                                         sched_tele.get('warps_executed', total_warps))

            return {
                "device":           self.DEVICE_NAME,
                "version":          self.VERSION,
                "platform":         self._platform,
                "os":               self._os,
                "uptime_seconds":   uptime,
                "ops_executed":     self._op_count,
                "total_compute_ms": round(self._total_compute_ms, 2),
                "is_degraded":      self.memory.is_degraded,
                # Full sub-dicts (used by cuda_shim/status and live telemetry)
                "scheduler":        sched_tele,
                "memory":           mem_tele,
                # Flat shim fields (used by get_telemetry() callers expecting these keys)
                "warps_executed":         int(total_warps),
                "external_warps":         int(external_warps),
                "warp_throughput_per_sec": round(float(avg_throughput), 2),
                "kernels_dispatched":     int(kernels),
                "active_streams":         sched_tele.get('warps_in_flight', 0),
                "vram_used_mb":           int(mem_tele.get('vram_used_mb', 0)),
                "vram_total_mb":          int(mem_tele.get('vram_total_mb', 128)),
                "shim_version":           f"v{__version__}",
                "shim_active":            True,
            }
        except Exception:
            return {
                "device": "SynthGPU", "version": self.VERSION,
                "scheduler": {}, "memory": {},
                "warps_executed": 0, "external_warps": 0,
                "warp_throughput_per_sec": 0.0, "kernels_dispatched": 0,
                "active_streams": 0, "vram_used_mb": 0, "vram_total_mb": 128,
                "shim_version": f"v{__version__}", "shim_active": True,
            }

    def device_info(self) -> dict:
        return self.get_telemetry()

    def __repr__(self):
        return (f"SynthGPU({self.scheduler.num_compute_units} CUs, "
                f"{self.memory.total_mb:.0f}MB vRAM)")
