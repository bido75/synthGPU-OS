"""
SynthGPU - Main Device Interface
===================================
The top-level SynthGPU device. This is what applications talk to.

Usage:
    gpu = SynthGPU()
    result = gpu.matmul(A, B)
    logits = gpu.run_transformer_block(x, weights)

The device exposes a familiar GPU-like API while internally managing
the warp scheduler, memory manager, and operation dispatch.
"""

import numpy as np
import time
import os
import platform
from typing import Optional, Dict, Tuple, List

from synthgpu.core.warp_scheduler import WarpScheduler, WARP_SIZE
from synthgpu.core.memory_manager import VirtualMemoryManager
from synthgpu.ops import gpu_ops


class SynthGPU:
    """
    SynthGPU Virtual GPU Device.

    Presents a GPU-like compute interface backed entirely by CPU
    resources — no physical GPU required. Designed to be a drop-in
    accelerator for AI inference workloads.
    """

    VERSION = "0.1.0-mvp"
    DEVICE_NAME = "SynthGPU Virtual Accelerator"

    def __init__(self, vram_mb: int = 2048, compute_units: int = None):
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

        print(f"  Device ready: {self.DEVICE_NAME}")
        print("=" * 60)
        print()

    # ─────────────────────────────────────────
    #  Core Compute Operations
    # ─────────────────────────────────────────

    def matmul(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Matrix multiply — dispatched through warp scheduler."""
        t0 = time.perf_counter()
        # For large matrices: use warp-parallel row-wise dispatch
        if A.shape[0] >= WARP_SIZE:
            def row_kernel(data_slice, shared):
                B_mat = shared['B']
                return gpu_ops.gemm(data_slice, B_mat)
            result = self.scheduler.dispatch_kernel(
                kernel=row_kernel,
                data=A,
                shared_mem={'B': B}
            )
        else:
            result = gpu_ops.gemm(A, B)
        self._record_op(time.perf_counter() - t0)
        return result

    def linear(self, x: np.ndarray, weight: np.ndarray,
               bias: Optional[np.ndarray] = None) -> np.ndarray:
        """Linear layer: y = xW^T + b"""
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

    # ─────────────────────────────────────────
    #  High-Level Model Execution
    # ─────────────────────────────────────────

    def run_mlp(self, x: np.ndarray, layers: List[Dict]) -> np.ndarray:
        """
        Run a Multi-Layer Perceptron through the virtual GPU.
        layers: [{'weight': W, 'bias': b, 'activation': 'relu'|'gelu'}, ...]
        """
        h = x
        for layer in layers:
            h = self.linear(h, layer['weight'], layer.get('bias'))
            act = layer.get('activation', 'relu')
            if act == 'relu':  h = self.relu(h)
            elif act == 'gelu': h = self.gelu(h)
            elif act == 'softmax': h = self.softmax(h)
        return h

    def run_transformer_block(self, x: np.ndarray, config: Dict) -> np.ndarray:
        """
        Run one Transformer block (attention + FFN) through the virtual GPU.

        config keys:
            d_model, num_heads, d_ff
            Wq, Wk, Wv, Wo  — attention weight matrices
            W1, b1, W2, b2  — FFN weights
            gamma1, beta1, gamma2, beta2 — LayerNorm params
        """
        batch, seq_len, d_model = x.shape
        num_heads = config['num_heads']
        d_k = d_model // num_heads

        # — Layer Norm 1 —
        h = self.layer_norm(x, config['gamma1'], config['beta1'])

        # — Multi-Head Self-Attention —
        Q = self.linear(h, config['Wq']).reshape(batch, seq_len, num_heads, d_k).transpose(0,2,1,3)
        K = self.linear(h, config['Wk']).reshape(batch, seq_len, num_heads, d_k).transpose(0,2,1,3)
        V = self.linear(h, config['Wv']).reshape(batch, seq_len, num_heads, d_k).transpose(0,2,1,3)

        attn_out = self.attention(Q, K, V)
        attn_out = attn_out.transpose(0,2,1,3).reshape(batch, seq_len, d_model)
        attn_out = self.linear(attn_out, config['Wo'])

        # Residual 1
        h = x + attn_out

        # — Layer Norm 2 —
        h2 = self.layer_norm(h, config['gamma2'], config['beta2'])

        # — Feed-Forward Network —
        ffn = self.linear(h2, config['W1'], config.get('b1'))
        ffn = self.gelu(ffn)
        ffn = self.linear(ffn, config['W2'], config.get('b2'))

        # Residual 2
        return h + ffn

    def run_inference(self, x: np.ndarray, model_config: Dict) -> np.ndarray:
        """
        Run complete Transformer inference (N layers) through virtual GPU.
        """
        h = x
        num_layers = model_config.get('num_layers', 1)
        for i in range(num_layers):
            layer_cfg = model_config['layers'][i]
            h = self.run_transformer_block(h, layer_cfg)
        # Final projection to vocabulary
        if 'lm_head' in model_config:
            h = self.linear(h, model_config['lm_head'])
            h = self.softmax(h, axis=-1)
        return h

    # ─────────────────────────────────────────
    #  Device Info
    # ─────────────────────────────────────────

    def _record_op(self, elapsed_s: float):
        self._op_count += 1
        self._total_compute_ms += elapsed_s * 1000

    def device_info(self) -> dict:
        return {
            "device":          self.DEVICE_NAME,
            "version":         self.VERSION,
            "compute_units":   self.scheduler.num_compute_units,
            "warp_size":       WARP_SIZE,
            **self.memory.get_stats(),
            **self.scheduler.get_stats(),
            "ops_executed":    self._op_count,
            "total_compute_ms": round(self._total_compute_ms, 2),
        }

    def __repr__(self):
        info = self.device_info()
        return (f"SynthGPU({info['compute_units']} CUs, "
                f"{info['vram_total_mb']}MB vRAM, "
                f"warp_size={info['warp_size']})")
