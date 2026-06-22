"""
SynthGPU CUDA Bridge API
========================
Python-side of the C-to-Python bridge.
The C shim (device_info.c / cublas.c / cudnn.c) imports this module and
calls these functions for every GPU compute operation it intercepts.

Contract:
  - Input arrays wrap C memory via ctypes (zero-copy where possible)
  - Results written in-place to output array
  - Every call updates the SynthGPU WarpScheduler counters so the
    dashboard telemetry stays accurate
"""

import os
import sys
import time

import numpy as np

_project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Prefer flat backend/synthgpu/ layout, fall back to direct import
try:
    from synthgpu.warp_scheduler import WarpScheduler
    from synthgpu.memory_manager import VirtualMemoryManager
except ModuleNotFoundError:
    try:
        import importlib
        _sg = importlib.import_module("synthgpu.warp_scheduler")
        WarpScheduler = _sg.WarpScheduler
        _mm = importlib.import_module("synthgpu.memory_manager")
        VirtualMemoryManager = _mm.VirtualMemoryManager
    except Exception:
        raise RuntimeError("Cannot load SynthGPU packages — ensure backend/ is on PYTHONPATH")

_scheduler = WarpScheduler()
_memory_manager = VirtualMemoryManager()
_init_time = time.time()


def get_scheduler() -> WarpScheduler:
    return _scheduler


def record_warp_dispatch(warp_count: int, exec_ms: float) -> None:
    """Record external shim work using the shared scheduler instance."""
    get_scheduler().record_external_warps(warp_count, exec_ms)


def get_memory_manager() -> VirtualMemoryManager:
    return _memory_manager


def cuda_gemm(
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    alpha: float,
    beta: float,
    trans_a: bool,
    trans_b: bool,
) -> None:
    """
    Called by: cublasSgemm, cublasGemmEx, cublasDgemm
    C = alpha * op(A) @ op(B) + beta * C  (written in-place to C)
    """
    t0 = time.perf_counter()

    Aop = A.T if trans_a else A
    Bop = B.T if trans_b else B

    if Aop.shape[0] >= 32:
        def _gemm_kernel(data_slice, shared):
            return np.dot(data_slice, shared["B"])

        result = _scheduler.dispatch_kernel(
            kernel=_gemm_kernel,
            data=Aop,
            shared_mem={"B": Bop},
        )
    else:
        result = np.dot(Aop, Bop)
        _scheduler.record_external_warps(1, (time.perf_counter() - t0) * 1000)

    np.multiply(result, alpha, out=result)
    if beta != 0.0:
        np.multiply(C, beta, out=C)
        np.add(result, C, out=C)
    else:
        np.copyto(C, result)


def cuda_softmax(input: np.ndarray, output: np.ndarray, axis: int = -1) -> None:
    """Called by: cudnnSoftmaxForward"""
    t0 = time.perf_counter()
    x = input - input.max(axis=axis, keepdims=True)
    np.exp(x, out=output)
    output /= output.sum(axis=axis, keepdims=True)
    _scheduler.record_external_warps(1, (time.perf_counter() - t0) * 1000)


def cuda_relu(input: np.ndarray, output: np.ndarray) -> None:
    """Called by: cudnnActivationForward (RELU mode)"""
    np.maximum(input, 0, out=output)
    _scheduler.record_external_warps(1, 0.1)


def cuda_gelu(input: np.ndarray, output: np.ndarray) -> None:
    """Called by: cudnnActivationForward (GELU mode)"""
    tmp = np.tanh(np.sqrt(2.0 / np.pi) * (input + 0.044715 * input ** 3))
    np.multiply(0.5 * input, 1.0 + tmp, out=output)
    _scheduler.record_external_warps(1, 0.5)


def cuda_layer_norm(
    input: np.ndarray,
    gamma: np.ndarray,
    beta: np.ndarray,
    output: np.ndarray,
    eps: float = 1e-5,
) -> None:
    """Called by: cudnnLayerNorm, apex LayerNorm"""
    mean = input.mean(axis=-1, keepdims=True)
    var = input.var(axis=-1, keepdims=True)
    normalized = (input - mean) / np.sqrt(var + eps)
    np.multiply(normalized, gamma, out=output)
    output += beta
    _scheduler.record_external_warps(2, 1.0)


def cuda_embedding(
    indices: np.ndarray, weight: np.ndarray, output: np.ndarray
) -> None:
    """Called by: custom embedding CUDA kernels"""
    np.take(
        weight,
        indices.flatten(),
        axis=0,
        out=output.reshape(-1, weight.shape[1]),
    )
    _scheduler.record_external_warps(1, 0.2)


def cuda_attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    output: np.ndarray,
    scale: float,
    mask: np.ndarray = None,
) -> None:
    """Called by: custom attention CUDA kernels, cudnnMultiHeadAttn"""
    t0 = time.perf_counter()
    scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) * scale
    if mask is not None:
        scores += mask
    scores -= scores.max(axis=-1, keepdims=True)
    np.exp(scores, out=scores)
    scores /= scores.sum(axis=-1, keepdims=True)
    np.matmul(scores, V, out=output)
    _scheduler.record_external_warps(4, (time.perf_counter() - t0) * 1000)


def cuda_adam_step(
    param: np.ndarray,
    grad: np.ndarray,
    exp_avg: np.ndarray,
    exp_avg_sq: np.ndarray,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    step: int,
    weight_decay: float = 0.0,
) -> None:
    """Called by: custom Adam CUDA kernel"""
    if weight_decay != 0:
        grad = grad + weight_decay * param
    exp_avg[:] = beta1 * exp_avg + (1 - beta1) * grad
    exp_avg_sq[:] = beta2 * exp_avg_sq + (1 - beta2) * grad ** 2
    bias_c1 = 1 - beta1 ** step
    bias_c2 = 1 - beta2 ** step
    step_size = lr / bias_c1
    denom = np.sqrt(exp_avg_sq / bias_c2) + eps
    param -= step_size * exp_avg / denom
    _scheduler.record_external_warps(3, 1.0)


def get_telemetry() -> dict:
    """Called by dashboard WebSocket broadcast — keeps shim stats in sync."""
    stats = _scheduler.get_stats()
    return {
        "shim_active": True,
        "scheduler": stats,
        "uptime_seconds": time.time() - _init_time,
    }
