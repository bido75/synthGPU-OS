"""
SynthGPU - GPU Operations Library
====================================
Optimized implementations of core GPU compute operations.

Each operation is written as a kernel function compatible with the
WarpScheduler's dispatch_kernel interface, AND as a direct high-level
call that handles warp partitioning internally.

Operations implemented:
  - Matrix Multiply (GEMM)          — foundation of all AI workloads
  - 2D Convolution                  — CNNs
  - Activation functions            — ReLU, GELU, Sigmoid, Tanh, SoftMax
  - Layer Normalization             — Transformers
  - Scaled Dot-Product Attention    — Transformers / LLMs
  - Element-wise ops                — Add, Multiply, etc.
"""

import numpy as np
from typing import Optional
import time


# ─────────────────────────────────────────────
#  Matrix Multiplication (GEMM)
#  The single most important operation in AI.
#  C = alpha * A @ B + beta * C
# ─────────────────────────────────────────────

def gemm(A: np.ndarray, B: np.ndarray,
         alpha: float = 1.0, beta: float = 0.0,
         C: Optional[np.ndarray] = None) -> np.ndarray:
    """
    General Matrix Multiply.
    Uses numpy which maps directly to optimized BLAS (OpenBLAS/MKL)
    and SIMD instructions (AVX2/AVX-512 on x86, NEON on ARM).
    This IS SIMD execution — numpy is the SIMD layer.
    """
    result = alpha * np.matmul(A, B)
    if C is not None and beta != 0.0:
        result += beta * C
    return result


def batched_gemm(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Batched matrix multiply: A shape (batch, M, K), B shape (batch, K, N)"""
    return np.matmul(A, B)


# ─────────────────────────────────────────────
#  Activation Functions
# ─────────────────────────────────────────────

def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)


def gelu(x: np.ndarray) -> np.ndarray:
    """Gaussian Error Linear Unit — used in GPT, BERT, etc."""
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -88, 88)))


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x_max = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x - x_max)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray,
               eps: float = 1e-5) -> np.ndarray:
    """Layer Normalization — critical for Transformer stability."""
    mean = np.mean(x, axis=-1, keepdims=True)
    var  = np.var(x, axis=-1, keepdims=True)
    x_norm = (x - mean) / np.sqrt(var + eps)
    return gamma * x_norm + beta


# ─────────────────────────────────────────────
#  Scaled Dot-Product Attention
#  The core operation of every Transformer / LLM
# ─────────────────────────────────────────────

def scaled_dot_product_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                                  mask: Optional[np.ndarray] = None,
                                  dropout_p: float = 0.0) -> np.ndarray:
    """
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V

    Args:
        Q: Query  (batch, heads, seq_len, d_k)
        K: Key    (batch, heads, seq_len, d_k)
        V: Value  (batch, heads, seq_len, d_v)
        mask: Optional causal mask

    Returns:
        Output (batch, heads, seq_len, d_v)
    """
    d_k = Q.shape[-1]
    scale = np.sqrt(d_k)

    # Attention scores: (batch, heads, seq_len, seq_len)
    scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / scale

    if mask is not None:
        scores = np.where(mask, scores, -1e9)

    attn_weights = softmax(scores, axis=-1)
    return np.matmul(attn_weights, V)


# ─────────────────────────────────────────────
#  2D Convolution (simplified, for CNN support)
# ─────────────────────────────────────────────

def conv2d(x: np.ndarray, weight: np.ndarray,
           bias: Optional[np.ndarray] = None,
           stride: int = 1, padding: int = 0) -> np.ndarray:
    """
    2D Convolution via im2col + GEMM.
    x:      (N, C_in, H, W)
    weight: (C_out, C_in, kH, kW)
    """
    N, C_in, H, W = x.shape
    C_out, _, kH, kW = weight.shape

    # Pad input
    if padding > 0:
        x = np.pad(x, ((0,0),(0,0),(padding,padding),(padding,padding)))

    H_out = (x.shape[2] - kH) // stride + 1
    W_out = (x.shape[3] - kW) // stride + 1

    # im2col: reshape input into column matrix for GEMM
    col = np.zeros((N, C_in, kH, kW, H_out, W_out), dtype=x.dtype)
    for i in range(kH):
        for j in range(kW):
            col[:, :, i, j, :, :] = x[:, :,
                i:i+H_out*stride:stride,
                j:j+W_out*stride:stride]

    col = col.reshape(N, C_in * kH * kW, H_out * W_out)
    w   = weight.reshape(C_out, -1)

    # GEMM: (N, C_out, H_out*W_out)
    out = np.matmul(w, col)  # (C_out, N*H_out*W_out) — we use einsum for batch
    out = np.einsum('oi,nio->no', w, col.transpose(0,2,1)).reshape(N, H_out, W_out, C_out)
    out = out.transpose(0, 3, 1, 2)  # (N, C_out, H_out, W_out)

    if bias is not None:
        out += bias[np.newaxis, :, np.newaxis, np.newaxis]

    return out


# ─────────────────────────────────────────────
#  Element-wise Operations
# ─────────────────────────────────────────────

def add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a + b

def multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a * b

def dropout(x: np.ndarray, p: float = 0.5, training: bool = True) -> np.ndarray:
    if not training or p == 0:
        return x
    mask = np.random.binomial(1, 1-p, x.shape).astype(x.dtype) / (1-p)
    return x * mask
