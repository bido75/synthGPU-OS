"""
SynthGPU - GPU Operations Library v0.2
All GPU kernels implemented via NumPy BLAS/SIMD.
"""

import numpy as np
from typing import Optional


def gemm(A: np.ndarray, B: np.ndarray,
         alpha: float = 1.0, beta: float = 0.0,
         C: Optional[np.ndarray] = None) -> np.ndarray:
    result = alpha * np.matmul(A, B)
    if C is not None and beta != 0.0:
        result += beta * C
    return result


def batched_gemm(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return np.matmul(A, B)


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)


def gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -88, 88)))


def tanh_act(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x_max = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x - x_max)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


def silu(x: np.ndarray) -> np.ndarray:
    return x * sigmoid(x)


def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray,
               eps: float = 1e-5) -> np.ndarray:
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    x_norm = (x - mean) / np.sqrt(var + eps)
    return gamma * x_norm + beta


def rms_norm(x: np.ndarray, gamma: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)
    return gamma * (x / rms)


def scaled_dot_product_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                                  mask: Optional[np.ndarray] = None) -> np.ndarray:
    d_k = Q.shape[-1]
    scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(d_k)
    if mask is not None:
        scores = np.where(mask, scores, -1e9)
    attn_weights = softmax(scores, axis=-1)
    return np.matmul(attn_weights, V)


def conv2d(x: np.ndarray, weight: np.ndarray,
           bias: Optional[np.ndarray] = None,
           stride: int = 1, padding: int = 0) -> np.ndarray:
    N, C_in, H, W = x.shape
    C_out, _, kH, kW = weight.shape
    if padding > 0:
        x = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    H_out = (x.shape[2] - kH) // stride + 1
    W_out = (x.shape[3] - kW) // stride + 1
    col = np.zeros((N, C_in, kH, kW, H_out, W_out), dtype=x.dtype)
    for i in range(kH):
        for j in range(kW):
            col[:, :, i, j, :, :] = x[:, :,
                                     i:i + H_out * stride:stride,
                                     j:j + W_out * stride:stride]
    col = col.reshape(N, C_in * kH * kW, H_out * W_out)
    w = weight.reshape(C_out, -1)
    out = np.tensordot(w, col, axes=([1], [1])).transpose(1, 0, 2).reshape(N, C_out, H_out, W_out)
    if bias is not None:
        out += bias[np.newaxis, :, np.newaxis, np.newaxis]
    return out


def add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a + b


def multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a * b


def dropout(x: np.ndarray, p: float = 0.5, training: bool = True) -> np.ndarray:
    if not training or p == 0:
        return x
    mask = (np.random.random(x.shape) > p).astype(x.dtype) / (1.0 - p)
    return x * mask


def embedding_lookup(indices: np.ndarray, weight_table: np.ndarray) -> np.ndarray:
    return weight_table[indices]
