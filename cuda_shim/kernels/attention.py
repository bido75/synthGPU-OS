"""Scaled dot-product attention kernel."""

import numpy as np


def scaled_dot_product_attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    scale: float = None,
    mask: np.ndarray = None,
) -> np.ndarray:
    if scale is None:
        scale = 1.0 / np.sqrt(Q.shape[-1])

    scores = np.matmul(Q, K.swapaxes(-2, -1)) * scale

    if mask is not None:
        scores += mask

    scores -= scores.max(axis=-1, keepdims=True)
    attn_weights = np.exp(scores)
    attn_weights /= attn_weights.sum(axis=-1, keepdims=True)

    return np.matmul(attn_weights, V)


def multi_head_attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    num_heads: int,
    scale: float = None,
    mask: np.ndarray = None,
) -> np.ndarray:
    batch, seq_len, d_model = Q.shape
    head_dim = d_model // num_heads

    Q_h = Q.reshape(batch, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
    K_h = K.reshape(batch, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
    V_h = V.reshape(batch, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)

    out = scaled_dot_product_attention(Q_h, K_h, V_h, scale, mask)
    return out.transpose(0, 2, 1, 3).reshape(batch, seq_len, d_model)
