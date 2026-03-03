"""Embedding lookup kernel."""

import numpy as np


def embedding_forward(
    indices: np.ndarray,
    weight: np.ndarray,
    padding_idx: int = None,
) -> np.ndarray:
    out = weight[indices]
    if padding_idx is not None:
        mask = indices == padding_idx
        out[mask] = 0.0
    return out


def embedding_backward(
    grad_output: np.ndarray,
    indices: np.ndarray,
    num_embeddings: int,
    embedding_dim: int,
) -> np.ndarray:
    grad_weight = np.zeros((num_embeddings, embedding_dim), dtype=grad_output.dtype)
    np.add.at(grad_weight, indices.flatten(), grad_output.reshape(-1, embedding_dim))
    return grad_weight
