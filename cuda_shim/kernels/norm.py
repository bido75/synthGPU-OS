"""Normalisation kernels — LayerNorm and BatchNorm."""

import numpy as np


def layer_norm(
    x: np.ndarray,
    gamma: np.ndarray,
    beta: np.ndarray,
    eps: float = 1e-5,
    out: np.ndarray = None,
) -> np.ndarray:
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    normalized = (x - mean) / np.sqrt(var + eps)
    result = gamma * normalized + beta
    if out is not None:
        np.copyto(out, result)
        return out
    return result


def batch_norm(
    x: np.ndarray,
    gamma: np.ndarray,
    beta: np.ndarray,
    running_mean: np.ndarray,
    running_var: np.ndarray,
    training: bool = False,
    momentum: float = 0.1,
    eps: float = 1e-5,
) -> np.ndarray:
    if training:
        mean = x.mean(axis=(0, 2, 3), keepdims=True)
        var = x.var(axis=(0, 2, 3), keepdims=True)
        running_mean[:] = (1 - momentum) * running_mean + momentum * mean.squeeze()
        running_var[:] = (1 - momentum) * running_var + momentum * var.squeeze()
    else:
        mean = running_mean.reshape(1, -1, 1, 1)
        var = running_var.reshape(1, -1, 1, 1)

    normalized = (x - mean) / np.sqrt(var + eps)
    g = gamma.reshape(1, -1, 1, 1)
    b = beta.reshape(1, -1, 1, 1)
    return g * normalized + b
