"""Reduction kernels — softmax, sum, max, mean."""

import numpy as np


def softmax(x: np.ndarray, axis: int = -1, out: np.ndarray = None) -> np.ndarray:
    shifted = x - x.max(axis=axis, keepdims=True)
    exp_x = np.exp(shifted)
    result = exp_x / exp_x.sum(axis=axis, keepdims=True)
    if out is not None:
        np.copyto(out, result)
        return out
    return result


def log_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = x - x.max(axis=axis, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=axis, keepdims=True))


def reduce_sum(x: np.ndarray, axis=None, keepdims: bool = False) -> np.ndarray:
    return np.sum(x, axis=axis, keepdims=keepdims)


def reduce_mean(x: np.ndarray, axis=None, keepdims: bool = False) -> np.ndarray:
    return np.mean(x, axis=axis, keepdims=keepdims)


def reduce_max(x: np.ndarray, axis=None, keepdims: bool = False) -> np.ndarray:
    return np.max(x, axis=axis, keepdims=keepdims)
