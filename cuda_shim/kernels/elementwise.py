"""Element-wise activation kernels."""

import numpy as np


def relu(x: np.ndarray, out: np.ndarray = None) -> np.ndarray:
    return np.maximum(x, 0, out=out)


def gelu(x: np.ndarray, out: np.ndarray = None) -> np.ndarray:
    tmp = np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3))
    result = 0.5 * x * (1.0 + tmp)
    if out is not None:
        np.copyto(out, result)
        return out
    return result


def sigmoid(x: np.ndarray, out: np.ndarray = None) -> np.ndarray:
    result = 1.0 / (1.0 + np.exp(-x))
    if out is not None:
        np.copyto(out, result)
        return out
    return result


def tanh(x: np.ndarray, out: np.ndarray = None) -> np.ndarray:
    return np.tanh(x, out=out)


def swish(x: np.ndarray, out: np.ndarray = None) -> np.ndarray:
    result = x * sigmoid(x)
    if out is not None:
        np.copyto(out, result)
        return out
    return result


def clipped_relu(x: np.ndarray, clip: float = 6.0, out: np.ndarray = None) -> np.ndarray:
    result = np.clip(x, 0, clip)
    if out is not None:
        np.copyto(out, result)
        return out
    return result
