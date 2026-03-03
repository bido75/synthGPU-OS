"""Optimizer step kernels — Adam and SGD."""

import numpy as np


def adam_step(
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
    amsgrad: bool = False,
    max_exp_avg_sq: np.ndarray = None,
) -> None:
    if weight_decay != 0.0:
        grad = grad + weight_decay * param

    exp_avg[:] = beta1 * exp_avg + (1 - beta1) * grad
    exp_avg_sq[:] = beta2 * exp_avg_sq + (1 - beta2) * grad ** 2

    bias_c1 = 1 - beta1 ** step
    bias_c2 = 1 - beta2 ** step

    if amsgrad and max_exp_avg_sq is not None:
        np.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
        denom = np.sqrt(max_exp_avg_sq / bias_c2) + eps
    else:
        denom = np.sqrt(exp_avg_sq / bias_c2) + eps

    step_size = lr / bias_c1
    param -= step_size * exp_avg / denom


def sgd_step(
    param: np.ndarray,
    grad: np.ndarray,
    lr: float,
    momentum: float = 0.0,
    dampening: float = 0.0,
    weight_decay: float = 0.0,
    buf: np.ndarray = None,
) -> None:
    if weight_decay != 0.0:
        grad = grad + weight_decay * param

    if momentum != 0.0 and buf is not None:
        buf[:] = momentum * buf + (1 - dampening) * grad
        grad = buf

    param -= lr * grad
