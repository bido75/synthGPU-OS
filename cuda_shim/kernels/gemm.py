"""GEMM kernel — matrix multiply backed by numpy/OpenBLAS."""

import numpy as np


def sgemm(
    A: np.ndarray,
    B: np.ndarray,
    alpha: float = 1.0,
    beta: float = 0.0,
    C: np.ndarray = None,
    trans_a: bool = False,
    trans_b: bool = False,
) -> np.ndarray:
    Aop = A.T if trans_a else A
    Bop = B.T if trans_b else B
    result = np.dot(Aop, Bop) * alpha
    if C is not None and beta != 0.0:
        result += beta * C
    return result


def dgemm(
    A: np.ndarray,
    B: np.ndarray,
    alpha: float = 1.0,
    beta: float = 0.0,
    C: np.ndarray = None,
    trans_a: bool = False,
    trans_b: bool = False,
) -> np.ndarray:
    return sgemm(
        A.astype(np.float64),
        B.astype(np.float64),
        alpha, beta, C, trans_a, trans_b,
    )


def batched_gemm(
    A_batch: np.ndarray,
    B_batch: np.ndarray,
    alpha: float = 1.0,
) -> np.ndarray:
    return np.einsum("bij,bjk->bik", A_batch, B_batch) * alpha
