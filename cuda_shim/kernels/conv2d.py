"""2-D convolution kernel (naive but correct)."""

import numpy as np

try:
    from scipy.signal import correlate2d as _correlate2d
except ImportError:
    _correlate2d = None


def correlate2d(a, b, mode="full", boundary="fill", fillvalue=0):
    if _correlate2d is not None:
        return _correlate2d(a, b, mode=mode, boundary=boundary, fillvalue=fillvalue)
    from numpy.fft import fft2, ifft2
    pad_r = b.shape[0] - 1
    pad_c = b.shape[1] - 1
    a_pad = np.pad(a, ((pad_r, pad_r), (pad_c, pad_c)), constant_values=fillvalue)
    result = np.real(ifft2(fft2(a_pad) * np.conj(fft2(b, s=a_pad.shape))))
    if mode == "valid":
        return result[pad_r * 2: a.shape[0], pad_c * 2: a.shape[1]]
    if mode == "same":
        return result[pad_r: pad_r + a.shape[0], pad_c: pad_c + a.shape[1]]
    return result


def conv2d_forward(
    x: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray = None,
    stride: int = 1,
    padding: int = 0,
) -> np.ndarray:
    N, C_in, H, W = x.shape
    C_out, _, kH, kW = weight.shape

    if padding > 0:
        x = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))

    H_out = (H + 2 * padding - kH) // stride + 1
    W_out = (W + 2 * padding - kW) // stride + 1

    out = np.zeros((N, C_out, H_out, W_out), dtype=x.dtype)

    for n in range(N):
        for c_out in range(C_out):
            for c_in in range(C_in):
                for i in range(H_out):
                    for j in range(W_out):
                        i0, j0 = i * stride, j * stride
                        out[n, c_out, i, j] += np.sum(
                            x[n, c_in, i0:i0 + kH, j0:j0 + kW] * weight[c_out, c_in]
                        )
            if bias is not None:
                out[n, c_out] += bias[c_out]

    return out
