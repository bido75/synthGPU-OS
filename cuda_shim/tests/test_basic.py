"""
SynthGPU CUDA Shim — Core API Tests (test_basic.py)
====================================================
Tests the Python bridge layer directly.
No PyTorch required — pure numpy + bridge_api.

Run with:
    pytest cuda_shim/tests/test_basic.py -v
    python cuda_shim/tests/test_basic.py        # standalone
"""

import sys
import os
import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


# ── Helpers (must be defined before check() calls) ────────────────
def assert_(cond):
    if not cond:
        raise AssertionError("Condition is False")


def _check_warp_increment():
    before = _scheduler.get_stats()["warps_executed"]
    _scheduler.record_external_warps(64, 0.5)
    after  = _scheduler.get_stats()["warps_executed"]
    assert after == before + 64, f"Expected +64, got {after - before}"


def _check_identity_gemm():
    A = np.eye(4, dtype=np.float32)
    B = np.eye(4, dtype=np.float32)
    C = np.zeros((4, 4), dtype=np.float32)
    cuda_gemm(A, B, C, 1.0, 0.0, False, False)
    assert np.allclose(C, np.eye(4), atol=1e-5), f"GEMM identity mismatch:\n{C}"


def _check_gemm_alpha():
    A = np.ones((2, 2), dtype=np.float32)
    B = np.ones((2, 2), dtype=np.float32)
    C = np.zeros((2, 2), dtype=np.float32)
    cuda_gemm(A, B, C, 2.0, 0.0, False, False)
    assert np.allclose(C, np.full((2, 2), 4.0), atol=1e-5), f"Alpha scaling failed: {C}"


def _check_gemm_warp():
    before = _scheduler.get_stats()["warps_executed"]
    A = np.eye(64, dtype=np.float32)
    B = np.eye(64, dtype=np.float32)
    C = np.zeros((64, 64), dtype=np.float32)
    cuda_gemm(A, B, C, 1.0, 0.0, False, False)
    after = _scheduler.get_stats()["warps_executed"]
    assert after > before, "Warp counter did not increment after GEMM"


def _check_relu():
    x = np.array([-2., -1., 0., 1., 2.], dtype=np.float32)
    y = np.zeros_like(x)
    cuda_relu(x, y)
    expected = np.array([0., 0., 0., 1., 2.], dtype=np.float32)
    assert np.allclose(y, expected), f"ReLU failed: {y}"


def _check_softmax():
    x = np.array([[1., 2., 3., 4.]], dtype=np.float32)
    y = np.zeros_like(x)
    cuda_softmax(x, y, axis=-1)
    assert abs(y.sum() - 1.0) < 1e-5, f"Softmax sum = {y.sum()}"
    assert (y > 0).all(), "Softmax produced non-positive values"


def _check_gelu_shape():
    from cuda_shim.kernels.bridge_api import cuda_gelu
    x = np.random.randn(8, 16).astype(np.float32)
    y = np.zeros_like(x)
    cuda_gelu(x, y)
    assert y.shape == x.shape


def _check_layernorm_shape():
    x     = np.random.randn(4, 8).astype(np.float32)
    gamma = np.ones(8, dtype=np.float32)
    beta  = np.zeros(8, dtype=np.float32)
    y     = np.zeros_like(x)
    cuda_layer_norm(x, gamma, beta, y, eps=1e-5)
    assert y.shape == x.shape


def _check_embedding():
    weight  = np.arange(20, dtype=np.float32).reshape(10, 2)
    indices = np.array([0, 3, 7], dtype=np.int64)
    out     = np.zeros((3, 2), dtype=np.float32)
    cuda_embedding(indices, weight, out)
    assert np.allclose(out, weight[[0, 3, 7]]), f"Embedding mismatch: {out}"


def _check_attention():
    b, h, s, d = 1, 2, 4, 8
    Q = np.random.randn(b, h, s, d).astype(np.float32)
    K = np.random.randn(b, h, s, d).astype(np.float32)
    V = np.random.randn(b, h, s, d).astype(np.float32)
    O = np.zeros_like(Q)
    cuda_attention(Q, K, V, O, scale=d ** -0.5)
    assert O.shape == Q.shape


def _check_adam():
    param      = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    grad       = np.ones(3, dtype=np.float32) * 0.1
    exp_avg    = np.zeros(3, dtype=np.float32)
    exp_avg_sq = np.zeros(3, dtype=np.float32)
    original   = param.copy()
    cuda_adam_step(param, grad, exp_avg, exp_avg_sq,
                   lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, step=1)
    assert not np.allclose(param, original), "Adam step had no effect"


# ── Test runner ───────────────────────────────────────────────────
PASS = "PASS"
FAIL = "FAIL"
results = []

def check(name, fn):
    try:
        fn()
        print(f"  [{PASS}] {name}")
        results.append((name, True))
    except Exception as e:
        print(f"  [{FAIL}] {name}: {e}")
        results.append((name, False))


# ── Imports needed by helper functions ────────────────────────────
from cuda_shim.kernels.bridge_api import (
    _scheduler, get_scheduler, get_telemetry,
    cuda_gemm, cuda_relu, cuda_softmax, cuda_layer_norm,
    cuda_embedding, cuda_attention, cuda_adam_step,
)

check("bridge_api imports cleanly",
      lambda: __import__("cuda_shim.kernels.bridge_api", fromlist=["bridge_api"]))

check("All kernel modules importable", lambda: (
    __import__("cuda_shim.kernels.gemm",        fromlist=["gemm"]),
    __import__("cuda_shim.kernels.attention",    fromlist=["attention"]),
    __import__("cuda_shim.kernels.elementwise",  fromlist=["elementwise"]),
    __import__("cuda_shim.kernels.reduction",    fromlist=["reduction"]),
    __import__("cuda_shim.kernels.norm",         fromlist=["norm"]),
    __import__("cuda_shim.kernels.embedding",    fromlist=["embedding"]),
    __import__("cuda_shim.kernels.conv2d",       fromlist=["conv2d"]),
    __import__("cuda_shim.kernels.optimizer",    fromlist=["optimizer"]),
))

# ── Group 2: Scheduler ────────────────────────────────────────────
print("\n[2] WarpScheduler integration")

from cuda_shim.kernels.bridge_api import _scheduler, get_scheduler, get_telemetry

check("_scheduler singleton exists",
      lambda: (assert_(_scheduler is not None)))

check("get_scheduler() returns same instance",
      lambda: assert_(get_scheduler() is _scheduler))

check("get_telemetry() returns shim_active=True",
      lambda: assert_(get_telemetry()["shim_active"] is True))

check("record_external_warps increments counter", lambda: (
    _check_warp_increment()
))

def _check_warp_increment():
    before = _scheduler.get_stats()["warps_executed"]
    _scheduler.record_external_warps(64, 0.5)
    after  = _scheduler.get_stats()["warps_executed"]
    assert after == before + 64, f"Expected +64, got {after - before}"

# ── Group 3: GEMM ─────────────────────────────────────────────────
print("\n[3] GEMM — matrix multiply")

from cuda_shim.kernels.bridge_api import cuda_gemm

check("Identity GEMM: I*I=I", lambda: (
    _check_identity_gemm()
))

def _check_identity_gemm():
    A = np.eye(4, dtype=np.float32)
    B = np.eye(4, dtype=np.float32)
    C = np.zeros((4, 4), dtype=np.float32)
    cuda_gemm(A, B, C, 1.0, 0.0, False, False)
    assert np.allclose(C, np.eye(4), atol=1e-5), f"GEMM identity mismatch:\n{C}"

check("GEMM alpha scaling", lambda: (
    _check_gemm_alpha()
))

def _check_gemm_alpha():
    A = np.ones((2, 2), dtype=np.float32)
    B = np.ones((2, 2), dtype=np.float32)
    C = np.zeros((2, 2), dtype=np.float32)
    cuda_gemm(A, B, C, 2.0, 0.0, False, False)
    # alpha=2: result = 2*(1*1+1*1) = 4 per element
    assert np.allclose(C, np.full((2, 2), 4.0), atol=1e-5), f"Alpha scaling failed: {C}"

check("GEMM warp counter increments after call", lambda: (
    _check_gemm_warp()
))

def _check_gemm_warp():
    before = _scheduler.get_stats()["warps_executed"]
    A = np.eye(64, dtype=np.float32)
    B = np.eye(64, dtype=np.float32)
    C = np.zeros((64, 64), dtype=np.float32)
    cuda_gemm(A, B, C, 1.0, 0.0, False, False)
    after = _scheduler.get_stats()["warps_executed"]
    assert after > before, "Warp counter did not increment after GEMM"

# ── Group 4: Activations ──────────────────────────────────────────
print("\n[4] Activations")

from cuda_shim.kernels.bridge_api import cuda_relu, cuda_gelu, cuda_softmax

check("ReLU zeros negatives", lambda: (
    _check_relu()
))

def _check_relu():
    x = np.array([-2., -1., 0., 1., 2.], dtype=np.float32)
    y = np.zeros_like(x)
    cuda_relu(x, y)
    expected = np.array([0., 0., 0., 1., 2.], dtype=np.float32)
    assert np.allclose(y, expected), f"ReLU failed: {y}"

check("Softmax sums to 1.0", lambda: (
    _check_softmax()
))

def _check_softmax():
    x = np.array([[1., 2., 3., 4.]], dtype=np.float32)
    y = np.zeros_like(x)
    cuda_softmax(x, y, axis=-1)
    assert abs(y.sum() - 1.0) < 1e-5, f"Softmax sum = {y.sum()}"
    assert (y > 0).all(), "Softmax produced non-positive values"

check("GELU preserves shape", lambda: (
    _check_gelu_shape()
))

def _check_gelu_shape():
    from cuda_shim.kernels.bridge_api import cuda_gelu
    x = np.random.randn(8, 16).astype(np.float32)
    y = np.zeros_like(x)
    cuda_gelu(x, y)
    assert y.shape == x.shape

# ── Group 5: Norm / Embedding / Attention ─────────────────────────
print("\n[5] Norm, Embedding, Attention")

from cuda_shim.kernels.bridge_api import cuda_layer_norm, cuda_embedding, cuda_attention

check("LayerNorm output shape matches input", lambda: (
    _check_layernorm_shape()
))

def _check_layernorm_shape():
    x     = np.random.randn(4, 8).astype(np.float32)
    gamma = np.ones(8, dtype=np.float32)
    beta  = np.zeros(8, dtype=np.float32)
    y     = np.zeros_like(x)
    cuda_layer_norm(x, gamma, beta, y, eps=1e-5)
    assert y.shape == x.shape

check("Embedding lookup correct values", lambda: (
    _check_embedding()
))

def _check_embedding():
    weight  = np.arange(20, dtype=np.float32).reshape(10, 2)
    indices = np.array([0, 3, 7], dtype=np.int64)
    out     = np.zeros((3, 2), dtype=np.float32)
    cuda_embedding(indices, weight, out)
    assert np.allclose(out, weight[[0, 3, 7]]), f"Embedding mismatch: {out}"

check("Attention output shape correct", lambda: (
    _check_attention()
))

def _check_attention():
    b, h, s, d = 1, 2, 4, 8
    Q = np.random.randn(b, h, s, d).astype(np.float32)
    K = np.random.randn(b, h, s, d).astype(np.float32)
    V = np.random.randn(b, h, s, d).astype(np.float32)
    O = np.zeros_like(Q)
    cuda_attention(Q, K, V, O, scale=d ** -0.5)
    assert O.shape == Q.shape

# ── Group 6: Adam optimizer ───────────────────────────────────────
print("\n[6] Optimizer")

from cuda_shim.kernels.bridge_api import cuda_adam_step

check("Adam step modifies parameters", lambda: (
    _check_adam()
))

def _check_adam():
    param      = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    grad       = np.ones(3, dtype=np.float32) * 0.1
    exp_avg    = np.zeros(3, dtype=np.float32)
    exp_avg_sq = np.zeros(3, dtype=np.float32)
    original   = param.copy()
    cuda_adam_step(param, grad, exp_avg, exp_avg_sq,
                   lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, step=1)
    assert not np.allclose(param, original), "Adam step had no effect"

# ── Summary ───────────────────────────────────────────────────────
def assert_(cond):
    if not cond:
        raise AssertionError("Condition is False")

passed = sum(1 for _, ok in results if ok)
failed = len(results) - passed

print(f"\n{'='*50}")
print(f"  Results: {passed}/{len(results)} passed")
if failed:
    print("  Failed tests:")
    for name, ok in results:
        if not ok:
            print(f"    FAIL  {name}")
print(f"{'='*50}\n")

if __name__ == "__main__":
    sys.exit(0 if failed == 0 else 1)
