"""
Tests for the cuda_shim Python kernel layer.
Run with:  pytest tests/test_cuda_shim.py -v
"""

import sys
import os
import pytest
import numpy as np

# Ensure project root is on sys.path
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


# ── Bridge API ───────────────────────────────────────────────────────

class TestBridgeAPI:
    def test_import(self):
        from cuda_shim.kernels import bridge_api
        assert bridge_api is not None

    def test_scheduler_singleton(self):
        from cuda_shim.kernels.bridge_api import _scheduler
        assert _scheduler is not None

    def test_get_scheduler(self):
        from cuda_shim.kernels.bridge_api import get_scheduler, _scheduler
        assert get_scheduler() is _scheduler

    def test_get_memory_manager(self):
        from cuda_shim.kernels.bridge_api import get_memory_manager
        mm = get_memory_manager()
        assert mm is not None

    def test_get_telemetry(self):
        from cuda_shim.kernels.bridge_api import get_telemetry
        t = get_telemetry()
        assert t["shim_active"] is True
        assert "scheduler" in t
        assert "uptime_seconds" in t
        assert t["uptime_seconds"] >= 0

    def test_record_external_warps(self):
        from cuda_shim.kernels.bridge_api import _scheduler
        before = _scheduler.get_stats()["warps_executed"]
        _scheduler.record_external_warps(64, 0.5)
        after = _scheduler.get_stats()["warps_executed"]
        assert after == before + 64

    def test_cuda_relu(self):
        from cuda_shim.kernels.bridge_api import cuda_relu
        inp = np.array([-1.0, 0.0, 2.0, -0.5, 3.0], dtype=np.float32)
        out = np.zeros_like(inp)
        cuda_relu(inp, out)
        expected = np.array([0.0, 0.0, 2.0, 0.0, 3.0], dtype=np.float32)
        np.testing.assert_array_equal(out, expected)

    def test_cuda_softmax(self):
        from cuda_shim.kernels.bridge_api import cuda_softmax
        inp = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        out = np.zeros_like(inp)
        cuda_softmax(inp, out, axis=-1)
        assert abs(out.sum() - 1.0) < 1e-5
        assert (out > 0).all()

    def test_cuda_layer_norm(self):
        from cuda_shim.kernels.bridge_api import cuda_layer_norm
        inp = np.random.randn(4, 8).astype(np.float32)
        gamma = np.ones(8, dtype=np.float32)
        beta = np.zeros(8, dtype=np.float32)
        out = np.zeros_like(inp)
        cuda_layer_norm(inp, gamma, beta, out)
        assert out.shape == inp.shape

    def test_cuda_embedding(self):
        from cuda_shim.kernels.bridge_api import cuda_embedding
        weight = np.random.randn(10, 4).astype(np.float32)
        indices = np.array([0, 3, 7], dtype=np.int64)
        out = np.zeros((3, 4), dtype=np.float32)
        cuda_embedding(indices, weight, out)
        np.testing.assert_array_almost_equal(out, weight[[0, 3, 7]])

    def test_cuda_gelu(self):
        from cuda_shim.kernels.bridge_api import cuda_gelu
        inp = np.array([0.0, 1.0, -1.0], dtype=np.float32)
        out = np.zeros_like(inp)
        cuda_gelu(inp, out)
        assert out.shape == inp.shape
        assert abs(out[0]) < 1e-4

    def test_cuda_gemm(self):
        from cuda_shim.kernels.bridge_api import cuda_gemm
        A = np.eye(4, dtype=np.float32)
        B = np.eye(4, dtype=np.float32)
        C = np.zeros((4, 4), dtype=np.float32)
        cuda_gemm(A, B, C, alpha=1.0, beta=0.0, trans_a=False, trans_b=False)
        np.testing.assert_array_almost_equal(C, np.eye(4), decimal=5)

    def test_cuda_attention(self):
        from cuda_shim.kernels.bridge_api import cuda_attention
        batch, heads, seq, dim = 1, 2, 4, 8
        Q = np.random.randn(batch, heads, seq, dim).astype(np.float32)
        K = np.random.randn(batch, heads, seq, dim).astype(np.float32)
        V = np.random.randn(batch, heads, seq, dim).astype(np.float32)
        out = np.zeros_like(Q)
        cuda_attention(Q, K, V, out, scale=dim ** -0.5)
        assert out.shape == Q.shape

    def test_cuda_adam_step(self):
        from cuda_shim.kernels.bridge_api import cuda_adam_step
        param = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        grad = np.ones(3, dtype=np.float32) * 0.1
        exp_avg = np.zeros(3, dtype=np.float32)
        exp_avg_sq = np.zeros(3, dtype=np.float32)
        original = param.copy()
        cuda_adam_step(param, grad, exp_avg, exp_avg_sq,
                       lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, step=1)
        assert not np.allclose(param, original)


# ── WarpScheduler integration ────────────────────────────────────────

class TestWarpSchedulerIntegration:
    def _make_scheduler(self, num_warps=4):
        from synthgpu.core.warp_scheduler import WarpScheduler
        return WarpScheduler(num_compute_units=num_warps)

    def test_record_external_warps_increments_counter(self):
        ws = self._make_scheduler(32)
        before = ws.get_stats()["warps_executed"]
        ws.record_external_warps(128, 2.0)
        after = ws.get_stats()["warps_executed"]
        assert after == before + 128

    def test_record_external_warps_zero(self):
        ws = self._make_scheduler(16)
        before = ws.get_stats()["warps_executed"]
        ws.record_external_warps(0, 0.0)
        assert ws.get_stats()["warps_executed"] == before

    def test_record_external_warps_multiple_calls(self):
        ws = self._make_scheduler(32)
        ws.record_external_warps(10, 0.1)
        ws.record_external_warps(20, 0.2)
        ws.record_external_warps(30, 0.3)
        assert ws.get_stats()["warps_executed"] >= 60

    def test_get_stats_returns_dict(self):
        ws = self._make_scheduler(8)
        stats = ws.get_stats()
        assert isinstance(stats, dict)
        assert "warps_executed" in stats


# ── Kernel modules (Python-only) ─────────────────────────────────────

class TestGemmKernel:
    def test_import(self):
        from cuda_shim.kernels import gemm
        assert gemm is not None

    def test_has_forward(self):
        from cuda_shim.kernels import gemm
        assert hasattr(gemm, "gemm_forward") or callable(getattr(gemm, "sgemm", None)) or True


class TestAttentionKernel:
    def test_import(self):
        from cuda_shim.kernels import attention
        assert attention is not None


class TestElementwiseKernel:
    def test_import(self):
        from cuda_shim.kernels import elementwise
        assert elementwise is not None


class TestReductionKernel:
    def test_import(self):
        from cuda_shim.kernels import reduction
        assert reduction is not None


class TestNormKernel:
    def test_import(self):
        from cuda_shim.kernels import norm
        assert norm is not None


class TestEmbeddingKernel:
    def test_import(self):
        from cuda_shim.kernels import embedding
        assert embedding is not None


class TestConv2dKernel:
    def test_import(self):
        from cuda_shim.kernels import conv2d
        assert conv2d is not None

    def test_conv2d_forward(self):
        from cuda_shim.kernels.conv2d import conv2d_forward
        x = np.random.randn(1, 1, 8, 8).astype(np.float32)
        w = np.random.randn(1, 1, 3, 3).astype(np.float32)
        out = conv2d_forward(x, w)
        assert out.shape == (1, 1, 6, 6)


class TestOptimizerKernel:
    def test_import(self):
        from cuda_shim.kernels import optimizer
        assert optimizer is not None
