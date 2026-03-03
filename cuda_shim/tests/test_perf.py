"""
SynthGPU CUDA Shim — Performance Benchmark (test_perf.py)
==========================================================
Measures:
  - GEMM throughput (GFLOPS) at various matrix sizes
  - Tokens/sec for a minimal transformer layer
  - Memory bandwidth (effective MB/s)

Run with:
    python cuda_shim/tests/test_perf.py
    python cuda_shim/tests/test_perf.py --quick   # fast subset only
"""

import sys
import os
import time
import argparse
import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from cuda_shim.kernels.bridge_api import cuda_gemm, cuda_attention, cuda_relu, _scheduler


def bench_gemm(sizes=(64, 128, 256, 512, 1024), repeats=5, quick=False):
    """Measure GEMM throughput (GFLOPS) via bridge_api."""
    print("\n=== GEMM Benchmark ===")
    print(f"  {'N':>6}  {'ms/iter':>10}  {'GFLOPS':>10}  {'warps':>8}")
    print(f"  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*8}")

    if quick:
        sizes = (64, 256)
        repeats = 3

    for n in sizes:
        A = np.random.randn(n, n).astype(np.float32)
        B = np.random.randn(n, n).astype(np.float32)
        C = np.zeros((n, n), dtype=np.float32)

        # Warmup
        cuda_gemm(A, B, C, 1.0, 0.0, False, False)

        w_before = _scheduler.get_stats()["warps_executed"]
        t0 = time.perf_counter()
        for _ in range(repeats):
            cuda_gemm(A, B, C, 1.0, 0.0, False, False)
        elapsed = (time.perf_counter() - t0) / repeats * 1000   # ms

        flops     = 2 * n ** 3           # 2*N^3 FLOPs for N×N GEMM
        gflops    = (flops / elapsed) / 1e6  # GFLOPS (ms denominator)
        w_after   = _scheduler.get_stats()["warps_executed"]
        warps_per = (w_after - w_before) // repeats

        print(f"  {n:>6}  {elapsed:>10.2f}  {gflops:>10.2f}  {warps_per:>8}")


def bench_attention(seq_lens=(16, 64, 128), heads=4, d_model=64, repeats=5, quick=False):
    """Measure scaled dot-product attention throughput."""
    print("\n=== Attention Benchmark ===")
    print(f"  {'seq':>6}  {'ms/iter':>10}  {'warps':>8}")
    print(f"  {'-'*6}  {'-'*10}  {'-'*8}")

    if quick:
        seq_lens = (16, 64)
        repeats = 3

    head_dim = d_model // heads

    for seq in seq_lens:
        Q = np.random.randn(1, heads, seq, head_dim).astype(np.float32)
        K = np.random.randn(1, heads, seq, head_dim).astype(np.float32)
        V = np.random.randn(1, heads, seq, head_dim).astype(np.float32)
        O = np.zeros_like(Q)

        cuda_attention(Q, K, V, O, scale=head_dim ** -0.5)  # warmup

        w_before = _scheduler.get_stats()["warps_executed"]
        t0 = time.perf_counter()
        for _ in range(repeats):
            cuda_attention(Q, K, V, O, scale=head_dim ** -0.5)
        elapsed = (time.perf_counter() - t0) / repeats * 1000
        w_after = _scheduler.get_stats()["warps_executed"]

        print(f"  {seq:>6}  {elapsed:>10.2f}  {(w_after-w_before)//repeats:>8}")


def bench_tokens_per_sec(seq_len=32, d_model=128, vocab=1000, repeats=3, quick=False):
    """Estimate tokens/sec for a minimal transformer block."""
    print("\n=== Tokens/sec Benchmark ===")
    from cuda_shim.kernels.bridge_api import cuda_layer_norm, cuda_embedding

    if quick:
        repeats = 1

    weight = np.random.randn(vocab, d_model).astype(np.float32)
    tokens = np.random.randint(0, vocab, (seq_len,), dtype=np.int64)
    W      = np.random.randn(d_model, d_model).astype(np.float32) * 0.02
    gamma  = np.ones(d_model, dtype=np.float32)
    beta   = np.zeros(d_model, dtype=np.float32)

    def one_forward():
        hidden = np.zeros((seq_len, d_model), dtype=np.float32)
        cuda_embedding(tokens, weight, hidden)
        normed = np.zeros_like(hidden)
        cuda_layer_norm(hidden, gamma, beta, normed)
        proj = np.zeros_like(normed)
        cuda_gemm(normed, W, proj, 1.0, 0.0, False, False)
        out = np.zeros_like(proj)
        cuda_relu(proj, out)
        return out

    one_forward()  # warmup
    t0 = time.perf_counter()
    for _ in range(repeats):
        one_forward()
    elapsed = (time.perf_counter() - t0) / repeats

    tps = seq_len / elapsed
    print(f"  seq_len={seq_len}  d_model={d_model}  → {tps:.1f} tokens/sec")
    return tps


def bench_memory_bandwidth(size_mb=64, repeats=5, quick=False):
    """Measure effective memory bandwidth (cudaMemcpy simulation)."""
    print("\n=== Memory Bandwidth ===")
    if quick:
        size_mb = 4
        repeats = 3

    n = size_mb * 1024 * 1024 // 4  # float32 elements
    src = np.random.randn(n).astype(np.float32)
    dst = np.empty_like(src)

    np.copyto(dst, src)  # warmup
    t0 = time.perf_counter()
    for _ in range(repeats):
        np.copyto(dst, src)
    elapsed = (time.perf_counter() - t0) / repeats

    bw_gbs = (size_mb / 1024) / elapsed
    print(f"  {size_mb} MB copy → {elapsed*1000:.2f} ms  ({bw_gbs:.2f} GB/s)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Fast subset only")
    args = parser.parse_args()

    print("=" * 50)
    print("  SynthGPU CUDA Shim — Performance Benchmark")
    print("=" * 50)

    bench_gemm(quick=args.quick)
    bench_attention(quick=args.quick)
    bench_tokens_per_sec(quick=args.quick)
    bench_memory_bandwidth(quick=args.quick)

    stats = _scheduler.get_stats()
    print(f"\n=== Scheduler Stats ===")
    print(f"  Total warps executed: {stats['warps_executed']}")
    print(f"  Compute units:        {stats['compute_units']}")
    print(f"  Uptime (s):           {stats['uptime_seconds']:.1f}")
    print("\nBenchmark complete.\n")
