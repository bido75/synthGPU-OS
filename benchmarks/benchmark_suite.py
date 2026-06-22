"""
SynthGPU Benchmark Suite
==========================
Compares SynthGPU virtual GPU performance against naive CPU baseline
across real AI workload patterns.

Benchmarks:
  1. GEMM (Matrix Multiply) — various sizes
  2. MLP Inference           — feed-forward network
  3. Transformer Block       — attention + FFN
  4. LLM Token Generation    — autoregressive inference
  5. Batch Throughput        — requests/second
"""

import numpy as np
import time
import sys
import os
_BENCH_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _BENCH_ROOT)
sys.path.insert(0, os.path.join(_BENCH_ROOT, "backend"))

from synthgpu import SynthGPU
from synthgpu.ops import gpu_ops


def timer(fn, *args, runs=3, **kwargs):
    """Run fn multiple times, return best time in ms and result."""
    best_ms = float('inf')
    result = None
    for _ in range(runs):
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        ms = (time.perf_counter() - t0) * 1000
        if ms < best_ms:
            best_ms = ms
    return best_ms, result


def print_header(title: str):
    print()
    print("─" * 60)
    print(f"  {title}")
    print("─" * 60)


def print_result(name, cpu_ms, gpu_ms, unit="ms"):
    speedup = cpu_ms / gpu_ms if gpu_ms > 0 else 0
    bar_len = min(int(speedup * 5), 40)
    bar = "█" * bar_len
    print(f"  {name:<35}")
    print(f"    CPU baseline:  {cpu_ms:>8.2f} ms")
    print(f"    SynthGPU:      {gpu_ms:>8.2f} ms")
    print(f"    Speedup:       {speedup:>7.2f}x  {bar}")
    print()


def benchmark_gemm(gpu: SynthGPU):
    print_header("BENCHMARK 1: Matrix Multiply (GEMM)")
    print("  The fundamental operation of all AI — testing at real model sizes.\n")

    sizes = [
        ("Small  (256×256)",   256,  256,  256),
        ("Medium (1024×1024)", 1024, 1024, 1024),
        ("Large  (2048×2048)", 2048, 2048, 2048),
        ("LLM    (4096×4096)", 4096, 4096, 4096),
    ]

    for name, M, K, N in sizes:
        A = np.random.randn(M, K).astype(np.float32)
        B = np.random.randn(K, N).astype(np.float32)

        cpu_ms, _ = timer(np.matmul, A, B)
        gpu_ms, _ = timer(gpu.matmul, A, B)

        flops = 2 * M * K * N
        gpu_tflops = (flops / 1e12) / (gpu_ms / 1000)
        print_result(name, cpu_ms, gpu_ms)
        print(f"    Throughput: {gpu_tflops:.3f} TFLOPS\n")


def benchmark_mlp(gpu: SynthGPU):
    print_header("BENCHMARK 2: MLP Inference (3-layer network)")
    print("  Feed-forward neural network — classic image classification style.\n")

    # ResNet-style dimensions
    configs = [
        ("Batch=1,   dim=768",  1,   768),
        ("Batch=32,  dim=768",  32,  768),
        ("Batch=128, dim=1024", 128, 1024),
    ]

    for name, batch, dim in configs:
        layers = [
            {'weight': np.random.randn(dim*4, dim).astype(np.float32),
             'bias':   np.random.randn(dim*4).astype(np.float32),
             'activation': 'gelu'},
            {'weight': np.random.randn(dim*2, dim*4).astype(np.float32),
             'bias':   np.random.randn(dim*2).astype(np.float32),
             'activation': 'relu'},
            {'weight': np.random.randn(1000, dim*2).astype(np.float32),
             'activation': 'softmax'},
        ]

        x = np.random.randn(batch, dim).astype(np.float32)

        # CPU baseline: manual forward pass
        def cpu_mlp(x, layers):
            h = x
            for l in layers:
                h = h @ l['weight'].T
                if 'bias' in l: h += l['bias']
                act = l.get('activation', 'relu')
                if act == 'relu': h = np.maximum(0, h)
                elif act == 'gelu':
                    h = 0.5*h*(1+np.tanh(np.sqrt(2/np.pi)*(h+0.044715*h**3)))
                elif act == 'softmax':
                    h = np.exp(h - h.max(axis=-1, keepdims=True))
                    h = h / h.sum(axis=-1, keepdims=True)
            return h

        cpu_ms, cpu_out = timer(cpu_mlp, x, layers)
        gpu_ms, gpu_out = timer(gpu.run_mlp, x, layers)

        # Verify correctness
        max_diff = np.max(np.abs(cpu_out - gpu_out))

        print_result(name, cpu_ms, gpu_ms)
        print(f"    Numerical accuracy (max diff): {max_diff:.2e}  "
              f"{'✓ CORRECT' if max_diff < 1e-4 else '✗ MISMATCH'}\n")


def benchmark_transformer(gpu: SynthGPU):
    print_header("BENCHMARK 3: Transformer Block (Attention + FFN)")
    print("  Core of every LLM — GPT, BERT, LLaMA-style architecture.\n")

    def make_transformer_config(d_model, num_heads, d_ff, seq_len, batch):
        np.random.seed(42)
        scale = 0.02
        return {
            'num_heads': num_heads,
            'Wq': np.random.randn(d_model, d_model).astype(np.float32) * scale,
            'Wk': np.random.randn(d_model, d_model).astype(np.float32) * scale,
            'Wv': np.random.randn(d_model, d_model).astype(np.float32) * scale,
            'Wo': np.random.randn(d_model, d_model).astype(np.float32) * scale,
            'W1': np.random.randn(d_ff, d_model).astype(np.float32) * scale,
            'b1': np.zeros(d_ff, dtype=np.float32),
            'W2': np.random.randn(d_model, d_ff).astype(np.float32) * scale,
            'b2': np.zeros(d_model, dtype=np.float32),
            'gamma1': np.ones(d_model, dtype=np.float32),
            'beta1':  np.zeros(d_model, dtype=np.float32),
            'gamma2': np.ones(d_model, dtype=np.float32),
            'beta2':  np.zeros(d_model, dtype=np.float32),
        }

    configs = [
        ("GPT-2 Small  (d=768,  h=12, seq=128)", 768,  12, 3072, 128, 1),
        ("GPT-2 Medium (d=1024, h=16, seq=128)", 1024, 16, 4096, 128, 1),
        ("LLM-style    (d=2048, h=16, seq=256)", 2048, 16, 8192, 256, 1),
    ]

    for name, d_model, num_heads, d_ff, seq_len, batch in configs:
        cfg = make_transformer_config(d_model, num_heads, d_ff, seq_len, batch)
        x   = np.random.randn(batch, seq_len, d_model).astype(np.float32) * 0.1

        gpu_ms, _ = timer(gpu.run_transformer_block, x, cfg, runs=2)

        # Tokens per second for this config
        tokens_per_sec = (batch * seq_len) / (gpu_ms / 1000)
        print(f"  {name}")
        print(f"    SynthGPU time:      {gpu_ms:.2f} ms")
        print(f"    Throughput:         {tokens_per_sec:,.0f} tokens/sec")
        print()


def benchmark_token_generation(gpu: SynthGPU):
    print_header("BENCHMARK 4: LLM Token Generation (Autoregressive)")
    print("  Simulating next-token prediction — what ChatGPT does per token.\n")

    d_model = 512
    num_heads = 8
    vocab_size = 32000
    num_layers = 4

    np.random.seed(0)
    scale = 0.02

    def make_layer():
        return {
            'num_heads': num_heads,
            'Wq': np.random.randn(d_model, d_model).astype(np.float32) * scale,
            'Wk': np.random.randn(d_model, d_model).astype(np.float32) * scale,
            'Wv': np.random.randn(d_model, d_model).astype(np.float32) * scale,
            'Wo': np.random.randn(d_model, d_model).astype(np.float32) * scale,
            'W1': np.random.randn(d_model*4, d_model).astype(np.float32) * scale,
            'b1': np.zeros(d_model*4, dtype=np.float32),
            'W2': np.random.randn(d_model, d_model*4).astype(np.float32) * scale,
            'b2': np.zeros(d_model, dtype=np.float32),
            'gamma1': np.ones(d_model, dtype=np.float32),
            'beta1':  np.zeros(d_model, dtype=np.float32),
            'gamma2': np.ones(d_model, dtype=np.float32),
            'beta2':  np.zeros(d_model, dtype=np.float32),
        }

    model_config = {
        'num_layers': num_layers,
        'layers': [make_layer() for _ in range(num_layers)],
        'lm_head': np.random.randn(vocab_size, d_model).astype(np.float32) * scale,
    }

    print(f"  Model: {num_layers}-layer Transformer, d={d_model}, vocab={vocab_size:,}")
    print(f"  Generating 20 tokens...\n")

    tokens_generated = []
    total_ms = 0

    for step in range(20):
        # Each step: context grows by 1 token
        seq_len = step + 1
        x = np.random.randn(1, seq_len, d_model).astype(np.float32) * 0.1

        t0 = time.perf_counter()
        logits = gpu.run_inference(x, model_config)
        step_ms = (time.perf_counter() - t0) * 1000
        total_ms += step_ms

        # Greedy decode (argmax of last token logits)
        next_token = int(np.argmax(logits[0, -1, :]))
        tokens_generated.append(next_token)

        if step % 5 == 0:
            print(f"    Token {step+1:2d}: id={next_token:5d}  ({step_ms:.1f}ms)")

    avg_ms = total_ms / 20
    tokens_per_sec = 1000 / avg_ms
    print(f"\n  Average: {avg_ms:.1f}ms/token = {tokens_per_sec:.1f} tokens/sec")
    print(f"  Total 20 tokens: {total_ms:.0f}ms")
    print(f"  Generated token IDs: {tokens_generated[:5]}... (showing first 5)")


def run_all_benchmarks():
    print("\n" + "═" * 60)
    print("  SynthGPU MVP Benchmark Suite")
    print("  Proving the concept: CPU → Virtual GPU Acceleration")
    print("═" * 60)

    gpu = SynthGPU(vram_mb=1024)

    benchmark_gemm(gpu)
    benchmark_mlp(gpu)
    benchmark_transformer(gpu)
    benchmark_token_generation(gpu)

    print_header("DEVICE STATISTICS")
    stats = gpu.device_info()
    for k, v in stats.items():
        print(f"  {k:<30}: {v}")

    print()
    print("═" * 60)
    print("  SynthGPU MVP Benchmark Complete")
    print("  All AI workloads executed on virtual GPU — no physical GPU used.")
    print("═" * 60)
    print()

    gpu.scheduler.shutdown()


if __name__ == "__main__":
    run_all_benchmarks()
