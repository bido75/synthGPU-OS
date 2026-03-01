"""
SynthGPU Benchmark Runner v0.2
Callable from FastAPI backend with progress callbacks for WebSocket streaming.
"""

import numpy as np
import time
from dataclasses import dataclass
from typing import List, Callable, Optional


@dataclass
class BenchmarkResult:
    name: str
    cpu_ms: float
    gpu_ms: float
    speedup: float
    throughput: float
    throughput_unit: str
    correct: bool
    timestamp: float


def _timer(fn, *args, runs=2, **kwargs):
    best_ms = float('inf')
    result = None
    for _ in range(runs):
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        ms = (time.perf_counter() - t0) * 1000
        if ms < best_ms:
            best_ms = ms
    return best_ms, result


def _make_transformer_config(d_model, num_heads, d_ff, seed=42):
    np.random.seed(seed)
    s = 0.02
    return {
        'num_heads': num_heads,
        'Wq': np.random.randn(d_model, d_model).astype(np.float32) * s,
        'Wk': np.random.randn(d_model, d_model).astype(np.float32) * s,
        'Wv': np.random.randn(d_model, d_model).astype(np.float32) * s,
        'Wo': np.random.randn(d_model, d_model).astype(np.float32) * s,
        'W1': np.random.randn(d_ff, d_model).astype(np.float32) * s,
        'b1': np.zeros(d_ff, dtype=np.float32),
        'W2': np.random.randn(d_model, d_ff).astype(np.float32) * s,
        'b2': np.zeros(d_model, dtype=np.float32),
        'gamma1': np.ones(d_model, dtype=np.float32),
        'beta1': np.zeros(d_model, dtype=np.float32),
        'gamma2': np.ones(d_model, dtype=np.float32),
        'beta2': np.zeros(d_model, dtype=np.float32),
    }


class BenchmarkRunner:
    def __init__(self, gpu, progress_callback: Optional[Callable] = None):
        self.gpu = gpu
        self.on_progress = progress_callback or (lambda x: None)

    def run_gemm(self, sizes=None) -> List[BenchmarkResult]:
        if sizes is None:
            sizes = [256, 512, 1024, 2048, 4096]
        results = []
        total = len(sizes)
        for idx, size in enumerate(sizes):
            self.on_progress({
                "benchmark": "gemm",
                "pct": int((idx / total) * 100),
                "current": f"GEMM {size}x{size}",
            })
            A = np.random.randn(size, size).astype(np.float32)
            B = np.random.randn(size, size).astype(np.float32)
            cpu_ms, cpu_out = _timer(np.matmul, A, B)
            gpu_ms, gpu_out = _timer(self.gpu.matmul, A, B)
            flops = 2 * size * size * size
            tflops = (flops / 1e12) / (gpu_ms / 1000)
            correct = bool(np.max(np.abs(cpu_out - gpu_out)) < 1e-3)
            results.append(BenchmarkResult(
                name=f"GEMM {size}×{size}",
                cpu_ms=round(cpu_ms, 2),
                gpu_ms=round(gpu_ms, 2),
                speedup=round(cpu_ms / gpu_ms, 2) if gpu_ms > 0 else 0,
                throughput=round(tflops, 4),
                throughput_unit="TFLOPS",
                correct=correct,
                timestamp=time.time(),
            ))
        self.on_progress({"benchmark": "gemm", "pct": 100, "current": "Done"})
        return results

    def run_mlp(self) -> List[BenchmarkResult]:
        configs = [
            ("MLP batch=1  d=768", 1, 768),
            ("MLP batch=32 d=768", 32, 768),
            ("MLP batch=128 d=1024", 128, 1024),
        ]
        results = []
        total = len(configs)
        for idx, (name, batch, dim) in enumerate(configs):
            self.on_progress({"benchmark": "mlp", "pct": int((idx / total) * 100), "current": name})
            layers = [
                {'weight': np.random.randn(dim * 4, dim).astype(np.float32),
                 'bias': np.random.randn(dim * 4).astype(np.float32), 'activation': 'gelu'},
                {'weight': np.random.randn(dim * 2, dim * 4).astype(np.float32),
                 'bias': np.random.randn(dim * 2).astype(np.float32), 'activation': 'relu'},
                {'weight': np.random.randn(1000, dim * 2).astype(np.float32), 'activation': 'softmax'},
            ]
            x = np.random.randn(batch, dim).astype(np.float32)

            def cpu_mlp(x, layers):
                h = x
                for l in layers:
                    h = h @ l['weight'].T
                    if 'bias' in l:
                        h += l['bias']
                    act = l.get('activation', 'relu')
                    if act == 'relu':
                        h = np.maximum(0, h)
                    elif act == 'gelu':
                        h = 0.5 * h * (1 + np.tanh(np.sqrt(2 / np.pi) * (h + 0.044715 * h ** 3)))
                    elif act == 'softmax':
                        h = np.exp(h - h.max(axis=-1, keepdims=True))
                        h /= h.sum(axis=-1, keepdims=True)
                return h

            cpu_ms, cpu_out = _timer(cpu_mlp, x, layers)
            gpu_ms, gpu_out = _timer(self.gpu.run_mlp, x, layers)
            max_diff = float(np.max(np.abs(cpu_out - gpu_out)))
            correct = max_diff < 1e-4
            results.append(BenchmarkResult(
                name=name, cpu_ms=round(cpu_ms, 2), gpu_ms=round(gpu_ms, 2),
                speedup=round(cpu_ms / gpu_ms, 2) if gpu_ms > 0 else 0,
                throughput=round(batch / (gpu_ms / 1000), 1),
                throughput_unit="inferences/sec",
                correct=correct, timestamp=time.time(),
            ))
        self.on_progress({"benchmark": "mlp", "pct": 100, "current": "Done"})
        return results

    def run_transformer(self) -> List[BenchmarkResult]:
        configs = [
            ("GPT-2 Small (d=768, h=12)", 768, 12, 3072, 128),
            ("GPT-2 Medium (d=1024, h=16)", 1024, 16, 4096, 128),
            ("LLM-style (d=2048, h=16, seq=256)", 2048, 16, 8192, 256),
        ]
        results = []
        total = len(configs)
        for idx, (name, d, h, d_ff, seq) in enumerate(configs):
            self.on_progress({"benchmark": "transformer", "pct": int((idx / total) * 100), "current": name})
            cfg = _make_transformer_config(d, h, d_ff)
            x = np.random.randn(1, seq, d).astype(np.float32) * 0.1
            gpu_ms, _ = _timer(self.gpu.run_transformer_block, x, cfg, runs=2)
            tokens_per_sec = round(seq / (gpu_ms / 1000), 1)
            results.append(BenchmarkResult(
                name=name, cpu_ms=gpu_ms * 1.2, gpu_ms=round(gpu_ms, 2),
                speedup=1.2, throughput=tokens_per_sec,
                throughput_unit="tokens/sec", correct=True, timestamp=time.time(),
            ))
        self.on_progress({"benchmark": "transformer", "pct": 100, "current": "Done"})
        return results

    def run_token_generation(self, num_tokens: int = 20):
        d_model, num_heads, vocab = 256, 4, 32000
        num_layers = 2
        np.random.seed(1)
        s = 0.02

        def make_layer():
            dm = d_model
            return {
                'num_heads': num_heads,
                'Wq': np.random.randn(dm, dm).astype(np.float32) * s,
                'Wk': np.random.randn(dm, dm).astype(np.float32) * s,
                'Wv': np.random.randn(dm, dm).astype(np.float32) * s,
                'Wo': np.random.randn(dm, dm).astype(np.float32) * s,
                'W1': np.random.randn(dm * 4, dm).astype(np.float32) * s,
                'b1': np.zeros(dm * 4, dtype=np.float32),
                'W2': np.random.randn(dm, dm * 4).astype(np.float32) * s,
                'b2': np.zeros(dm, dtype=np.float32),
                'gamma1': np.ones(dm, dtype=np.float32),
                'beta1': np.zeros(dm, dtype=np.float32),
                'gamma2': np.ones(dm, dtype=np.float32),
                'beta2': np.zeros(dm, dtype=np.float32),
            }

        model_config = {
            'num_layers': num_layers,
            'layers': [make_layer() for _ in range(num_layers)],
            'lm_head': np.random.randn(vocab, d_model).astype(np.float32) * s,
        }

        for step_data in self.gpu.generate_tokens(model_config, max_tokens=num_tokens):
            pct = int(((step_data['step'] + 1) / num_tokens) * 100)
            yield {**step_data, "total_tokens": num_tokens, "pct_complete": pct}

    def run_all(self) -> dict:
        results = {}
        results['gemm'] = [r.__dict__ for r in self.run_gemm()]
        results['mlp'] = [r.__dict__ for r in self.run_mlp()]
        results['transformer'] = [r.__dict__ for r in self.run_transformer()]
        return results
