"""
SynthGPU CUDA Shim — Python Kernel Package
===========================================
Exposes all compute kernels and the bridge API.
Imports are best-effort so that a missing optional dependency
(e.g. scipy) does not prevent the core shim from loading.
"""

__all__ = [
    "gemm",
    "attention",
    "elementwise",
    "reduction",
    "norm",
    "embedding",
    "conv2d",
    "optimizer",
]

def __getattr__(name):
    if name in __all__:
        import importlib
        mod = importlib.import_module(f"cuda_shim.kernels.{name}")
        globals()[name] = mod
        return mod
    raise AttributeError(f"module 'cuda_shim.kernels' has no attribute {name!r}")
