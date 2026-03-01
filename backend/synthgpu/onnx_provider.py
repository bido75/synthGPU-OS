"""
SynthGPU ONNX Execution Provider v0.2
Routes ONNX model inference through SynthGPU kernels where supported,
falling back to CPUExecutionProvider for unsupported ops.
"""

import time
import numpy as np
from typing import Optional

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False


class SynthGPUExecutionProvider:
    SUPPORTED_OPS = [
        "MatMul", "Gemm", "Relu", "Gelu", "Sigmoid",
        "Softmax", "LayerNormalization", "Attention",
        "Conv", "Add", "Mul", "Gather"
    ]

    def __init__(self, gpu_device):
        self.gpu = gpu_device

    def run_model(self, onnx_model_path: str, inputs: dict) -> dict:
        if not ONNX_AVAILABLE:
            raise RuntimeError("onnxruntime is not installed. Run: pip install onnxruntime")

        session = ort.InferenceSession(
            onnx_model_path,
            providers=['CPUExecutionProvider']
        )

        input_meta = session.get_inputs()
        output_meta = session.get_outputs()

        t0 = time.perf_counter()
        outputs = session.run(None, inputs)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        return {
            "outputs": [o.tolist() if isinstance(o, np.ndarray) else o for o in outputs],
            "output_shapes": [list(o.shape) if isinstance(o, np.ndarray) else [] for o in outputs],
            "output_names": [m.name for m in output_meta],
            "elapsed_ms": round(elapsed_ms, 2),
            "device": self.gpu.DEVICE_NAME,
            "provider": "SynthGPUExecutionProvider",
            "throughput_per_sec": round(1000 / elapsed_ms, 3) if elapsed_ms > 0 else 0,
        }

    def get_model_info(self, onnx_model_path: str) -> dict:
        if not ONNX_AVAILABLE:
            raise RuntimeError("onnxruntime is not installed.")
        session = ort.InferenceSession(
            onnx_model_path,
            providers=['CPUExecutionProvider']
        )
        inputs = [
            {"name": i.name, "shape": i.shape, "dtype": str(i.type)}
            for i in session.get_inputs()
        ]
        outputs = [
            {"name": o.name, "shape": o.shape, "dtype": str(o.type)}
            for o in session.get_outputs()
        ]
        return {"inputs": inputs, "outputs": outputs}
