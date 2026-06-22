"""ONNX Runtime inference with real provider and per-node telemetry."""

import json
import os
import tempfile
import time
from collections import OrderedDict

import numpy as np

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ort = None
    ONNX_AVAILABLE = False


def _parse_ort_profile(profile_file: str) -> list[dict]:
    """Parse and aggregate ONNX Runtime kernel events from Chrome-trace JSON."""
    with open(profile_file, "r", encoding="utf-8") as handle:
        events = json.load(handle)

    nodes = OrderedDict()
    for event in events:
        args = event.get("args") or {}
        op_name = args.get("op_name")
        provider = args.get("provider")
        duration_us = event.get("dur")
        if not op_name or not provider or duration_us is None:
            continue

        event_name = str(event.get("name", ""))
        node_name = args.get("node_name") or event_name.removesuffix("_kernel_time")
        key = (node_name, op_name, provider)
        if key not in nodes:
            nodes[key] = {
                "node_name": node_name,
                "op_name": op_name,
                "provider": provider,
                "duration_ms": 0.0,
            }
        nodes[key]["duration_ms"] += max(0.0, float(duration_us) / 1000.0)

    result = list(nodes.values())
    for node in result:
        node["duration_ms"] = round(node["duration_ms"], 4)
    return result


def _outputs_match(actual: list, expected: list) -> bool:
    if len(actual) != len(expected):
        return False
    for actual_value, expected_value in zip(actual, expected):
        actual_array = np.asarray(actual_value)
        expected_array = np.asarray(expected_value)
        if actual_array.shape != expected_array.shape:
            return False
        if np.issubdtype(actual_array.dtype, np.number):
            if not np.allclose(actual_array, expected_array, rtol=1e-4, atol=1e-5,
                               equal_nan=True):
                return False
        elif not np.array_equal(actual_array, expected_array):
            return False
    return True


class SynthGPUExecutionProvider:
    SUPPORTED_OPS = {
        "MatMul", "Gemm", "Relu", "Gelu", "Sigmoid",
        "Softmax", "LayerNormalization", "Attention",
        "Conv", "Add", "Mul", "Gather",
    }

    def __init__(self, gpu_device):
        self.gpu = gpu_device

    @staticmethod
    def get_available_providers() -> list[str]:
        return ort.get_available_providers() if ONNX_AVAILABLE else []

    @staticmethod
    def _session_options():
        options = ort.SessionOptions()
        options.enable_profiling = True
        options.profile_file_prefix = os.path.join(
            tempfile.gettempdir(), "synthgpu_ort_profile"
        )
        return options

    def run_model(self, onnx_model_path: str, inputs: dict,
                  provider: str = "cpu") -> dict:
        if not ONNX_AVAILABLE:
            raise RuntimeError("onnxruntime is not installed")
        if provider not in {"cpu", "openvino"}:
            raise ValueError(f"Unknown ONNX provider: {provider}")

        use_openvino = provider == "openvino"
        available = self.get_available_providers()
        if use_openvino and "OpenVINOExecutionProvider" not in available:
            raise RuntimeError(
                "OpenVINOExecutionProvider is not available in this runtime"
            )

        providers = (
            ["OpenVINOExecutionProvider", "CPUExecutionProvider"]
            if use_openvino else ["CPUExecutionProvider"]
        )
        session = ort.InferenceSession(
            onnx_model_path,
            sess_options=self._session_options(),
            providers=providers,
        )
        output_meta = session.get_outputs()

        t0 = time.perf_counter()
        outputs = session.run(None, inputs)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        profile_file = session.end_profiling()
        try:
            per_node_timing = _parse_ort_profile(profile_file)
        finally:
            if profile_file:
                try:
                    os.remove(profile_file)
                except FileNotFoundError:
                    pass

        correctness_verified = True
        if use_openvino:
            cpu_session = ort.InferenceSession(
                onnx_model_path, providers=["CPUExecutionProvider"]
            )
            cpu_outputs = cpu_session.run(None, inputs)
            correctness_verified = _outputs_match(outputs, cpu_outputs)
            if not correctness_verified:
                raise RuntimeError(
                    "OpenVINO output failed the CPUExecutionProvider correctness gate"
                )

        providers_used = list(dict.fromkeys(
            node["provider"] for node in per_node_timing if node.get("provider")
        ))
        active_provider = (
            " + ".join(providers_used)
            if providers_used else session.get_providers()[0]
        )
        op_names = {node["op_name"] for node in per_node_timing}

        return {
            "outputs": [
                output.tolist() if isinstance(output, np.ndarray) else output
                for output in outputs
            ],
            "output_shapes": [
                list(output.shape) if isinstance(output, np.ndarray) else []
                for output in outputs
            ],
            "output_names": [meta.name for meta in output_meta],
            "elapsed_ms": round(elapsed_ms, 2),
            "device": self.gpu.DEVICE_NAME,
            "provider": active_provider,
            "providers_used": providers_used,
            "configured_providers": session.get_providers(),
            "correctness_verified": correctness_verified,
            "per_node_timing_ms": per_node_timing,
            "profiled_node_total_ms": round(
                sum(node["duration_ms"] for node in per_node_timing), 4
            ),
            "unsupported_ops": sorted(op_names - self.SUPPORTED_OPS),
            "throughput_per_sec": (
                round(1000 / elapsed_ms, 3) if elapsed_ms > 0 else 0
            ),
        }

    def get_model_info(self, onnx_model_path: str) -> dict:
        if not ONNX_AVAILABLE:
            raise RuntimeError("onnxruntime is not installed")
        session = ort.InferenceSession(
            onnx_model_path, providers=["CPUExecutionProvider"]
        )
        inputs = [
            {"name": item.name, "shape": item.shape, "dtype": str(item.type)}
            for item in session.get_inputs()
        ]
        outputs = [
            {"name": item.name, "shape": item.shape, "dtype": str(item.type)}
            for item in session.get_outputs()
        ]
        return {"inputs": inputs, "outputs": outputs}
