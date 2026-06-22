import json
import os
import sys

import numpy as np


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND = os.path.join(ROOT, "backend")
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

from synthgpu.onnx_provider import _outputs_match, _parse_ort_profile


def test_parse_ort_profile_aggregates_real_kernel_events(tmp_path):
    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps([
        {
            "name": "dense_kernel_time",
            "dur": 1200,
            "args": {
                "node_name": "dense",
                "op_name": "MatMul",
                "provider": "CPUExecutionProvider",
            },
        },
        {
            "name": "dense_kernel_time",
            "dur": 300,
            "args": {
                "node_name": "dense",
                "op_name": "MatMul",
                "provider": "CPUExecutionProvider",
            },
        },
        {
            "name": "session_initialization",
            "dur": 9000,
            "args": {},
        },
    ]), encoding="utf-8")

    assert _parse_ort_profile(str(profile)) == [{
        "node_name": "dense",
        "op_name": "MatMul",
        "provider": "CPUExecutionProvider",
        "duration_ms": 1.5,
    }]


def test_outputs_match_uses_tolerance_and_shape_gate():
    baseline = [np.array([[1.0, 2.0]], dtype=np.float32)]
    close = [np.array([[1.00001, 2.00001]], dtype=np.float32)]
    wrong_shape = [np.array([1.0, 2.0], dtype=np.float32)]
    wrong_value = [np.array([[1.0, 3.0]], dtype=np.float32)]

    assert _outputs_match(close, baseline)
    assert not _outputs_match(wrong_shape, baseline)
    assert not _outputs_match(wrong_value, baseline)
