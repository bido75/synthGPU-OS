import ctypes
import json
from pathlib import Path
from urllib.request import urlopen


LIBRARY = "/usr/local/lib/synthgpu/libsynthgpu_vulkan_icd.so"
TELEMETRY = Path("/tmp/synthgpu_vulkan_warps.tmp")
SPIRV_MAGIC = 0x07230203


class BufferBinding(ctypes.Structure):
    _fields_ = [
        ("ptr", ctypes.c_void_p),
        ("size", ctypes.c_size_t),
        ("set", ctypes.c_uint32),
        ("binding", ctypes.c_uint32),
    ]


class DispatchContext(ctypes.Structure):
    _fields_ = [
        ("spirv_code", ctypes.POINTER(ctypes.c_uint32)),
        ("spirv_word_count", ctypes.c_size_t),
        ("local_size_x", ctypes.c_uint32),
        ("local_size_y", ctypes.c_uint32),
        ("local_size_z", ctypes.c_uint32),
        ("group_count_x", ctypes.c_uint32),
        ("group_count_y", ctypes.c_uint32),
        ("group_count_z", ctypes.c_uint32),
        ("bindings", BufferBinding * 32),
        ("binding_count", ctypes.c_uint32),
    ]


def telemetry_rows() -> list[str]:
    if not TELEMETRY.exists():
        return []
    return [line for line in TELEMETRY.read_text().splitlines() if line]


before_rows = telemetry_rows()
code = (ctypes.c_uint32 * 5)(SPIRV_MAGIC, 0, 0, 1, 0)
context = DispatchContext(
    spirv_code=code,
    spirv_word_count=5,
    local_size_x=2,
    local_size_y=1,
    local_size_z=1,
    group_count_x=3,
    group_count_y=1,
    group_count_z=1,
    binding_count=0,
)

library = ctypes.CDLL(LIBRARY)
dispatch = library.synthgpu_spirv_dispatch
dispatch.argtypes = [ctypes.POINTER(DispatchContext)]
dispatch.restype = ctypes.c_int32
assert dispatch(ctypes.byref(context)) == 0

after_rows = telemetry_rows()
assert len(after_rows) == len(before_rows) + 1
groups, exec_ms = after_rows[-1].split(",")
assert int(groups) == 3
assert float(exec_ms) == 0.01

with urlopen("http://localhost:8000/api/vulkan/status", timeout=5) as response:
    status = json.load(response)

print(f"rows_before={len(before_rows)}")
print(f"rows_after={len(after_rows)}")
print(f"last_row={after_rows[-1]}")
print(f"api_dispatch_count={status['dispatch_count']}")
print(f"api_last_dispatch_ms={status['last_dispatch_ms']}")
assert status["dispatch_count"] == len(after_rows)
assert status["last_dispatch_ms"] == float(exec_ms)
print("Vulkan telemetry writer/parser runtime check passed")
