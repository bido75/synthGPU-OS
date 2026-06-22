import ctypes
import os

from cuda_shim.kernels import bridge_api


library_path = os.environ["SYNTHGPU_BRIDGE_TEST_LIB"]
bridge = ctypes.PyDLL(library_path)
bridge.synthgpu_bridge_init.restype = ctypes.c_int
bridge.synthgpu_get_warps_executed.restype = ctypes.c_long
bridge.bridge_sgemm.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_float,
    ctypes.c_float,
    ctypes.c_int,
    ctypes.c_int,
]

assert bridge.synthgpu_bridge_init() == 0

scheduler = bridge_api.get_scheduler()
scheduler_before = scheduler.external_warp_count
fallback_before = bridge.synthgpu_get_warps_executed()

bridge.bridge_sgemm(None, None, None, 64, 64, 64, 1.0, 0.0, 0, 0)

scheduler_delta = scheduler.external_warp_count - scheduler_before
fallback_delta = bridge.synthgpu_get_warps_executed() - fallback_before

print(f"scheduler_delta={scheduler_delta}")
print(f"fallback_delta={fallback_delta}")
assert scheduler_delta == 9
assert fallback_delta == 0
print("Python scheduler bridge runtime check passed")
