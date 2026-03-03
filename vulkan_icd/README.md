# SynthGPU Vulkan ICD v0.3

A real Vulkan Installable Client Driver (ICD) that makes **SynthGPU Virtual Accelerator**
appear as a recognized compute device to `vulkaninfo` and any Vulkan application.

After installation, `vulkaninfo --summary` will show:

```
GPU0:
  apiVersion    = 1.3.0
  driverVersion = 0.3.0
  vendorID      = 0x5347
  deviceID      = 0x0003
  deviceType    = OTHER
  deviceName    = SynthGPU Virtual Accelerator v0.3
```

---

## Build

### Prerequisites
- CMake 3.20+
- Vulkan SDK (https://vulkan.lunarg.com)
- MSVC 2022 (Windows) or GCC/Clang (Linux)
- Python 3.x development headers (optional — enables warp scheduler bridge)

### Windows (Developer PowerShell)

```powershell
cd vulkan_icd
New-Item -ItemType Directory -Force build
cd build
cmake .. -G "Visual Studio 17 2022" -A x64 -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
cd ..
# Run as Administrator:
scripts\install_windows.bat
vulkaninfo --summary | Select-String SynthGPU
```

### Linux

```bash
cd vulkan_icd
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
cd ..
sudo scripts/install_linux.sh
vulkaninfo --summary | grep SynthGPU
```

---

## File Structure

```
vulkan_icd/
├── CMakeLists.txt              Build system
├── include/
│   ├── synthgpu_vk.h           Internal types + object definitions
│   └── spirv_dispatch.h        SPIR-V dispatch context
├── src/
│   ├── icd_main.c              Entry: vk_icdGetInstanceProcAddr + dispatch table
│   ├── instance.c              vkCreateInstance / vkDestroyInstance
│   ├── physical_device.c       vkEnumeratePhysicalDevices + device properties
│   ├── device.c                vkCreateDevice + queues
│   ├── memory.c                vkAllocateMemory + buffers (malloc-backed)
│   ├── pipeline.c              vkCreateComputePipelines + descriptors
│   ├── commands.c              Command buffer recording
│   ├── queue.c                 vkQueueSubmit + execution engine + sync objects
│   └── spirv_dispatch.c        SPIR-V interpreter + Python warp scheduler bridge
├── bridge/
│   └── py_bridge.c             ctypes bridge for Python warp scheduler
├── manifests/
│   ├── synthgpu_icd_win64.json Windows ICD manifest
│   └── synthgpu_icd_linux.json Linux ICD manifest
├── scripts/
│   ├── install_windows.bat     Registry registration + manifest install
│   ├── install_linux.sh        /etc/vulkan/icd.d installation
│   ├── uninstall.bat
│   └── uninstall.sh
└── tests/
    ├── test_enumeration.py     Verify vulkaninfo shows SynthGPU
    └── test_compute.py         Run a Vulkan buffer copy through SynthGPU
```

---

## Verification

```powershell
# After install:
python vulkan_icd/tests/test_enumeration.py
python vulkan_icd/tests/test_compute.py
```

---

## Architecture

```
Application
    │  vkCreateInstance()
    ▼
Vulkan Loader  (reads HKLM\SOFTWARE\Khronos\Vulkan\Drivers on Windows)
    │
    ├─► synthgpu_vulkan_icd.dll   ← THIS ICD
    │       │
    │       ├── physical_device.c   reports "SynthGPU Virtual Accelerator v0.3"
    │       ├── memory.c            malloc-backed 128MB virtual VRAM
    │       ├── pipeline.c          ingests SPIR-V, extracts LocalSize
    │       ├── commands.c          records CmdDispatch / CmdCopyBuffer
    │       ├── queue.c             synchronous execution engine
    │       └── spirv_dispatch.c    interprets SPIR-V → warp scheduler
    │                │
    │                └──► Python WarpScheduler.record_external_warps()
    │                         │
    │                         └──► SynthGPU Dashboard (CUDA Shim panel)
    │
    └─► Real GPU ICD (if present)
```

The ICD advertises a compute-only device (no graphics queue, no swapchain).
All memory is host-visible + host-coherent + device-local — zero copy.
Execution is synchronous (no async scheduling needed for CPU emulation).
