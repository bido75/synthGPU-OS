# SynthGPU CUDA Shim — v0.3.0

A shared library that intercepts NVIDIA CUDA API calls and routes them through SynthGPU's CPU-based warp scheduler. Any CUDA application runs on CPU-only hardware with **zero code changes**.

```bash
# On a machine with no GPU hardware:
export LD_PRELOAD=/usr/local/lib/synthgpu/libsynthgpu_cuda.so
python -c "import torch; print(torch.cuda.is_available())"
# → True
python -c "import torch; print(torch.cuda.get_device_name(0))"
# → SynthGPU Virtual Accelerator
```

---

## Directory Layout

```
cuda_shim/
├── CMakeLists.txt          Build configuration
├── setup.py                Python package setup (pip install)
├── README.md               This file
├── src/
│   ├── shim.c              ALL CUDA API interception (25+ functions)
│   ├── memory.c            Virtual VRAM allocator (mmap-backed)
│   ├── memory.h            VRAM allocator header
│   ├── bridge.c            C → Python warp scheduler bridge
│   ├── bridge.h            Bridge header
│   ├── stream.c            CUDA stream emulation
│   ├── event.c             CUDA event timing
│   └── telemetry.c         Dashboard stats reporter
├── include/
│   ├── cuda_runtime.h      CUDA Runtime types
│   ├── cuda_runtime_api.h  Full CUDA API declarations
│   ├── cublas.h            cuBLAS declarations
│   ├── cublas_v2.h         cuBLAS v2 alias header
│   └── cudnn.h             cuDNN declarations
├── kernels/                Python kernel implementations
│   ├── __init__.py
│   ├── bridge_api.py       Central dispatcher (CRITICAL)
│   ├── gemm.py             Matrix multiply
│   ├── attention.py        Scaled dot-product attention
│   ├── embedding.py        Embedding lookup
│   ├── elementwise.py      ReLU, GELU, sigmoid, tanh
│   ├── reduction.py        Softmax, sum, max, mean
│   ├── norm.py             LayerNorm, BatchNorm
│   ├── conv2d.py           2D convolution
│   └── optimizer.py        Adam, SGD step
├── install/
│   ├── install_linux.sh    Linux installer
│   ├── install_windows.bat Windows installer
│   └── synthgpu_profile.sh Shell profile additions
└── tests/
    ├── test_basic.py       Core tests (no PyTorch required)
    ├── test_pytorch.py     PyTorch compatibility (25+ tests)
    ├── test_inference.py   End-to-end inference test
    └── test_perf.py        Performance benchmarks
```

---

## Quick Start

### Option A — Python only (no compilation needed)

```bash
pip install -e cuda_shim/
python -c "from cuda_shim.kernels.bridge_api import cuda_gemm; print('bridge_api OK')"
python cuda_shim/tests/test_basic.py
```

### Option B — Full build with LD_PRELOAD (Linux)

```bash
bash cuda_shim/install/install_linux.sh
source /usr/local/lib/synthgpu/activate.sh
python -c "import torch; print(torch.cuda.is_available())"  # True
```

### Option C — Manual CMake build

```bash
cd cuda_shim
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
# Test:
LD_PRELOAD=./libsynthgpu_cuda.so python -c "import torch; print(torch.cuda.is_available())"
```

---

## Requirements

| Dependency | Purpose | Install |
|---|---|---|
| Python 3.9+ | Bridge API | included |
| NumPy ≥ 1.24 | Kernel compute | `pip install numpy` |
| OpenBLAS | Fast GEMM | `apt install libopenblas-dev` |
| CMake ≥ 3.15 | C build | `apt install cmake` |
| GCC / MSVC | C compiler | `apt install gcc` |

---

## How It Works

```
CUDA App
   │
   │  LD_PRELOAD intercepts CUDA API calls
   ▼
libsynthgpu_cuda.so (shim.c)
   │
   ├── cudaMalloc/Free → memory.c (mmap pool)
   ├── cudaStream*     → stream.c (stubs)
   ├── cudaEvent*      → event.c  (wall-clock timing)
   │
   └── cublasSgemm_v2  → bridge.c → bridge_api.py
                                         │
                                         └── WarpScheduler.dispatch_kernel()
                                                   │
                                                   └── numpy/OpenBLAS compute
                                                       + dashboard telemetry
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `SYNTHGPU_VRAM_MB` | 40% of RAM | Virtual VRAM pool size |
| `SYNTHGPU_ROOT` | auto-detected | Project root for Python path |
| `SYNTHGPU_ACTIVE` | unset | Set to `1` by activate.sh |

---

## Running Tests

```bash
# No PyTorch required:
python cuda_shim/tests/test_basic.py

# Full PyTorch suite (Linux with compiled shim):
LD_PRELOAD=cuda_shim/build/libsynthgpu_cuda.so \
    python cuda_shim/tests/test_pytorch.py

# Performance benchmarks:
python cuda_shim/tests/test_perf.py --quick

# End-to-end inference:
python cuda_shim/tests/test_inference.py
```

---

## CUDA API Coverage

### Runtime (shim.c)
`cudaGetDeviceCount` · `cudaGetDevice` · `cudaSetDevice` · `cudaGetDeviceProperties` ·
`cudaMalloc` · `cudaMallocManaged` · `cudaMallocHost` · `cudaFree` · `cudaFreeHost` ·
`cudaMemcpy` · `cudaMemcpyAsync` · `cudaMemset` · `cudaMemsetAsync` · `cudaMemGetInfo` ·
`cudaDeviceSynchronize` · `cudaDeviceReset` · `cudaGetLastError` · `cudaPeekAtLastError` ·
`cudaGetErrorString` · `cudaGetErrorName` · `cudaDriverGetVersion` · `cudaRuntimeGetVersion` ·
`cudaStreamCreate` · `cudaStreamDestroy` · `cudaStreamSynchronize` ·
`cudaEventCreate` · `cudaEventRecord` · `cudaEventElapsedTime` · `cudaEventDestroy`

### cuBLAS (shim.c)
`cublasCreate_v2` · `cublasDestroy_v2` · `cublasSetStream_v2` · `cublasSetMathMode` ·
`cublasSgemm_v2` · `cublasDgemm_v2` · `cublasGemmEx` ·
`cublasSgemmBatched` · `cublasSgemmStridedBatched`
