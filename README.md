# synthGPU-OS

[![CI/CD](https://github.com/bido75/synthGPU-OS/actions/workflows/ci.yml/badge.svg)](https://github.com/bido75/synthGPU-OS/actions/workflows/ci.yml)
[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/bido75/synthGPU-OS?quickstart=1)

Software-defined GPU compute, Vulkan/CUDA compatibility, ONNX/OpenVINO
instrumentation, and local or remote LLM inference on CPU hardware.

---

## v0.3 — Vulkan ICD (OS Recognition Milestone)

SynthGPU v0.3 implements a real Vulkan ICD (Installable Client Driver).
After installation, `vulkaninfo` and any Vulkan application recognizes
SynthGPU as a Vulkan 1.3 compute device.

### What vulkaninfo shows after install
```
GPU0:
  apiVersion    = 1.3.0
  driverVersion = 0.3.0
  vendorID      = 0x5347
  deviceID      = 0x0003
  deviceType    = OTHER
  deviceName    = SynthGPU Virtual Accelerator v0.3
```

### Build & Install (Windows)
```powershell
# Prerequisites: CMake 3.20+, Vulkan SDK from https://vulkan.lunarg.com

cd vulkan_icd
New-Item -ItemType Directory -Force build; cd build
cmake .. -G "Visual Studio 17 2022" -A x64 -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
cd ..\scripts

# Copy vk_icd.h from Vulkan SDK (one-time):
Copy-Item "$env:VULKAN_SDK\Include\vulkan\vk_icd.h" "..\include\vk_icd.h"

# Run as Administrator:
.\install_windows.bat
vulkaninfo --summary | Select-String SynthGPU
```

### Build & Install (Linux)
```bash
cd vulkan_icd && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release && make -j$(nproc)
cd .. && sudo scripts/install_linux.sh
vulkaninfo --summary | grep SynthGPU
```

### Verify
```powershell
python vulkan_icd/tests/test_enumeration.py
python vulkan_icd/tests/test_compute.py
```

---

**"GPU compute on any CPU — no physical GPU required."**

> *"The same way VMware made servers accessible to everyone, SynthGPU makes GPU compute accessible to everyone."*

---

## Beta v0.2 — Web Dashboard

This is the full beta with a real-time web dashboard, WebSocket telemetry, ONNX model support, and investor demo mode.

### Quick Start

**Backend:**
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

**Frontend (development):**
```bash
cd frontend
npm install
npm run dev
```
Open `http://localhost:5173` — Vite proxies API calls to the backend.

**Production (single port):**
```bash
cd frontend
npm install
npm run build
cd ../backend
uvicorn main:app --host 0.0.0.0 --port 8000
```
Open `http://localhost:8000`

**Windows one-shot:**
```
start.bat
```

---

## Container Installation

### Windows: Docker Engine in Ubuntu WSL2

Run from an Administrator PowerShell session:

```powershell
.\scripts\install.ps1
```

The installer checks virtualization and WSL, installs Ubuntu 24.04 when
needed, installs Docker Engine inside Ubuntu, configures the Windows-host
Ollama route, forwards Windows `localhost:8000` to the WSL2 VM, and starts
SynthGPU at `http://localhost:8000`.

WSL should be current before installation. Check with `wsl --version`; use
`wsl --update` only when an older WSL/kernel is installed.

PowerShell note for future installer scripts: native commands can write
informational progress to stderr even when they succeed. Keep
`$ErrorActionPreference = "Stop"` for PowerShell cmdlets, but judge native
commands such as `wsl.exe`, `apt-get`, `systemctl`, and `docker` by their exit
code instead of treating any stderr output as failure.

### Linux/macOS

```bash
./scripts/install.sh
```

Docker Engine and Docker Compose v2 must already be installed.

### Health Check and Uninstall

```powershell
.\scripts\check-docker-health.ps1
.\scripts\install.ps1 -Uninstall
```

```bash
./scripts/install.sh --uninstall
```

Uninstall removes only SynthGPU containers, locally built images, and
project volumes. It preserves source files and WSL distributions.

---

## Architecture

```
SynthGPU/
├── backend/
│   ├── main.py                    FastAPI + WebSocket server
│   ├── synthgpu/
│   │   ├── device.py              SynthGPU device (main interface)
│   │   ├── warp_scheduler.py      Warp emulation engine (SIMT on CPU threads)
│   │   ├── memory_manager.py      Virtual VRAM manager
│   │   ├── ops/gpu_ops.py         All GPU kernels (NumPy BLAS/SIMD)
│   │   └── onnx_provider.py       ONNX execution provider
│   ├── benchmarks/runner.py       Benchmark engine
│   └── requirements.txt
└── frontend/
    └── src/
        ├── App.jsx                Main layout + WebSocket + investor demo mode
        └── components/
            ├── DevicePanel.jsx    GPU device info card
            ├── WarpMonitor.jsx    Live warp execution visualizer (32 lanes)
            ├── MemoryGauge.jsx    vRAM arc gauge
            ├── PerformanceChart.jsx  Live throughput chart
            ├── BenchmarkRunner.jsx   Run benchmarks from UI
            ├── TokenGenerator.jsx    Live LLM token generation demo
            ├── ModelUploader.jsx     Upload and run ONNX models
            └── EconomicsPanel.jsx   Cost comparison vs real GPU
```

## What Is SynthGPU?

SynthGPU is a software-defined virtual GPU that exposes a GPU compute interface — including parallel kernel execution, virtual VRAM management, and AI inference — using only CPU resources. No physical GPU hardware required.

It works by implementing the **SIMT (Single Instruction, Multiple Thread) execution model** that real GPUs use, mapped to CPU cores and SIMD via NumPy/BLAS.

## Requirements

- Python 3.11+
- Node.js 18+
- No GPU required
