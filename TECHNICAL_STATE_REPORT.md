# SynthGPU — Technical State Report

**Generated:** 2026-06-19  
**Version in Code:** v0.1.0-mvp (root `synthgpu/`), v0.2.0-beta (`backend/synthgpu/`), v0.3.0 (Docker labels, shim versions)  
**Container State:** Running (healthy, port 8000)  
**Audit Scope:** Full repository — 10 top-level directories, ~156 files, ~890KB source  

---

## 1. Executive Summary

### Purpose & Value Proposition

SynthGPU is a **software-defined virtual GPU accelerator** that enables AI inference workloads to run on machines without physical GPU hardware. It achieves this by:

1. **Emulating GPU compute primitives** via NumPy/BLAS on CPU (GEMM, attention, layer norm, softmax)
2. **Providing ABI-compatible shims** that intercept CUDA runtime calls (`cuda_runtime.h`, `cublas.h`, `cudnn.h`) and route them through the virtual device
3. **Exposing a compatible Vulkan ICD** that registers a virtual GPU with the system Vulkan loader
4. **Serving a real-time dashboard** with WebSocket telemetry, warp execution visualization, and LLM inference console
5. **Proxying through to real LLM backends** (Ollama, LM Studio) transparently, enriching responses with simulation metadata

### System Maturity

| Dimension | Assessment |
|-----------|-----------|
| **Core simulation engine** | Mature — `WarpScheduler`, `VirtualMemoryManager`, `gpu_ops` functional with telemetry |
| **CUDA shim (C layer)** | Alpha — builds, installs, but Python bridge runtime integration incomplete |
| **Vulkan ICD** | Alpha — builds and registers with system, basic dispatch tracking |
| **LLM inference proxy** | Mature — Ollama/LM Studio integration with telemetry augmentation |
| **Frontend dashboard** | Beta — 13 components, WebSocket live telemetry, 4 tabs, investor demo mode |
| **Containerization** | Beta — 3-stage Dockerfile, compose orchestration, health-checked |
| **CI/CD** | Alpha — GitHub Actions with frontend build + lint, no actual test suite execution |
| **Test coverage** | **Low** — 2 test files (388 lines total), no coverage metrics |
| **Documentation** | Minimal — README, OLLAMA_INTEGRATION.md, no API docs, no architecture docs |

---

## 2. Architecture & System Topology

### Tech Stack

| Layer | Technology | Version |
|-------|-----------|---------|
| **Backend runtime** | Python 3.11 | 3.11.15 (slim-bookworm) |
| **Web framework** | FastAPI | 0.137.2 |
| **ASGI server** | Uvicorn | 0.49.0 (standard extras) |
| **Frontend framework** | React | 18.2.0 |
| **Frontend build** | Vite | 5.x |
| **Styling** | Tailwind CSS | 3.4.x |
| **Charts** | Recharts | 2.10.x |
| **Compute backend** | NumPy (BLAS) | 2.4.6 |
| **ONNX runtime** | onnxruntime | 1.27.0 |
| **System/process utils** | psutil | 7.2.2 |
| **Async HTTP** | httpx | 0.28.1 |
| **C extensions** | C11 + CMake | GCC (Debian bookworm) |
| **Container base** | python:3.11-slim-bookworm | Debian 12 |
| **Node build** | node:20-slim | Node 20 |

### Architectural Pattern

**Monolith with Service Proxy** — The backend is a single FastAPI process serving:

- REST API endpoints (~25 routes)
- WebSocket endpoints (telemetry + token streaming)
- Static frontend SPA
- LLM backend proxy (Ollama/LM Studio passthrough + telemetry injection)

The architecture is not microservices — one Python process handles everything. This is appropriate for the MVP phase.

### Data Flow

```
Browser (React SPA)          Docker Container (synthgpu-core)
┌──────────────────┐        ┌────────────────────────────────────┐
│  App.jsx          │  WSS   │  FastAPI (uvicorn)                  │
│  ┌──────────────┐ │◄─────►│  ┌──────────┐  ┌───────────────┐   │
│  │ WebSocket    │ │  REST │  │ REST API │  │ WebSocket     │   │
│  │ telemetry    │ │──────►│  │ 25 routes│  │ broadcast     │   │
│  │ WarpMonitor  │ │       │  └────┬─────┘  └───────┬───────┘   │
│  │ MemoryGauge  │ │       │       │                │           │
│  │ WarpMonitor  │ │       │       ▼                ▼           │
│  │ RAMMonitor   │ │       │  ┌─────────────────────────┐      │
│  │ CudaShimSt.  │ │       │  │ SynthGPU Virtual Device │      │
│  └──────────────┘ │       │  │  ┌───────────────────┐  │      │
│                   │       │  │  │ WarpScheduler     │  │      │
│  External:        │       │  │  │ ThreadPoolExecutor│  │      │
│  Ollama/LM Studio │       │  │  │ 2-4 compute units │  │      │
│  ┌────────────┐   │  HTTP  │  │  ├───────────────────┤  │      │
│  │ /api/chat  │◄──┼────────┼──┼──┤ VirtualMemoryMgr  │  │      │
│  │ /api/gen   │   │       │  │  │ mmap-backed pool   │  │      │
│  └────────────┘   │       │  │  └───────────────────┘  │      │
│                   │       │  └─────────────────────────┘      │
└──────────────────┘        └────────────────────────────────────┘

Storage:
  - In-memory: all device state, telemetry, session history
  - /tmp: uploaded ONNX models (ephemeral)
  - mmap: virtual VRAM backing store (ephemeral)
  - Docker volume synthgpu-tmp: /tmp (persists across container restarts)
```

---

## 3. Component & Module Breakdown

### 3.1 Directory Map

```
SynthGPU/
├── backend/                    # Python backend (v0.2.0-beta structure)
│   ├── main.py                 # FastAPI app — 1032 lines, 25+ REST endpoints, 2 WebSockets
│   ├── requirements.txt        # Backend Python dependencies
│   ├── benchmarks/
│   │   ├── runner.py           # Benchmark runner (GEMM, MLP, Transformer, token gen)
│   │   └── __init__.py         # Empty
│   └── synthgpu/               # Core device package (flat layout, no core/ subdir)
│       ├── __init__.py         # Exports SynthGPU, WarpScheduler, VirtualMemoryManager
│       ├── device.py           # SynGPU class — top-level device interface, self-test, matmul/linear/attention/MLP
│       ├── warp_scheduler.py   # WarpScheduler — SIMT warp emulation via ThreadPoolExecutor
│       ├── memory_manager.py   # VirtualMemoryManager — mmap-backed VRAM with constrained/degraded modes
│       ├── onnx_provider.py    # SynthGPUExecutionProvider — ONNX Runtime integration (CPU fallback)
│       ├── inference_proxy.py  # SynthGPUTelemetryEngine + InferenceProxyRouter — Ollama/LM Studio proxy
│       └── ops/
│           ├── __init__.py     # Re-exports all GPU ops
│           └── gpu_ops.py      # Pure NumPy implementations: gemm, softmax, layer_norm, attention, conv2d, etc.
│
├── synthgpu/                   # Legacy root-level package (v0.1.0-mvp) — *** CAUSES IMPORT COLLISIONS ***
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── device.py           # Older device implementation (core/ subpackage layout)
│   │   ├── warp_scheduler.py   # Older scheduler
│   │   └── memory_manager.py   # Older memory manager
│   └── ops/
│       ├── __init__.py
│       └── gpu_ops.py          # Older ops
│
├── cuda_shim/                  # CUDA ABI shim — C interceptors + Python kernel bridge
│   ├── CMakeLists.txt          # CMake build
│   ├── setup.py                # pip install for Python layer
│   ├── include/                # C headers that intercept CUDA APIs
│   │   ├── cuda_runtime.h      # Intercepts cudaMalloc, cudaMemcpy, cudaFree
│   │   ├── cublas.h            # Intercepts cublasSgemm, cublasGemmEx
│   │   ├── cudnn.h             # Intercepts cudnnSoftmaxForward, cudnnActivationForward
│   │   └── synthgpu_cuda.h     # SynthGPU shim internal API
│   ├── src/                    # C implementation
│   │   ├── shim.c              # Main shim orchestrator (~17KB, largest C file)
│   │   ├── cublas.c            # cuBLAS op routing
│   │   ├── cudnn.c             # cuDNN op routing
│   │   ├── device_info.c       # Device property reporting
│   │   ├── memory.c            # Memory intercept (cudaMalloc/cudaFree tracking)
│   │   ├── virtual_vram.c      # Virtual VRAM pool management
│   │   ├── python_bridge.c     # C → Python bridge (calls bridge_api.py)
│   │   ├── bridge.c/h          # Bridge state management
│   │   ├── event.c             # CUDA event interceptor
│   │   ├── stream.c            # CUDA stream interceptor
│   │   └── telemetry.c         # Warp dispatch counters
│   └── kernels/                # Python-side kernel implementations
│       ├── bridge_api.py       # C → Python bridge: scheduler, memory, GEMM/softmax/ReLU etc.
│       ├── gemm.py, attention.py, norm.py, conv2d.py, etc.
│
├── vulkan_icd/                 # Vulkan Installable Client Driver
│   ├── CMakeLists.txt
│   ├── include/                # Vulkan headers + SynthGPU VK extensions
│   ├── src/                    # ICD implementation (icd_main.c, device.c, queue.c, etc.)
│   ├── manifests/              # ICD JSON manifests (Linux, Windows, Debug)
│   └── tests/                  # Python tests for Vulkan ICD
│
├── frontend/                   # React SPA (v0.2.0)
│   ├── src/
│   │   ├── App.jsx             # Root — tab routing, WebSocket, demo mode
│   │   ├── main.jsx            # ReactDOM entry
│   │   ├── index.css           # Tailwind + custom animations
│   │   └── components/         # 13 UI components
│   │       ├── DevicePanel.jsx         # Virtual GPU device info card
│   │       ├── WarpMonitor.jsx         # Real-time warp lane visualization
│   │       ├── MemoryGauge.jsx         # VRAM usage gauge
│   │       ├── RAMMonitor.jsx          # System RAM + swap warning
│   │       ├── SystemRAMMonitor.jsx    # RAM utilization + risk levels
│   │       ├── CudaShimStatus.jsx      # CUDA shim status widget
│   │       ├── PerformanceChart.jsx    # Recharts line chart
│   │       ├── BenchmarkRunner.jsx     # Benchmark UI + results table
│   │       ├── TokenGenerator.jsx      # Token generation demo
│   │       ├── LLMInference.jsx        # Full LLM inference console (926 lines)
│   │       ├── ModelUploader.jsx       # ONNX model upload + run
│   │       ├── EconomicsPanel.jsx      # Cost comparison display
│   │       └── DemoReadyBadge.jsx      # Infrastructure readiness badge
│   └── configs: vite.config.js, tailwind.config.js, postcss.config.js
│
├── tests/                      # Test suites (limited)
│   ├── e2e/test_dashboard.py   # Playwright E2E + API tests (173 lines)
│   └── test_cuda_shim.py       # CUDA shim Python kernel tests (215 lines)
│
├── benchmarks/                 # Standalone benchmark suite
│   └── benchmark_suite.py      # Offline benchmarks (GEMM, MLP, Transformer, token gen)
│
├── demos/                      # Investor demo script
│   └── investor_demo.py        # Narrated product demo (296 lines)
│
├── startup scripts:
│   ├── start.bat               # Windows launcher
│   ├── start_all_windows.bat   # Full Windows startup
│   ├── start_ollama_windows.bat
│   └── push.ps1                # Docker push helper
│
├── Dockerfile                  # 3-stage: c-builder → frontend-builder → runtime
├── docker-compose.yml          # 2 services: synthgpu-core + synthgpu-client-node
├── synthgpu-pod.yaml            # Kubernetes pod manifest
├── README.md                   # Project overview
├── OLLAMA_INTEGRATION.md       # Ollama usage guide
└── requirements.txt            # Root-level deps (superset of backend deps)
```

### 3.2 Key Entry Points

| Entry Point | File | Purpose |
|------------|------|---------|
| `uvicorn backend.main:app` | `backend/main.py:1` | FastAPI application — the main process |
| `GET /` | `backend/main.py:1018` | Serves frontend SPA (`index.html`) |
| `WS /ws/telemetry` | `backend/main.py:249` | Real-time device telemetry (200ms interval) |
| `WS /ws/tokens` | `backend/main.py:266` | Token streaming during inference |
| `GET /api/device/status` | `backend/main.py:454` | Health check + device status |
| `GET /api/device/info` | `backend/main.py:384` | Full device telemetry snapshot |
| `GET /api/cuda_shim/status` | `backend/main.py:901` | CUDA shim availability + stats |
| `GET /api/vulkan/status` | `backend/main.py:949` | Vulkan ICD status + dispatch count |
| `POST /api/inference/run` | `inference_proxy.py:719` | LLM inference streaming entry |
| `GET /api/system/ram` | `backend/main.py:464` | RAM + swap status |
| `GET /api/health/demo_ready` | `backend/main.py:512` | Demo readiness check |

### 3.3 Core Business Logic Files

| File | Lines | Responsibility |
|------|-------|---------------|
| `backend/synthgpu/device.py` | 245 | Top-level SynthGPU device — public API surface: `matmul()`, `linear()`, `attention()`, `run_inference()`, `generate_tokens()` |
| `backend/synthgpu/warp_scheduler.py` | 192 | SIMT warp emulation — `ThreadPoolExecutor`-based, `WarpSize=32`, per-warp lane masking, throughput sampling |
| `backend/synthgpu/memory_manager.py` | 447 | Virtual VRAM — mmap-backed, constrained/degraded mode detection (<16GB RAM), allocation tracking |
| `backend/synthgpu/ops/gpu_ops.py` | 111 | GPU operation primitives — all via NumPy: `gemm`, `softmax`, `layer_norm`, `attention`, `conv2d`, `gelu` |
| `backend/synthgpu/inference_proxy.py` | 986 | LLM proxy — Ollama/LM Studio auto-detection, `SynthGPUTelemetryEngine`, `InferenceProxyRouter` |
| `backend/synthgpu/onnx_provider.py` | 69 | ONNX Runtime integration — routes through `CPUExecutionProvider` |
| `cuda_shim/src/shim.c` | ~500 | C shim orchestrator — intercepts CUDA calls, routes to Python bridge |
| `cuda_shim/kernels/bridge_api.py` | 205 | Python bridge — C-to-Python dispatch, warp recording, kernel implementations |
| `vulkan_icd/src/icd_main.c` | ~600 | Vulkan ICD driver entry — `vkCreateInstance`, `vkEnumeratePhysicalDevices`, etc. |
| `frontend/src/App.jsx` | 362 | React root — tab routing, WebSocket lifecycle, demo mode state machine |

---

## 4. Data Model & Storage

### 4.1 In-Memory State (No Database)

SynthGPU is **stateless with respect to persistent storage** — there is no database, no SQL, no schema migrations. All state lives in Python process memory:

| Object | Type | Scope | Lifetime |
|--------|------|-------|----------|
| `gpu` (SynthGPU) | Global singleton | Process | Entire uptime |
| `warp_scheduler` | Instance attribute | Per-gpu | Entire uptime |
| `memory_manager` | Instance attribute | Per-gpu | Entire uptime |
| `inference_engine` | Optional singleton | Process | Entire uptime (if proxy available) |
| `_telemetry_connections` | `Set[WebSocket]` | Global | Runtime (connections come/go) |
| `_token_connections` | `Set[WebSocket]` | Global | Runtime |
| `_benchmark_progress` | `Optional[dict]` | Global | During benchmark run |
| `_uploaded_models` | `Dict[str, dict]` | Global | Until server restart |
| `_inference_state` | Module-level dict | `inference_proxy` | Process |
| `vulkan_dispatch_count` | Global int | `main.py` | Entire uptime |
| `warp_history` | `deque(maxlen=100)` | Per-scheduler | Rolling 100 entries |
| `token_history` | `deque(maxlen=100)` | Per-engine | Rolling 100 entries |
| `session_history` | `list` | Per-engine | Last 20 sessions |

### 4.2 Virtual VRAM Model

`VirtualMemoryManager` uses an **mmap-backed** allocation pool:

```
┌────────────────────────────────────────────────┐
│  Virtual VRAM Pool (mmap -1, size)              │
│  ├─ Allocation #1 (handle=1, size, offset=0)    │
│  ├─ Allocation #2 (handle=2, size, offset=N)    │
│  └─ ... linear layout by allocation order        │
├────────────────────────────────────────────────┤
│  Two modes:                                      │
│  • Standard (≥16GB RAM): 40% of host RAM        │
│  • Constrained (<16GB RAM): 128-256MB cap        │
│  Degraded mode: matrix ops capped at 64×64       │
└────────────────────────────────────────────────┘
```

Key thresholds:
- **CONSTRAINED_RAM_THRESHOLD_MB**: 16,384 (16GB)
- **CONSTRAINED_POOL_MAX_MB**: 256
- **DEGRADED_MATRIX_MAX_DIM**: 64
- Allocations rounded to 64MB boundaries

### 4.3 State Management

**Server-side**: Fully in-memory, no caching layer, no database. State is lost on process restart except for `/tmp` files (Docker volume persists ONNX models).

**Client-side**: React component state (`useState`, `useRef`) with WebSocket-driven updates. No Redux, no React Query. Each component fetches its own data via REST or WebSocket:
- `RAMMonitor`: Polls `GET /api/system/ram` every 2 seconds
- `CudaShimStatus`: Polls `GET /api/cuda_shim/status` every 3 seconds
- `WarpMonitor`: Receives push updates via WebSocket every 200ms
- `App.jsx`: Maintains centralized `telemetry` state from WebSocket

---

## 5. Integration & Infrastructure

### 5.1 Authentication & Authorization

**None.** The application has:
- No authentication middleware
- No API keys
- No CORS restrictions (allow all origins)
- No rate limiting
- No user accounts or sessions

The `CORSMiddleware` is configured with `allow_origins=["*"]`, meaning any website can make cross-origin requests to the API. This is appropriate for a local-development/Docker internal-network tool but would be a significant security concern if exposed to a network.

### 5.2 Third-Party Integrations

| Integration | Type | Protocol | Notes |
|-------------|------|----------|-------|
| **Ollama** | LLM backend | HTTP REST (`/api/generate`, `/api/chat`, `/api/tags`) | Auto-detected on startup, retried every 15s |
| **LM Studio** | LLM backend | HTTP REST (`/v1/models`, OpenAI-compatible) | Auto-detected on startup |
| **ONNX Runtime** | ML inference | Python SDK (`onnxruntime`) | CPU provider only — SynthGPUExecutionProvider is a pass-through |
| **psutil** | System metrics | Python SDK | RAM, swap, process info |
| **NumPy** | Compute | Python SDK | All GPU operations implemented via NumPy |

### 5.3 Deployment & CI/CD

**Container Build (3-stage Dockerfile):**

```
Stage 1: c-builder (python:3.11-slim-bookworm)
  ├─ Install build-essential, cmake, libvulkan-dev, libopenblas-dev
  ├─ Build vulkan_icd/ → libsynthgpu_vulkan_icd.so
  └─ Build cuda_shim/  → libsynthgpu_cuda.so

Stage 2: frontend-builder (node:20-slim)
  ├─ npm ci + npm run build → frontend/dist/

Stage 3: runtime (python:3.11-slim-bookworm)
  ├─ Install runtime deps: libvulkan1, libopenblas0
  ├─ COPY backend/ + cuda_shim/ + requirements.txt
  ├─ pip install (two requirements.txt files)
  ├─ COPY frontend/dist/ + C .so files
  ├─ Install Vulkan ICD manifest
  ├─ HEALTHCHECK on /api/device/status
  └─ USER synthgpu → uvicorn backend.main:app
```

**Build Characteristics:**
- Final image size: ~670MB (153MB content, ~517MB shared base layers)
- Build time: ~5-10 minutes (cached builds in ~2-3 minutes)
- Retry loops on apt-get and pip (3 attempts for flaky networks)
- Graceful failure on C extension build failures (Vulkan ICD + CUDA shim are optional)

**CI/CD Pipeline (`.github/workflows/ci.yml`):**
- Triggers: push to main/develop, PR to main
- Jobs:
  1. **frontend-build**: Node 20, npm ci, npm run build, upload dist artifact
  2. **backend-test**: Python 3.11, pip install, `ruff check` (ignore E501), `pytest --tb=short -q` (but no actual test discovery path configured)
  3. **release**: Download frontend dist, package zip, upload as artifact
- **Critical gap**: No Docker build/push in CI, no integration tests, pytest discovers nothing by default (test files are in `tests/` but `pytest` runs from root without `tests/` path)

**Environment Variables:**
```
PYTHONPATH=/opt/synthgpu/backend:/opt/synthgpu
SYNTHGPU_ROOT=/opt/synthgpu
SYNTHGPU_DOCKER=1
OLLAMA_NUM_PARALLEL=1
OLLAMA_MAX_LOADED_MODELS=1
OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-4}
OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
SYNTHGPU_VRAM_MB="${SYNTHGPU_VRAM_MB:-}"  # Optional override
```

**Note:** No `.env` or `.env.example` files exist in the repository. Environment variable documentation is embedded in source code only.

**Kubernetes:** A `synthgpu-pod.yaml` manifest exists (4.7KB) suggesting K8s deployment was considered but is not actively used.

---

## 6. Current Technical Debt & Recommendations

### 6.1 Critical Issues

#### 6.1.1 Dual `synthgpu/` Package Collision

**Severity: HIGH** — Causes `ModuleNotFoundError` on container startup

The repository contains TWO conflicting `synthgpu` Python packages:
1. `SynthGPU/synthgpu/` — root level, v0.1.0-mvp, structure: `synthgpu/core/*`, `synthgpu/ops/*`
2. `SynthGPU/backend/synthgpu/` — backend level, v0.2.0-beta, structure: `synthgpu/device.py`, `synthgpu/warp_scheduler.py` (flat)

The root-level package was accidentally included in the Docker build context and copied into the runtime image. Since `PYTHONPATH=/opt/synthgpu/backend:/opt/synthgpu` AND `''` (CWD = `/opt/synthgpu`) is the first `sys.path` entry, Python discovers `/opt/synthgpu/synthgpu/` (v0.1.0-mvp) before `/opt/synthgpu/backend/synthgpu/` (v0.2.0-beta). The v0.1.0-mvp package lacks `device.py` at the top level (it's in `core/`), causing:

```
ModuleNotFoundError: No module named 'synthgpu.device'
```

**Fix applied (2026-06-19):** Removed `COPY synthgpu/ synthgpu/` from the Docker runtime stage. However, the root-level `synthgpu/` directory remains in the repository, creating ongoing confusion. It should be deleted, moved to `archive/`, or clearly documented.

**Recommendation:**
- [ ] Delete `SynthGPU/synthgpu/` directory (or move to `archive/v0.1.0-mvp/`)
- [ ] Add `.dockerignore` entry for `synthgpu/` to prevent accidental re-inclusion
- [ ] Normalize version strings to a single source of truth

#### 6.1.2 CUDA Shim Python Package Missing from Runtime Image

**Severity: HIGH** — CUDA shim status endpoint returns fake data

The `cuda_shim/` Python package (containing `kernels/bridge_api.py` and all kernel modules) was never copied into the Docker runtime stage. The C library `libsynthgpu_cuda.so` IS present at `/usr/local/lib/synthgpu/`, but the Python bridge that wraps it at `/opt/synthgpu/cuda_shim/kernels/bridge_api.py` is missing.

The `/api/cuda_shim/status` endpoint catches the `ImportError` and returns `"available": True` anyway, making the dashboard show a green "Active" badge when the shim is completely absent.

**Fix applied (2026-06-19):** Added `COPY cuda_shim/ cuda_shim/` to the Docker runtime stage and changed the endpoint to return `"active": _SHIM_AVAILABLE` instead of hardcoded `True`.

#### 6.1.3 Swap Warning Threshold Too Aggressive

**Severity: MEDIUM** — False-positive "performance degraded" warning

The threshold in `backend/main.py:505` was `swap_used_mb > 100`, which triggered in Docker Desktop's Linux VM even with 2.4GB free RAM (normal kernel page-out behavior). The frontend `RAMMonitor.jsx:60` showed a red "Disk swap active — performance degraded" warning that was misleading.

**Fix applied (2026-06-19):** Changed to `swap_used_mb > 500 or (swap_used_mb > 100 and available_mb < 512)` — only warns when swap is heavy (>500MB) OR when swap is active AND RAM is critically low.

### 6.2 Moderate Issues

#### 6.2.1 Package Version Inconsistency

Three different version strings exist across the codebase:
- `v0.1.0-mvp` — root-level `synthgpu/__init__.py`
- `v0.2.0-beta` — `backend/synthgpu/__init__.py`, `frontend/package.json`
- `v0.3.0` — Dockerfile labels, CUDA shim `setup.py`, shim version fields in API responses

**Recommendation:**
- [ ] Unify all version strings to a single `VERSION` source (e.g., `synthgpu/_version.py`)
- [ ] Import version into Dockerfile labels via build args or environment

#### 6.2.2 Build Context Bloat

The Docker build context includes `backend/venv/` (Python virtual environment, ~100MB+). The `.dockerignore` does NOT exclude `backend/venv/` or `backend/.venv/`. This slows down context transfer to the Docker daemon.

Similarly, the root-level `synthgpu/` directory and other development artifacts are in the build context.

**Recommendation:**
- [ ] Add `backend/venv/`, `backend/.venv/`, `backend/__pycache__/` to `.dockerignore`
- [ ] Verify `.dockerignore` matches `.gitignore` patterns

#### 6.2.3 Duplicate Dependency Files

Two `requirements.txt` files exist:
- Root-level: `numpy`, `psutil`, `onnxruntime` (project-level deps)
- `backend/requirements.txt`: `fastapi`, `uvicorn`, `websockets`, `numpy`, `psutil`, `onnxruntime`, `python-multipart`, `httpx`

Both are installed in the Dockerfile, leading to duplicate pip operations. The root-level file is a subset of the backend file.

**Recommendation:**
- [ ] Consolidate into a single `requirements.txt` (prefer `backend/requirements.txt`)
- [ ] Remove the root-level `requirements.txt` to avoid confusion

#### 6.2.4 Missing `.env` Configuration

No `.env` or `.env.example` file exists. Environment variables for configuration are scattered:
- `backend/main.py:6-15` — BLAS thread counts
- `docker-compose.yml:26-30` — `SYNTHGPU_VRAM_MB`, OpenBLAS threads, OMP threads
- `Dockerfile:103-107` — `PYTHONPATH`, `SYNTHGPU_ROOT`, `SYNTHGPU_DOCKER`

**Recommendation:**
- [ ] Create `.env.example` with all configurable variables and documentation
- [ ] Use `pydantic-settings` for centralized configuration management

#### 6.2.5 No Database — All State Ephemeral

Every container restart loses:
- Uploaded ONNX models (stored in `/tmp` which IS persisted via Docker volume, but not indexed)
- Benchmark results
- Inference session history
- Warp telemetry history

**Recommendation:**
- [ ] Add SQLite for lightweight persistent state (session history, metrics)
- [ ] Or implement a telemetry export endpoint for periodic data collection

### 6.3 Security Issues

#### 6.3.1 CORS Fully Open

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)
```

**Risk:** Any website visited by a user on the same machine can make API calls to SynthGPU, potentially triggering model execution or data extraction.

**Recommendation:**
- [ ] Restrict to specific origins in production (`localhost:8000`, `localhost:5173` for dev)
- [ ] Or serve API and frontend from same origin (already the case in Docker — this is safe)

#### 6.3.2 No Input Validation on Inference Endpoints

The `/api/inference/run` and `/api/model/{id}/run` endpoints accept arbitrary JSON bodies and forward them to Ollama/ONNX Runtime without comprehensive validation. The preflight check does basic size estimation but does not validate model names, prompt content, or parameter ranges.

**Recommendation:**
- [ ] Add Pydantic models with field validation for all request bodies
- [ ] Implement rate limiting per client IP
- [ ] Add maximum prompt length enforcement

#### 6.3.3 No Authentication on Uploaded Model Execution

The `/api/model/upload` and `/api/model/{id}/run` endpoints allow arbitrary ONNX model upload and execution with no authentication. A malicious user could upload a model designed to consume excessive memory.

**Recommendation:**
- [ ] Add memory limits during ONNX execution
- [ ] Validate model structure before loading

### 6.4 Test Coverage Gaps

| Area | Lines of Code | Tests | Coverage (est.) |
|------|---------------|-------|-----------------|
| `backend/main.py` | 1032 | 3 E2E API tests | <5% |
| `backend/synthgpu/device.py` | 245 | 0 direct tests | 0% |
| `backend/synthgpu/warp_scheduler.py` | 192 | 4 integration tests | ~15% |
| `backend/synthgpu/memory_manager.py` | 447 | 0 direct tests | 0% |
| `backend/synthgpu/inference_proxy.py` | 986 | 0 tests | 0% |
| `backend/synthgpu/ops/gpu_ops.py` | 111 | 0 tests | 0% |
| `backend/synthgpu/onnx_provider.py` | 69 | 0 tests | 0% |
| `cuda_shim/kernels/bridge_api.py` | 205 | 11 Python tests | ~50% |
| `cuda_shim/src/*.c` | ~2,500 | 0 C tests | 0% |
| `vulkan_icd/src/*.c` | ~3,000 | 2 Python end-to-end tests | <1% |
| `frontend/src/**/*.jsx` | ~3,500 | 0 frontend tests | 0% |
| **Total** | **~12,000** | **~20 tests** | **<3%** |

**Recommendation:**
- [ ] Add pytest discovery path: `pytest tests/` (currently just `pytest` which finds nothing)
- [ ] Add unit tests for `device.py`, `memory_manager.py`, `gpu_ops.py`
- [ ] Add API contract tests using `httpx` against the running FastAPI app
- [ ] Add WebSocket telemetry schema validation tests
- [ ] Configure `pytest-cov` and set a coverage target (e.g., 30%)

### 6.5 Performance Considerations

#### 6.5.1 NumPy BLAS Underutilization

All GPU operations use NumPy's `np.matmul`, which delegates to OpenBLAS. The `WarpScheduler` dispatches work via `ThreadPoolExecutor` (2-4 threads), but OpenBLAS itself may use internal threading, leading to oversubscription.

**Current settings (from `backend/main.py:6-15`):**
```python
_os.environ.setdefault("OPENBLAS_NUM_THREADS", str(os.cpu_count() or 4))
_os.environ.setdefault("OMP_NUM_THREADS", str(os.cpu_count() or 4))
```

On a 4-core machine with `WarpScheduler(num_compute_units=2)` and OpenBLAS threads=4, each warp's `np.matmul` call may spawn 4 threads, resulting in up to 8 active threads for 2 warps on a 4-core system. This causes context switching overhead.

**Recommendation:**
- [ ] Set `OPENBLAS_NUM_THREADS=1` when dispatching through `WarpScheduler` (warp-level parallelism replaces BLAS threading)
- [ ] Or align BLAS threads with `compute_units` to avoid oversubscription

#### 6.5.2 Background Warp Heartbeat

The `background_warp_heartbeat()` task runs a continuous `matmul` loop at 1-second intervals to generate warp telemetry for the dashboard. This consumes CPU cycles even when no user workload is active.

**Recommendation:**
- [ ] Reduce heartbeat rate when no WebSocket clients are connected
- [ ] Stop heartbeat entirely when no dashboard is open

### 6.6 Code Quality & Maintainability

#### 6.6.1 `backend/main.py` Size

At **1,032 lines**, `main.py` is too large for a single file. It contains:
- App setup and configuration
- Global state declarations
- Background task definitions
- 25+ REST endpoints
- 2 WebSocket endpoints
- Static file serving logic
- CUDA shim sys.path manipulation

**Recommendation:**
- [ ] Split into modules: `app.py` (setup), `routes/` (endpoint groups), `tasks.py` (background tasks)
- [ ] Move sys.path manipulation and CUDA shim import to a separate `shim_loader.py`

#### 6.6.2 Error Handling Inconsistency

Error handling patterns vary across the codebase:
- Some endpoints use `try/except` with generic `{"error": str(e)}` responses
- Some let FastAPI's default exception handler return HTML
- Some catch all exceptions silently (e.g., `telemetry_loop`)
- The `SynthGPU` class has multiple telemetry accessors: `get_telemetry()`, `device_info()`, `get_stats()`

**Recommendation:**
- [ ] Add middleware for consistent error response formatting
- [ ] Consolidate telemetry access into a single method
- [ ] Log exceptions with structured logging (not just `print()`)

#### 6.6.3 Hardcoded Ports and URLs

Several URLs are hardcoded:
- `http://localhost:8000` — in frontend components (RAMMonitor.jsx, CudaShimStatus.jsx)
- `http://localhost:11434` — Ollama URL (in inference_proxy.py)
- `http://localhost:1234` — LM Studio URL (in inference_proxy.py)
- `http://localhost:8000` — in WebSocket connection strings

The frontend `CudaShimStatus.jsx` uses `import.meta.env.VITE_API_URL || "http://localhost:8000"` which is correct for Vite, but `RAMMonitor.jsx` hardcodes the URL directly, bypassing the environment variable.

**Recommendation:**
- [ ] Move all service URLs to environment variables
- [ ] Fix `RAMMonitor.jsx` to use the same API base URL pattern as other components
- [ ] Add `VITE_API_URL` to Docker Compose environment

#### 6.6.4 Import Side Effects in `bridge_api.py`

`cuda_shim/kernels/bridge_api.py` has module-level import side effects:
```python
from synthgpu.core.warp_scheduler import WarpScheduler  # May fail
...
_scheduler = WarpScheduler()  # Created at import time
_memory_manager = VirtualMemoryManager()  # Created at import time
```

This means importing any function from `bridge_api` creates a `WarpScheduler` and `VirtualMemoryManager` as a side effect. This complicates testing and can cause issues when importing the module in multiple contexts.

**Recommendation:**
- [ ] Use lazy initialization: create `_scheduler` and `_memory_manager` on first access
- [ ] Or make them function-level: `get_scheduler()` creates if not exists

### 6.7 Documentation Gaps

| Document | Status | Notes |
|----------|--------|-------|
| **README.md** | Exists | Brief overview, no architecture, no API docs |
| **OLLAMA_INTEGRATION.md** | Exists | Good Ollama setup guide |
| **API documentation** | Missing | No OpenAPI/Swagger customization beyond FastAPI defaults |
| **Architecture documentation** | Missing | No system design docs, no data flow diagrams |
| **Developer setup guide** | Missing | No instructions for local development without Docker |
| **Contribution guidelines** | Missing | No PR template, no coding standards doc |
| **Environment variable reference** | Missing | No centralized list of configurable env vars |
| **Test documentation** | Missing | No test running instructions |

---

## 7. Actionable Next Steps (Priority Order)

### Immediate (Week 1)

| Priority | Action | Effort | Impact |
|----------|--------|--------|--------|
| P0 | Delete root-level `synthgpu/` directory or move to `archive/` | 15 min | Eliminates package collision permanently |
| P0 | Add `backend/venv/`, `backend/.venv/`, `synthgpu/` to `.dockerignore` | 5 min | Reduces build context, prevents rebuild issues |
| P1 | Fix `RAMMonitor.jsx` to use `VITE_API_URL` instead of hardcoded URL | 10 min | Consistent API URL handling |
| P1 | Fix `pytest` discovery in CI workflow (add `tests/` path) | 5 min | CI actually runs tests |
| P1 | Create `.env.example` with all documented env vars | 30 min | Centralized configuration reference |

### Short-term (Week 2-3)

| Priority | Action | Effort | Impact |
|----------|--------|--------|--------|
| P1 | Add unit tests for `memory_manager.py` (allocation, free, mmap read/write) | 4h | Core memory logic coverage |
| P1 | Add unit tests for `device.py` (matmul, linear, attention, inference) | 4h | Core compute logic coverage |
| P1 | Split `backend/main.py` into modules (`routes/`, `tasks.py`) | 3h | Improves maintainability |
| P2 | Consolidate duplicate `requirements.txt` files | 30 min | Cleaner dependency management |
| P2 | Normalize version strings to single source | 1h | Version consistency |
| P2 | Set `OPENBLAS_NUM_THREADS=1` for warp-dispatched operations | 2h | Prevents thread oversubscription |

### Medium-term (Month 1-2)

| Priority | Action | Effort | Impact |
|----------|--------|--------|--------|
| P2 | Add SQLite persistence for session history and telemetry | 2d | Data survives restarts |
| P2 | Add Docker build step to CI/CD pipeline | 1d | Verified container builds |
| P2 | Add API integration tests using pytest + httpx | 2d | Catches endpoint regressions |
| P2 | Add Pydantic request/response models for all endpoints | 1d | Input validation + API docs |
| P3 | Implement lazy initialization in `bridge_api.py` | 1h | Cleaner import side effects |
| P3 | Add frontend component tests (Vitest + React Testing Library) | 2d | UI reliability |

### Long-term (Month 3+)

| Priority | Action | Effort | Impact |
|----------|--------|--------|--------|
| P3 | Implement real CUDA shim library loading (dlopen `libsynthgpu_cuda.so`) | 3d | Actual CUDA ABI interception |
| P3 | Add authentication middleware for production deployments | 2d | Security for exposed instances |
| P3 | Background warp heartbeat optimization (stop when no WS clients) | 1d | CPU resource savings |
| P3 | Kubernetes readiness — Helm chart, health probes, scaling | 3d | Production deployment path |
| P4 | Structured logging (replace `print()` with `logging`) | 1d | Production observability |
| P4 | OpenAPI/Swagger customization with operation IDs and tags | 2h | Developer-friendly API docs |

---

## Appendix A: File Inventory Summary

| Directory | Files | Total Size | Purpose |
|-----------|-------|-----------|---------|
| `backend/` | 14 | 137 KB | FastAPI application, device engine, inference proxy |
| `cuda_shim/` | 50 | 177 KB | CUDA ABI shim — C interceptors + Python bridge |
| `vulkan_icd/` | 27 | 141 KB | Vulkan Installable Client Driver |
| `frontend/` | 22 | 229 KB | React SPA dashboard |
| `synthgpu/` | 7 | 29 KB | **Legacy v0.1.0-mvp — should be deleted** |
| `tests/` | 2 | 15 KB | Test suites |
| `benchmarks/` | 1 | 10 KB | Standalone benchmark suite |
| `demos/` | 1 | 13 KB | Investor demo script |
| `.github/` | 1 | 3 KB | CI/CD workflow |
| (root) | 18 | 137 KB | Docker, compose, scripts, docs, configs |
| **Total** | **~156** | **~890 KB** | |

## Appendix B: API Endpoint Map (REST)

| Method | Path | Source | Purpose |
|--------|------|--------|---------|
| GET | `/` | main.py:1018 | Serve frontend SPA |
| GET | `/{path}` | main.py:1023 | SPA fallback routing |
| GET | `/api/device/status` | main.py:454 | Health check (used by Docker HEALTHCHECK) |
| GET | `/api/device/info` | main.py:384 | Full device telemetry |
| GET | `/api/system/ram` | main.py:464 | RAM + swap status with tiered warnings |
| GET | `/api/system/memory` | main.py:978 | System memory stats for inference feasibility |
| GET | `/api/cuda_shim/status` | main.py:901 | CUDA shim availability + warp/kernel stats |
| GET | `/api/vulkan/status` | main.py:949 | Vulkan ICD status + dispatch telemetry |
| POST | `/api/vulkan/record_dispatch` | main.py:990 | Record Vulkan dispatch (called by ICD) |
| GET | `/api/health/demo_ready` | main.py:512 | Demo readiness check (Ollama, RAM, inference) |
| GET | `/api/debug/telemetry` | main.py:572 | Nested telemetry debug snapshot |
| POST | `/api/benchmark/run` | main.py:592 | Run benchmark suite |
| POST | `/api/generate/tokens` | main.py:621 | Token generation demo |
| POST | `/api/model/upload` | main.py:671 | Upload ONNX model |
| POST | `/api/model/{id}/run` | main.py:785 | Run uploaded ONNX model |
| POST | `/api/inference/preflight` | main.py:402 | RAM/model fit check before inference |
| GET | `/api/economics` | main.py:817 | Cost comparison data |
| GET | `/api/inference/status` | proxy:550 | LLM backend + current inference status |
| GET | `/api/inference/models` | proxy:568 | List available LLM models |
| POST | `/api/inference/connect` | proxy:575 | Connect to Ollama/LM Studio |
| POST | `/api/inference/disconnect` | proxy:692 | Disconnect LLM backend |
| POST | `/api/inference/run` | proxy:698 | Start streaming inference |
| GET | `/api/inference/memory` | proxy:800 | Virtual VRAM + system RAM detailed stats |
| POST | `/api/inference/pull` | proxy:822 | Pull a new Ollama model |
| POST | `/api/generate` | proxy:848 | Ollama-compatible generate proxy |
| POST | `/api/chat` | proxy:909 | Ollama-compatible chat proxy |
| GET | `/api/tags` | proxy:969 | List Ollama model tags |
| POST | `/v1/chat/completions` | proxy:932 | OpenAI-compatible chat completions |
| GET | `/v1/models` | proxy:951 | OpenAI-compatible model listing |
| GET | `/synthgpu/status` | proxy:977 | SynthGPU device status |
| GET | `/synthgpu/memory` | proxy:988 | SynthGPU memory info |
| GET | `/health` | proxy:1009 | Simple health check |

## Appendix C: WebSocket Endpoints

| Path | Source | Purpose | Frequency |
|------|--------|---------|-----------|
| `/ws/telemetry` | main.py:249 | Device telemetry stream | 200ms |
| `/ws/tokens` | main.py:266 | Token streaming during LLM inference | Per-token |
