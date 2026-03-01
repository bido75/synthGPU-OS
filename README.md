# SynthGPU Beta v0.2 — Virtual GPU Accelerator

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
