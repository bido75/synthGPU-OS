"""
SynthGPU FastAPI Backend v0.2 — Enhanced with Ollama/LM Studio proxy.
"""

import asyncio
import json
import os
import time
import tempfile
import threading
import uuid
from pathlib import Path
from typing import Optional, Set

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from synthgpu.device import SynthGPU
from synthgpu.onnx_provider import SynthGPUExecutionProvider, ONNX_AVAILABLE
from benchmarks.runner import BenchmarkRunner

try:
    from synthgpu.inference_proxy import (
        SynthGPUTelemetryEngine,
        InferenceProxyRouter,
        detect_backend as detect_llm_backend,
    )
    INFERENCE_PROXY_AVAILABLE = True
except ImportError as e:
    print(f"[SynthGPU] Warning: inference proxy not available ({e})")
    INFERENCE_PROXY_AVAILABLE = False

# ── App Setup ──────────────────────────────────────────────────
app = FastAPI(title="SynthGPU API", version="0.2.0-beta")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global State ───────────────────────────────────────────────
gpu = SynthGPU(vram_mb=4096)
onnx_provider = SynthGPUExecutionProvider(gpu)

_telemetry_connections: Set[WebSocket] = set()
_token_connections: Set[WebSocket] = set()
_benchmark_progress: Optional[dict] = None
_uploaded_models: dict = {}
_startup_time = time.time()

# ── Inference Proxy ────────────────────────────────────────────
inference_engine: Optional[SynthGPUTelemetryEngine] = None
proxy_router_obj: Optional[InferenceProxyRouter] = None

if INFERENCE_PROXY_AVAILABLE:
    inference_engine = SynthGPUTelemetryEngine(gpu_device=gpu)
    proxy_router_obj = InferenceProxyRouter(inference_engine)
    app.include_router(proxy_router_obj.router, prefix="")


# ── WebSocket Broadcast ────────────────────────────────────────
async def broadcast(connections: Set[WebSocket], message: dict):
    dead = set()
    for ws in connections:
        try:
            await ws.send_json(message)
        except Exception:
            dead.add(ws)
    connections -= dead


# Wire up token broadcast to inference proxy
if INFERENCE_PROXY_AVAILABLE and proxy_router_obj is not None:
    proxy_router_obj.set_broadcast(_token_connections, broadcast)


# ── Background Tasks ───────────────────────────────────────────
async def telemetry_loop():
    while True:
        await asyncio.sleep(0.2)
        if not _telemetry_connections:
            continue
        try:
            telemetry = gpu.get_telemetry()
            msg = {
                "type": "telemetry",
                "timestamp": time.time(),
                "device": {
                    "name": telemetry["device"],
                    "version": telemetry["version"],
                    "platform": telemetry.get("platform", ""),
                    "os": telemetry.get("os", ""),
                    "uptime_seconds": telemetry["uptime_seconds"],
                    "ops_executed": telemetry["ops_executed"],
                },
                "scheduler": telemetry["scheduler"],
                "memory": telemetry["memory"],
                "benchmark_progress": _benchmark_progress,
            }
            # Extend with inference telemetry if proxy is active
            if inference_engine:
                msg["inference"] = inference_engine.get_inference_telemetry()
                mem_ext = inference_engine.get_memory_extension()
                msg["memory"].update(mem_ext)

            await broadcast(_telemetry_connections, msg)
        except Exception:
            pass


async def auto_detect_llm_backend():
    if not INFERENCE_PROXY_AVAILABLE:
        return
    found = await detect_llm_backend()
    if not found:
        print("[SynthGPU] No LLM backend detected — connect manually in dashboard")
        print("[SynthGPU] Tip: Start Ollama with `ollama serve`, then reload")


async def periodic_backend_health_check():
    if not INFERENCE_PROXY_AVAILABLE:
        return
    while True:
        await asyncio.sleep(30)
        if inference_engine and not inference_engine.active_session:
            await detect_llm_backend()


@app.on_event("startup")
async def startup_event():
    asyncio.create_task(telemetry_loop())
    asyncio.create_task(auto_detect_llm_backend())
    asyncio.create_task(periodic_backend_health_check())
    print("[SynthGPU] FastAPI server started. Telemetry + inference proxy active.")


# ── WebSocket Endpoints ────────────────────────────────────────
@app.websocket("/ws/telemetry")
async def ws_telemetry(websocket: WebSocket):
    await websocket.accept()
    _telemetry_connections.add(websocket)
    try:
        initial = {"type": "connected", "device": gpu.get_telemetry()}
        if inference_engine:
            initial["inference"] = inference_engine.get_inference_telemetry()
        await websocket.send_json(initial)
        while True:
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        pass
    finally:
        _telemetry_connections.discard(websocket)


@app.websocket("/ws/tokens")
async def ws_tokens(websocket: WebSocket):
    await websocket.accept()
    _token_connections.add(websocket)
    try:
        while True:
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        pass
    finally:
        _token_connections.discard(websocket)


# ── REST Endpoints (existing) ──────────────────────────────────
@app.get("/api/device/info")
async def device_info():
    telem = gpu.get_telemetry()
    if inference_engine:
        telem["inference"] = inference_engine.get_inference_telemetry()
    return telem


@app.get("/api/device/status")
async def device_status():
    return {
        "status": "busy" if _benchmark_progress else "ready",
        "uptime_seconds": round(time.time() - _startup_time, 1),
        "version": gpu.VERSION,
        "inference_backend": (inference_engine.get_inference_telemetry().get("backend")
                              if inference_engine else None),
    }


class BenchmarkRequest(BaseModel):
    benchmark: str = "all"


@app.post("/api/benchmark/run")
async def run_benchmark(req: BenchmarkRequest):
    global _benchmark_progress
    if _benchmark_progress:
        raise HTTPException(status_code=409, detail="Benchmark already running")
    results = {}

    def on_progress(data: dict):
        global _benchmark_progress
        _benchmark_progress = data

    def _run():
        global _benchmark_progress
        runner = BenchmarkRunner(gpu, progress_callback=on_progress)
        try:
            if req.benchmark == "gemm":
                results['gemm'] = [r.__dict__ for r in runner.run_gemm()]
            elif req.benchmark == "mlp":
                results['mlp'] = [r.__dict__ for r in runner.run_mlp()]
            elif req.benchmark == "transformer":
                results['transformer'] = [r.__dict__ for r in runner.run_transformer()]
            elif req.benchmark == "token_gen":
                results['token_gen'] = list(runner.run_token_generation(num_tokens=10))
            else:
                results.update(runner.run_all())
        finally:
            _benchmark_progress = None

    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _run)
    return {"status": "complete", "results": results}


class TokenGenRequest(BaseModel):
    num_tokens: int = 20
    d_model: int = 256
    num_layers: int = 2


@app.post("/api/generate/tokens")
async def generate_tokens(req: TokenGenRequest):
    d = req.d_model
    n_h = max(1, d // 64)
    s = 0.02
    np.random.seed(42)

    def make_layer():
        return {
            'num_heads': n_h,
            'Wq': np.random.randn(d, d).astype(np.float32) * s,
            'Wk': np.random.randn(d, d).astype(np.float32) * s,
            'Wv': np.random.randn(d, d).astype(np.float32) * s,
            'Wo': np.random.randn(d, d).astype(np.float32) * s,
            'W1': np.random.randn(d * 4, d).astype(np.float32) * s,
            'b1': np.zeros(d * 4, dtype=np.float32),
            'W2': np.random.randn(d, d * 4).astype(np.float32) * s,
            'b2': np.zeros(d, dtype=np.float32),
            'gamma1': np.ones(d, dtype=np.float32),
            'beta1': np.zeros(d, dtype=np.float32),
            'gamma2': np.ones(d, dtype=np.float32),
            'beta2': np.zeros(d, dtype=np.float32),
        }

    model_config = {
        'num_layers': req.num_layers,
        'layers': [make_layer() for _ in range(req.num_layers)],
        'lm_head': np.random.randn(32000, d).astype(np.float32) * s,
    }

    async def stream_tokens():
        runner = BenchmarkRunner(gpu)
        for token_data in runner.run_token_generation(num_tokens=req.num_tokens):
            msg = {"type": "token", **token_data, "total_tokens": req.num_tokens}
            await broadcast(_token_connections, msg)

    asyncio.create_task(stream_tokens())
    return {"status": "streaming", "num_tokens": req.num_tokens}


@app.post("/api/model/upload")
async def upload_model(file: UploadFile = File(...)):
    if not file.filename.endswith('.onnx'):
        raise HTTPException(status_code=400, detail="Only .onnx files are supported")
    content = await file.read()
    if len(content) > 500 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large (max 500MB)")
    model_id = str(uuid.uuid4())[:8]
    model_dir = Path(tempfile.gettempdir()) / "synthgpu_models"
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / f"{model_id}.onnx"
    model_path.write_bytes(content)
    try:
        info = onnx_provider.get_model_info(str(model_path))
    except Exception as e:
        model_path.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=f"Invalid ONNX model: {e}")
    size_mb = round(len(content) / 1e6, 2)
    _uploaded_models[model_id] = {
        "path": str(model_path), "filename": file.filename,
        "size_mb": size_mb, "info": info,
    }
    return {"model_id": model_id, "filename": file.filename,
            "size_mb": size_mb, "inputs": info["inputs"], "outputs": info["outputs"]}


class ModelRunRequest(BaseModel):
    input_shape: list
    dtype: str = "float32"


@app.post("/api/model/{model_id}/run")
async def run_model(model_id: str, req: ModelRunRequest):
    if model_id not in _uploaded_models:
        raise HTTPException(status_code=404, detail="Model not found")
    model_info = _uploaded_models[model_id]
    input_name = model_info["info"]["inputs"][0]["name"]
    dtype = np.float32 if req.dtype == "float32" else np.float16
    input_data = np.random.randn(*req.input_shape).astype(dtype)

    def _run():
        return onnx_provider.run_model(model_info["path"], {input_name: input_data})

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, _run)
    result.pop("outputs", None)
    return result


@app.get("/api/economics")
async def economics():
    return {
        "comparisons": [
            {"name": "NVIDIA H100 (AWS p4d)", "cost_per_hour": 32.77,
             "monthly_cost": 23594, "wait_time": "3-12 months",
             "hardware_required": True, "color": "#7c3aed"},
            {"name": "NVIDIA A100 (AWS p3)", "cost_per_hour": 12.24,
             "monthly_cost": 8813, "wait_time": "2-6 weeks",
             "hardware_required": True, "color": "#6d28d9"},
            {"name": "SynthGPU (CPU-only)", "cost_per_hour": 5.44,
             "monthly_cost": 3917, "wait_time": "Instant",
             "hardware_required": False, "color": "#00d4ff"},
        ],
        "savings_pct": 83,
        "monthly_savings": 19677,
    }


# ── Static Frontend Serving ────────────────────────────────────
frontend_dist = Path(__file__).parent.parent / "frontend" / "dist"
if frontend_dist.exists():
    app.mount("/assets", StaticFiles(directory=str(frontend_dist / "assets")), name="assets")

    @app.get("/")
    async def serve_frontend():
        return FileResponse(str(frontend_dist / "index.html"))

    @app.get("/{path:path}")
    async def serve_spa(path: str):
        file_path = frontend_dist / path
        if file_path.exists() and file_path.is_file():
            return FileResponse(str(file_path))
        return FileResponse(str(frontend_dist / "index.html"))
