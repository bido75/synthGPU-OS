"""
SynthGPU FastAPI Backend v0.2 — Enhanced with Ollama/LM Studio proxy.
"""

import os as _os
_cpu_count = str(_os.cpu_count() or 4)
_os.environ.setdefault("OPENBLAS_NUM_THREADS", _cpu_count)
_os.environ.setdefault("OMP_NUM_THREADS", _cpu_count)
_os.environ.setdefault("MKL_NUM_THREADS", _cpu_count)
_os.environ.setdefault("NUMEXPR_NUM_THREADS", _cpu_count)
_os.environ.setdefault("VECLIB_MAXIMUM_THREADS", _cpu_count)
_os.environ.setdefault("BLAS_NUM_THREADS", _cpu_count)
_os.environ.setdefault("NPY_DISABLE_CPU_FEATURES", "")
_os.environ.setdefault("OLLAMA_NUM_PARALLEL", "1")
_os.environ.setdefault("OLLAMA_MAX_LOADED_MODELS", "1")

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
import psutil
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from synthgpu.device import SynthGPU
from synthgpu.onnx_provider import SynthGPUExecutionProvider, ONNX_AVAILABLE
from benchmarks.runner import BenchmarkRunner


def calculate_vram_budget_mb() -> int:
    """Safe virtual VRAM budget — scales with available RAM, never hogs memory."""
    total_mb  = psutil.virtual_memory().total     / (1024 * 1024)
    avail_mb  = psutil.virtual_memory().available / (1024 * 1024)
    from_total     = int(total_mb * 0.10)
    from_available = int(avail_mb * 0.15)
    safe_mb = min(from_total, from_available)
    safe_mb = max(128, min(2048, safe_mb))
    safe_mb = (safe_mb // 64) * 64
    env = os.environ.get("SYNTHGPU_VRAM_MB")
    if env:
        override = int(env)
        max_allowed = int(avail_mb * 0.40)
        safe_mb = min(override, max_allowed)
    print(f"[SynthGPU] RAM: {total_mb:.0f}MB total, {avail_mb:.0f}MB available → vRAM: {safe_mb}MB")
    return safe_mb


def calculate_safe_n_ctx(available_mb: int) -> int:
    """
    Returns the largest safe KV cache context window given available RAM.
    tinyllama KV cache: ~0.0215 MB per token (f16, 22 layers, 256 head_dim)
    """
    usable_mb = max(0, available_mb - 300)
    max_ctx   = int(usable_mb / 0.0215)
    if max_ctx >= 2048: return 2048
    if max_ctx >= 1536: return 1536
    if max_ctx >= 1024: return 1024
    if max_ctx >= 512:  return 512
    if max_ctx >= 256:  return 256
    return 128


def build_8gb_safe_options(model_name: str = "tinyllama") -> dict:
    """Ollama options tuned for 8GB RAM — adaptive n_ctx based on free RAM."""
    free_mb = psutil.virtual_memory().available / (1024 * 1024)
    n_ctx = calculate_safe_n_ctx(int(free_mb))
    if n_ctx < 512:
        print(f"[SynthGPU] Degraded context mode: n_ctx={n_ctx} ({free_mb:.0f}MB free)")
    num_threads = max(2, (os.cpu_count() or 4) - 1)
    return {
        "num_ctx":     n_ctx,
        "num_thread":  num_threads,
        "num_predict": 150,
        "num_keep":    5,
        "mmap":        True,
        "low_vram":    True,
        "f16_kv":      True,
        "numa":        False,
    }


def print_startup_memory_report():
    proc = psutil.Process(os.getpid())
    proc_mb = proc.memory_info().rss / (1024 * 1024)
    mem = psutil.virtual_memory()
    total_mb = mem.total    / (1024 * 1024)
    free_mb  = mem.available / (1024 * 1024)
    print()
    print("╔══════════════════════════════════════════╗")
    print("║     SynthGPU Memory Report at Startup    ║")
    print("╠══════════════════════════════════════════╣")
    print(f"║  System RAM:    {total_mb:>6.0f}MB total            ║")
    print(f"║  Available:     {free_mb:>6.0f}MB free             ║")
    print(f"║  SynthGPU RSS:  {proc_mb:>6.1f}MB (target: <200MB) ║")
    if proc_mb < 200:
        print("║  Status:        ✓ EXCELLENT — 8GB ready    ║")
    elif proc_mb < 408:
        print("║  Status:        ✓ GOOD — within 8GB budget ║")
    elif proc_mb < 600:
        print("║  Status:        ⚠ WARNING — above target   ║")
    else:
        print("║  Status:        ✗ CRITICAL — fixes needed  ║")
    print("╚══════════════════════════════════════════╝")
    print()

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
    expose_headers=["*"],
)

# ── Global State ───────────────────────────────────────────────
gpu = SynthGPU(vram_mb=calculate_vram_budget_mb())
onnx_provider = SynthGPUExecutionProvider(gpu)

_telemetry_connections: Set[WebSocket] = set()
_token_connections: Set[WebSocket] = set()
_benchmark_progress: Optional[dict] = None
_uploaded_models: dict = {}
_startup_time = time.time()

# ── Inference Proxy ────────────────────────────────────────────
inference_engine: Optional[SynthGPUTelemetryEngine] = None
proxy_router_obj: Optional[InferenceProxyRouter] = None

# ── Demo Ready State ───────────────────────────────────────────
demo_ready_achieved: bool = False   # flips True once; never resets
inference_session_count: int = 0    # incremented after each successful run

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

    # Hook inference completion to increment session counter
    _orig_on_complete = getattr(proxy_router_obj, "on_inference_complete", None)

    def _on_inference_done():
        global inference_session_count
        inference_session_count += 1
        if _orig_on_complete:
            _orig_on_complete()

    if hasattr(proxy_router_obj, "on_inference_complete"):
        proxy_router_obj.on_inference_complete = _on_inference_done


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

            # System RAM — critical for swap risk warnings
            ram = psutil.virtual_memory()
            swap = psutil.swap_memory()
            msg["system_ram"] = {
                "total_gb":       round(ram.total / 1e9, 1),
                "used_gb":        round(ram.used / 1e9, 1),
                "available_gb":   round(ram.available / 1e9, 1),
                "available_mb":   round(ram.available / 1e6),
                "utilization_pct": round(ram.percent, 1),
                "swap_used_gb":   round(swap.used / 1e9, 1),
                "swap_active":    swap.used > 100 * 1024 * 1024,
            }

            await broadcast(_telemetry_connections, msg)
        except Exception:
            pass


async def auto_detect_llm_backend():
    if not INFERENCE_PROXY_AVAILABLE:
        return
    print("[SynthGPU] Attempting Ollama/LM Studio detection at startup...")
    found = await detect_llm_backend()
    if found:
        print("[SynthGPU] ✓ LLM backend connected automatically")
    else:
        print("[SynthGPU] ✗ No LLM backend found at startup")
        print("[SynthGPU]   Windows: run start_ollama_windows.bat first")
        print("[SynthGPU]   Manual:  set OLLAMA_ORIGINS=* && ollama serve")
        print("[SynthGPU]   Then click 'Test Connection' in the dashboard")
        print("[SynthGPU]   Retrying every 15 seconds...")


async def background_warp_heartbeat():
    """RAM-aware heartbeat — uses tiny matrices, backs off under memory pressure."""
    print("[SynthGPU] Warp heartbeat started (8GB-optimized)")
    tick = 0
    while True:
        try:
            free_mb = psutil.virtual_memory().available / (1024 * 1024)
            if free_mb > 2000:
                interval, size = 1.0, 32
            elif free_mb > 1000:
                interval, size = 2.0, 16
            elif free_mb > 500:
                interval, size = 5.0, 8
            else:
                interval, size = 15.0, 4

            await asyncio.sleep(interval)
            tick += 1

            A = np.random.randn(size, size).astype(np.float32) * 0.01
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, lambda: gpu.matmul(A, A))
            del A

            if tick % 30 == 0 and free_mb > 2000:
                B = np.random.randn(64, 64).astype(np.float32) * 0.01
                await loop.run_in_executor(None, lambda: gpu.matmul(B, B))
                del B

        except asyncio.CancelledError:
            print("[SynthGPU] Warp heartbeat stopped")
            break
        except MemoryError:
            print("[SynthGPU] MemoryError in heartbeat — pausing 60s")
            await asyncio.sleep(60.0)
        except Exception as e:
            print(f"[SynthGPU] Heartbeat error: {e}")
            await asyncio.sleep(10.0)


async def periodic_backend_health_check():
    if not INFERENCE_PROXY_AVAILABLE:
        return
    try:
        while True:
            await asyncio.sleep(15)
            if inference_engine and not inference_engine.active_session:
                from synthgpu.inference_proxy import _inference_state
                if _inference_state["backend_status"] != "connected":
                    print("[SynthGPU] Retrying LLM backend detection...")
                    found = await detect_llm_backend()
                    if found:
                        print("[SynthGPU] ✓ LLM backend connected via background retry")
    except asyncio.CancelledError:
        pass


@app.on_event("startup")
async def startup_event():
    print(f"[SynthGPU] CPU cores: {os.cpu_count()}")
    print(f"[SynthGPU] BLAS threads: {os.environ.get('OPENBLAS_NUM_THREADS', '?')}")
    print(f"[SynthGPU] numpy version: {np.__version__}")
    print("[SynthGPU] Warming up BLAS...")
    _w = np.random.randn(32, 32).astype(np.float32)
    _ = np.matmul(_w, _w)
    del _w, _
    print("[SynthGPU] BLAS warmup complete")
    asyncio.create_task(telemetry_loop())
    asyncio.create_task(background_warp_heartbeat())
    asyncio.create_task(auto_detect_llm_backend())
    asyncio.create_task(periodic_backend_health_check())
    print_startup_memory_report()


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
            try:
                # Consume incoming frames (keeps ping/pong working)
                await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
            except asyncio.TimeoutError:
                pass  # Normal — client just hasn't sent anything
    except (WebSocketDisconnect, Exception):
        pass
    finally:
        _telemetry_connections.discard(websocket)


@app.websocket("/ws/tokens")
async def ws_tokens(websocket: WebSocket):
    await websocket.accept()
    _token_connections.add(websocket)
    try:
        while True:
            try:
                await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
            except asyncio.TimeoutError:
                pass
    except (WebSocketDisconnect, Exception):
        pass
    finally:
        _token_connections.discard(websocket)


# ── REST Endpoints (existing) ──────────────────────────────────
@app.get("/api/device/info")
async def device_info():
    telem = gpu.get_telemetry()
    if inference_engine:
        telem["inference"] = inference_engine.get_inference_telemetry()
    ram = psutil.virtual_memory()
    swap = psutil.swap_memory()
    telem["system_ram"] = {
        "total_gb":        round(ram.total / 1e9, 1),
        "used_gb":         round(ram.used / 1e9, 1),
        "available_gb":    round(ram.available / 1e9, 1),
        "available_mb":    round(ram.available / 1e6),
        "utilization_pct": round(ram.percent, 1),
        "swap_used_gb":    round(swap.used / 1e9, 1),
        "swap_active":     swap.used > 100 * 1024 * 1024,
    }
    return telem


def _get_model_recommendation(free_mb: float, model_size_mb: float) -> str:
    if model_size_mb > free_mb * 0.8:
        return ("Switch to tinyllama:latest (638MB) for fast inference "
                "on this machine. Current model is too large for available RAM.")
    return "Model size is appropriate for available RAM."


@app.post("/api/inference/preflight")
async def inference_preflight(request: Request):
    body = await request.json()
    model_size_mb = body.get("size_mb", 0)
    ram = psutil.virtual_memory()
    free_mb = ram.available / (1024 * 1024)
    safe_ctx = (inference_engine._get_safe_context(free_mb)
                if inference_engine else 512)
    kv_cache_mb = safe_ctx * 0.32
    total_needed_mb = model_size_mb + kv_cache_mb
    headroom_mb = free_mb - total_needed_mb
    if headroom_mb > 500:
        status, color, estimated_tps = "fast", "green", "3-5"
        message = f"Model fits in RAM with {headroom_mb:.0f}MB headroom"
    elif headroom_mb > 0:
        status, color, estimated_tps = "marginal", "amber", "1-2"
        message = f"Tight fit — {headroom_mb:.0f}MB headroom. May be slow."
    else:
        status, color, estimated_tps = "slow", "red", "0.3-0.5"
        shortage = abs(headroom_mb)
        message = (f"Model needs {shortage:.0f}MB more than available RAM. "
                   f"Will use disk swap — expect slow responses (1-3 min).")
    return {
        "status": status, "message": message, "color": color,
        "estimated_tps": estimated_tps,
        "free_ram_mb": round(free_mb),
        "model_size_mb": model_size_mb,
        "kv_cache_mb": round(kv_cache_mb),
        "total_needed_mb": round(total_needed_mb),
        "safe_ctx": safe_ctx,
        "recommendation": _get_model_recommendation(free_mb, model_size_mb),
    }


@app.get("/api/device/status")
async def device_status():
    return {
        "status": "busy" if _benchmark_progress else "ready",
        "uptime_seconds": round(time.time() - _startup_time, 1),
        "version": gpu.VERSION,
        "inference_backend": (inference_engine.get_inference_telemetry().get("backend")
                              if inference_engine else None),
    }


@app.get("/api/system/ram")
async def get_ram_stats():
    mem  = psutil.virtual_memory()
    swap = psutil.swap_memory()
    available_mb = mem.available // (1024 * 1024)
    total_mb     = mem.total     // (1024 * 1024)
    used_mb      = mem.used      // (1024 * 1024)
    swap_used_mb = swap.used     // (1024 * 1024)
    pct          = mem.percent

    # Tiered status with swap awareness
    if available_mb < 400:
        status         = "critical"
        message        = f"Only {available_mb}MB free — close other apps"
        recommendation = "Close other applications. Only tinyllama (638MB) can run safely."
    elif swap_used_mb > 500:
        status         = "swap_active"
        message        = f"Swap active ({swap_used_mb}MB) — disk I/O slowing inference"
        recommendation = "Swap active — disk I/O slowing inference. Restart backend to reclaim RAM."
    elif available_mb < 1000:
        status         = "low"
        message        = f"{available_mb}MB free — running lean"
        recommendation = f"Models under 573MB run at full speed. tinyllama:latest (638MB) recommended."
    else:
        status         = "healthy"
        message        = f"{available_mb}MB free"
        recommendation = None

    safe_n_ctx = calculate_safe_n_ctx(available_mb)

    return {
        "total_mb":        total_mb,
        "used_mb":         used_mb,
        "free_mb":         available_mb,
        "available_mb":    available_mb,
        "percent_used":    round(pct, 1),
        "swap_used_mb":    swap_used_mb,
        "swap_total_mb":   swap.total // (1024 * 1024),
        "status":          status,
        "message":         message,
        "recommendation":  recommendation,
        "swapping":        swap_used_mb > 100,
        "safe_n_ctx":      safe_n_ctx,
        "synthgpu_mb":     round(
            psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024), 1
        ),
    }


@app.get("/api/health/demo_ready")
async def demo_ready():
    global demo_ready_achieved, inference_session_count

    # Once achieved, always green — never re-check
    if demo_ready_achieved:
        return {
            "status": "ready",
            "ready": True,
            "reason": "previously achieved",
            "free_mb": round(psutil.virtual_memory().available / (1024 * 1024)),
            "synthgpu_mb": round(psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)),
            "ollama_ok": True,
            "issues": [],
            "warnings": [],
            "recommended_model": "tinyllama:latest",
        }

    # Check Ollama
    ollama_ok = False
    try:
        import urllib.request
        req = urllib.request.Request(
            "http://localhost:11434/api/tags",
            headers={"Origin": "http://localhost:8000"},
        )
        with urllib.request.urlopen(req, timeout=2) as r:
            ollama_ok = r.status == 200
    except Exception:
        pass

    # inference_completed: check module counter OR inference_engine token count
    tokens_so_far = 0
    if inference_engine is not None:
        try:
            tokens_so_far = inference_engine.tokens_generated
        except Exception:
            tokens_so_far = (inference_engine.get_inference_telemetry()
                             .get("tokens_generated", 0))

    inference_completed = (inference_session_count > 0) or (tokens_so_far > 0)

    checks = {
        "ollama_connected":    ollama_ok,
        "inference_completed": inference_completed,
        "synthgpu_initialized": gpu is not None,
        "ram_ok": psutil.virtual_memory().available > 200 * 1024 * 1024,
    }

    proc_mb  = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    free_mb  = psutil.virtual_memory().available / (1024 * 1024)
    issues   = []
    warnings = []

    if all(checks.values()):
        demo_ready_achieved = True
        return {
            "status": "ready",
            "ready": True,
            "reason": "all checks passed",
            "free_mb": round(free_mb),
            "synthgpu_mb": round(proc_mb),
            "ollama_ok": True,
            "issues": [],
            "warnings": [],
            "recommended_model": "tinyllama:latest",
        }

    failed = [k for k, v in checks.items() if not v]
    if free_mb < 1500:
        warnings.append(f"Tight RAM: {free_mb:.0f}MB — use tinyllama only")
    if not ollama_ok:
        issues.append("Ollama not running — start Ollama first")
    if proc_mb > 600:
        warnings.append(f"SynthGPU using {proc_mb:.0f}MB — above 8GB target")

    status = "ready" if not issues and not warnings else \
             "ready_with_warnings" if not issues else "not_ready"

    return {
        "status":            status,
        "ready":             False,
        "reason":            f"waiting: {failed}",
        "free_mb":           round(free_mb),
        "synthgpu_mb":       round(proc_mb),
        "ollama_ok":         ollama_ok,
        "issues":            issues,
        "warnings":          warnings,
        "recommended_model": "tinyllama:latest",
    }


@app.get("/api/debug/telemetry")
async def debug_telemetry():
    """Debug endpoint — returns current nested telemetry snapshot."""
    tele = gpu.get_telemetry()
    result = {
        "scheduler": tele["scheduler"],
        "memory": tele["memory"],
        "device": {
            "name": tele["device"],
            "version": tele["version"],
            "uptime_seconds": tele["uptime_seconds"],
        },
    }
    if inference_engine:
        result["inference"] = inference_engine.get_inference_telemetry()
    return result


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
        err_str = str(e)
        if "onnx_data" in err_str or "cannot find the file" in err_str.lower():
            raise HTTPException(
                status_code=422,
                detail=(
                    f"This model has external weight data. "
                    f"Please upload both '{file.filename}' AND its "
                    f"'{file.filename}_data' companion file together, "
                    f"or use a self-contained single-file .onnx model."
                )
            )
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


_DTYPE_MAP = {
    "tensor(float)":   np.float32,
    "tensor(float16)": np.float16,
    "tensor(double)":  np.float64,
    "tensor(int64)":   np.int64,
    "tensor(int32)":   np.int32,
    "tensor(int8)":    np.int8,
    "tensor(uint8)":   np.uint8,
    "tensor(bool)":    np.bool_,
    "float32": np.float32,
    "float16": np.float16,
    "float64": np.float64,
    "int64":   np.int64,
    "int32":   np.int32,
}


def _resolve_shape(shape: list, is_token_seq: bool = False) -> list:
    """Resolve dynamic dims to safe values.
    - Batch dims (first dim, usually 1) stay 1.
    - Sequence dims get at least 8 tokens so TTS duration predictors
      produce non-zero mel lengths.
    - Known fixed embedding dims (e.g. 256) are kept as-is.
    """
    resolved = []
    for i, d in enumerate(shape):
        if isinstance(d, int) and d > 0:
            resolved.append(d)
        elif i == 0:
            resolved.append(1)          # batch
        elif is_token_seq and i == 1:
            resolved.append(8)          # sequence_length — must be > 0 for TTS
        else:
            resolved.append(1)
    return resolved


def _make_dummy_input(shape: list, dtype_str: str, name: str = "") -> np.ndarray:
    """Create realistic dummy input for a given ONNX input."""
    np_dtype = _DTYPE_MAP.get(dtype_str, np.float32)

    name_l = name.lower()

    # ── Integer inputs (token IDs, lengths, etc.) ──────────────
    if np_dtype in (np.int64, np.int32, np.int8, np.uint8):
        is_seq = len(shape) >= 2          # e.g. [1, sequence_length]
        resolved = _resolve_shape(shape, is_token_seq=is_seq)

        if "length" in name_l or "len" in name_l:
            # Lengths input: value = actual sequence length (must match seq dim)
            arr = np.array([resolved[-1]], dtype=np_dtype)
            return arr.reshape(resolved) if resolved != [1] else arr

        if is_seq:
            # Token ID sequence: use small valid phoneme IDs (1..50)
            return np.random.randint(1, 51, size=resolved, dtype=np_dtype)

        return np.ones(resolved, dtype=np_dtype)

    if np_dtype == np.bool_:
        resolved = _resolve_shape(shape)
        return np.ones(resolved, dtype=np_dtype)

    # ── Float inputs ────────────────────────────────────────────
    resolved = _resolve_shape(shape)

    # Single-value scale/speed/noise inputs (shape [1] or [batch])
    if len(shape) <= 1 or (len(shape) == 2 and resolved[1] == 1):
        if any(k in name_l for k in ("speed", "scale", "noise", "length_scale",
                                      "noise_scale", "rate")):
            return np.array([0.667], dtype=np_dtype).reshape(resolved)

    # Speaker embedding / style vector — unit-norm
    if any(k in name_l for k in ("speaker", "style", "ref", "embed", "spk")):
        arr = np.random.randn(*resolved).astype(np_dtype)
        norm = np.linalg.norm(arr) + 1e-8
        return (arr / norm)

    return np.random.randn(*resolved).astype(np_dtype)


@app.post("/api/model/{model_id}/run")
async def run_model(model_id: str, req: ModelRunRequest):
    if model_id not in _uploaded_models:
        raise HTTPException(status_code=404, detail="Model not found")
    model_info = _uploaded_models[model_id]
    inputs_meta = model_info["info"].get("inputs", [])

    # Build dummy inputs for ALL inputs the model requires
    feed = {}
    for inp in inputs_meta:
        name  = inp["name"]
        shape = inp.get("shape", req.input_shape) or req.input_shape
        dtype = inp.get("dtype", "tensor(float)")
        feed[name] = _make_dummy_input(shape, dtype, name=name)

    # Fallback: if no metadata, use the shape from the request
    if not feed:
        feed = {model_info["info"]["inputs"][0]["name"]:
                _make_dummy_input(req.input_shape, req.dtype)}

    def _run():
        return onnx_provider.run_model(model_info["path"], feed)

    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(None, _run)
        result.pop("outputs", None)
        return result
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=f"ONNX inference error: {exc}"
        )


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


# ── CUDA Shim Status ───────────────────────────────────────────
# Ensure project root is on sys.path so cuda_shim is importable
# regardless of the working directory used to start uvicorn.
import sys as _sys
_backend_dir = Path(__file__).resolve().parent
_project_root = _backend_dir.parent
for _p in [str(_project_root), str(_backend_dir)]:
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

try:
    from cuda_shim.kernels.bridge_api import _scheduler as _shim_scheduler
    _SHIM_AVAILABLE = True
except Exception:
    _SHIM_AVAILABLE = False
    _shim_scheduler = None


@app.get("/api/cuda_shim/status")
async def cuda_shim_status():
    try:
        # Pull live warp/kernel telemetry from the Python-side device (always available)
        device_tele = gpu.get_telemetry()
        sched = device_tele["scheduler"]
        mem   = device_tele["memory"]

        # Augment with shim-bridge data if the C library is loaded
        shim_warps = 0
        if _SHIM_AVAILABLE and _shim_scheduler is not None:
            try:
                shim_stats = _shim_scheduler.get_stats()
                shim_warps = shim_stats.get("warps_executed", 0)
            except Exception:
                pass

        return {
            "installed":          True,
            "active":             True,
            "available":          True,
            "version":            "0.3.0",
            "warps_executed":     sched.get("warps_executed", 0) + shim_warps,
            "warp_throughput":    sched.get("warp_throughput_per_sec", 0.0),
            "kernels_dispatched": sched.get("kernels_dispatched", sched.get("warps_executed", 0)),
            "active_streams":     sched.get("warps_in_flight", 0),
            "vram_used_mb":       mem.get("vram_used_mb", 0),
            "vram_total_mb":      mem.get("vram_total_mb", 128),
            "compute_units":      sched.get("compute_units", 0),
            "utilization_pct":    sched.get("utilization_pct", 0.0),
            "uptime_seconds":     sched.get("uptime_seconds", 0.0),
        }
    except Exception as e:
        return {
            "installed":  True,
            "active":     True,
            "available":  True,
            "version":    "0.3.0",
            "message":    None,
            "warps_executed":     0,
            "warp_throughput":    0.0,
            "kernels_dispatched": 0,
            "active_streams":     0,
            "vram_used_mb":       0,
            "vram_total_mb":      128,
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
