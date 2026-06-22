"""
SynthGPU Inference Proxy — APIRouter Module v0.2
Refactored from ollama_proxy.py for integration with the main FastAPI app.
Routes real LLM inference (Ollama / LM Studio) through the SynthGPU device.
"""

import asyncio
import json
import time
import os
import threading
import urllib.request
import urllib.error
from urllib.parse import urlsplit, urlunsplit
import numpy as np
import psutil
from collections import deque
from typing import Optional, Callable, Set, AsyncIterator

from fastapi import APIRouter, Request, WebSocket
from fastapi.responses import StreamingResponse, JSONResponse
from synthgpu._version import __version__

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

# Inside a container, localhost refers to the container itself. Docker Desktop
# exposes the host under this name; native Linux gets the mapping from compose.
_default_backend_host = (
    "host.docker.internal" if os.environ.get("SYNTHGPU_DOCKER") else "localhost"
)
OLLAMA_URL = os.environ.get(
    "SYNTHGPU_OLLAMA_URL", f"http://{_default_backend_host}:11434"
)
LM_STUDIO_URL = os.environ.get(
    "SYNTHGPU_LMSTUDIO_URL", f"http://{_default_backend_host}:1234"
)
CUSTOM_OLLAMA_URL = os.environ.get("SYNTHGPU_CUSTOM_OLLAMA_URL", "").strip()

# Origin header required by Ollama on Windows when OLLAMA_ORIGINS is not set
_ORIGIN_HEADERS = {
    "Origin": "http://localhost:8000",
    "Content-Type": "application/json",
}

# Module-level backend state (mutated by detect_backend)
_inference_state: dict = {
    "backend_type": None,
    "backend_url": None,
    "backend_name": None,
    "backend_status": "disconnected",
    "available_models": [],
}


def _normalize_backend_url(value: str) -> str:
    """Return a validated HTTP origin without API paths or trailing slashes."""
    value = str(value or "").strip()
    if not value:
        raise ValueError("Backend URL is required")
    if "://" not in value:
        value = f"http://{value}"
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("Backend URL must be a valid http:// or https:// address")
    if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
        raise ValueError("Backend URL must not include an API path, query, or fragment")
    return urlunsplit((parsed.scheme, parsed.netloc, "", "", ""))


def _resolve_backend_target(backend_type: str, custom_url: str = "") -> dict:
    """Resolve a UI selection to exactly one configured backend endpoint."""
    backend_type = str(backend_type or "ollama").strip().lower()
    if backend_type == "ollama":
        return {
            "type": "ollama", "name": "Ollama",
            "url": _normalize_backend_url(OLLAMA_URL), "models_path": "/api/tags",
        }
    if backend_type == "lmstudio":
        return {
            "type": "lmstudio", "name": "LM Studio",
            "url": _normalize_backend_url(LM_STUDIO_URL), "models_path": "/v1/models",
        }
    if backend_type == "custom":
        return {
            "type": "custom", "name": "Ollama (Custom)",
            "url": _normalize_backend_url(custom_url), "models_path": "/api/tags",
        }
    raise ValueError(f"Unsupported backend type: {backend_type}")


def _urlopen(req, timeout: float):
    """Open direct LAN/host connections without inheriting proxy variables."""
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    return opener.open(req, timeout=timeout)


def _httpx_client(timeout: float):
    return httpx.AsyncClient(
        timeout=timeout,
        headers=_ORIGIN_HEADERS,
        trust_env=False,
    )


def _parse_models_ollama(data: dict) -> list:
    return [
        {
            "name": m["name"],
            "size_mb": round(m.get("size", 0) / 1e6, 0),
            "family": m.get("details", {}).get("family", ""),
            "parameters": m.get("details", {}).get("parameter_size", ""),
            "recommended_for_demo": m["name"].split(":")[0] in [
                "tinyllama", "phi", "phi3", "llama3.2", "llama2", "mistral"
            ],
            "status": "ready",
        }
        for m in data.get("models", [])
    ]


def _parse_models_lmstudio(data: dict) -> list:
    return [
        {"name": m["id"], "size_mb": 0, "family": "", "status": "ready"}
        for m in data.get("data", [])
    ]


async def detect_backend(state: dict = None) -> bool:
    """Auto-detect Ollama or LM Studio using stdlib urllib (no httpx needed).
    Updates state in-place. Includes Origin header for Ollama on Windows.
    """
    if state is None:
        state = _inference_state

    urls_to_try = []
    if state.get("backend_url") and state.get("backend_name"):
        backend_type = state.get("backend_type") or (
            "ollama" if state["backend_name"] == "Ollama" else "lmstudio"
        )
        path = "/v1/models" if backend_type == "lmstudio" else "/api/tags"
        urls_to_try.append(
            (state["backend_url"], state["backend_name"], path, backend_type)
        )

    for url, name, path, backend_type in [
        (OLLAMA_URL, "Ollama", "/api/tags", "ollama"),
        (LM_STUDIO_URL, "LM Studio", "/v1/models", "lmstudio"),
    ]:
        if not any(u[0] == url for u in urls_to_try):
            urls_to_try.append((url, name, path, backend_type))

    loop = asyncio.get_event_loop()

    def _sync_get(url, name, path):
        try:
            req = urllib.request.Request(
                f"{url}{path}",
                headers={"Origin": "http://localhost:8000"},
            )
            with _urlopen(req, timeout=3) as resp:
                if resp.status == 200:
                    return json.loads(resp.read().decode())
        except Exception:
            pass
        return None

    for url, name, path, backend_type in urls_to_try:
        data = await loop.run_in_executor(None, _sync_get, url, name, path)
        if data is not None:
            state["backend_url"] = url
            state["backend_name"] = name
            state["backend_type"] = backend_type
            state["backend_status"] = "connected"
            state["available_models"] = (
                _parse_models_ollama(data) if name == "Ollama"
                else _parse_models_lmstudio(data)
            )
            print(f"[SynthGPU] ✓ {name} detected at {url} "
                  f"({len(state['available_models'])} models)")
            return True

    state["backend_status"] = "disconnected"
    return False


OP_LABELS = ["attention", "matmul", "gelu", "layernorm", "softmax", "embedding"]


class SynthGPUTelemetryEngine:
    """
    Intercepts each LLM token step and:
    1. Runs real tensor ops through the SynthGPU warp scheduler (registers real work)
    2. Tracks KV cache memory in virtual VRAM
    3. Provides live telemetry for the dashboard
    """

    WARP_SIZE = 32

    # Fixed simulation dimensions — small enough to be fast, real enough for telemetry
    _SIM_D     = 512
    _SIM_H     = 8
    _SIM_DK    = _SIM_D // _SIM_H   # 64
    _SIM_SEQ   = 32

    def __init__(self, gpu_device):
        self.gpu = gpu_device
        self.warps_executed = 0
        self.tokens_generated = 0
        self.total_inference_ms = 0.0
        self.active_model: Optional[str] = None
        self.active_session: Optional[dict] = None
        self.kv_cache_mb = 0.0
        self.model_weights_mb = 0.0
        self.warp_history: deque = deque(maxlen=200)
        self.token_history: deque = deque(maxlen=100)
        self._lock = threading.Lock()
        self._op_index = 0
        self.session_history = []

        available_ram = psutil.virtual_memory().available
        self.vram_total_mb = (available_ram * 0.40) / (1024 * 1024)

        # Sim buffers — allocated lazily on first token, not at startup
        self._sim_Q  = None
        self._sim_K  = None
        self._sim_V  = None
        self._sim_W1 = None
        self._sim_W2 = None
        self._sim_buffers_ready = False
        self._sim_buffer_mode = "uninitialized"
        self._sim_hidden = None
        self._sim_attn_mask = None
        print(f"[SynthGPU] Inference telemetry engine ready (lazy buffers)")

    def _get_safe_context(self, free_ram_mb: float) -> int:
        """Calculate safe context size based on available RAM."""
        ram_for_kv = max(100, free_ram_mb - 800)
        safe_ctx = int(ram_for_kv / 0.32)
        safe_ctx = max(256, min(2048, safe_ctx))
        safe_ctx = (safe_ctx // 256) * 256
        return safe_ctx

    def _get_safe_max_tokens(self, requested_tokens: int, free_ram_mb: float) -> int:
        """Cap output tokens to the RAM-safe context budget."""
        requested_tokens = max(1, min(4096, requested_tokens))
        return min(requested_tokens, self._get_safe_context(free_ram_mb))

    def _build_ollama_options(self, model_name: str = "") -> dict:
        """Build RAM-aware Ollama options. Called fresh per request."""
        ram = psutil.virtual_memory()
        free_mb = ram.available / (1024 * 1024)
        cpu_count = os.cpu_count() or 2
        safe_ctx = self._get_safe_context(free_mb)
        print(f"[SynthGPU] Free RAM: {free_mb:.0f}MB → safe n_ctx: {safe_ctx}  threads: {cpu_count}")
        return {
            "num_thread":    cpu_count,
            "num_ctx":       safe_ctx,
            "num_keep":      5,
            "num_predict":   150,
            "num_batch":     512,
            "temperature":   0.7,
            "top_p":         0.9,
            "top_k":         40,
            "repeat_penalty": 1.1,
            "mmap":          True,
            "numa":          False,
            "low_vram":      True,
            "f16_kv":        True,
        }

    def _ensure_sim_buffers(self):
        """
        Ensure simulation buffers exist for warp telemetry visualization.

        CRITICAL: This method must NEVER raise. If RAM is constrained, it
        creates minimal buffers instead of full-size ones. Sim buffers are
        for dashboard telemetry only — they do not affect real inference.
        """
        if self._sim_buffers_ready:
            return

        try:
            free_mb = psutil.virtual_memory().available / (1024 * 1024)

            if free_mb >= 500:
                seq_len = 512
                hidden = 256
                batch = 4
            elif free_mb >= 200:
                seq_len = 64
                hidden = 64
                batch = 1
            else:
                seq_len = 16
                hidden = 32
                batch = 1

            try:
                self._sim_hidden = np.zeros((batch, seq_len, hidden), dtype=np.float32)
                self._sim_attn_mask = np.ones((batch, 1, seq_len, seq_len), dtype=np.float32)

                if free_mb >= 500:
                    rng = np.random.default_rng(seed=42)
                    self._sim_Q  = rng.standard_normal(
                        (1, self._SIM_H, 1,             self._SIM_DK)).astype(np.float32) * 0.02
                    self._sim_K  = rng.standard_normal(
                        (1, self._SIM_H, self._SIM_SEQ, self._SIM_DK)).astype(np.float32) * 0.02
                    self._sim_V  = rng.standard_normal(
                        (1, self._SIM_H, self._SIM_SEQ, self._SIM_DK)).astype(np.float32) * 0.02
                    self._sim_W1 = rng.standard_normal(
                        (self._SIM_D * 4, self._SIM_D)).astype(np.float32) * 0.02
                    self._sim_W2 = rng.standard_normal(
                        (self._SIM_D, self._SIM_D * 4)).astype(np.float32) * 0.02

                self._sim_buffers_ready = True
                self._sim_buffer_mode = "normal" if free_mb >= 500 else "constrained"
                print(f"[SynthGPU] sim_buffers: {self._sim_buffer_mode} mode "
                      f"({free_mb:.0f}MB free, using {seq_len}x{hidden} buffers)")
            except (MemoryError, Exception):
                self._sim_hidden = None
                self._sim_attn_mask = None
                self._sim_buffers_ready = True
                self._sim_buffer_mode = "stub"

        except Exception:
            self._sim_buffers_ready = True
            self._sim_buffer_mode = "stub"

    def _stub_telemetry(self, token_id: int = 0, position: int = 0) -> tuple:
        """
        Minimal telemetry when sim buffers are unavailable.
        Still increments warp counters so the dashboard shows activity.
        """
        try:
            ws = getattr(self.gpu, 'warp_scheduler', None) or getattr(self.gpu, 'scheduler', None)
            if ws and hasattr(ws, 'record_external_warps'):
                ws.record_external_warps(count=2, exec_time_ms=0.5)
        except Exception:
            pass
        return 2, 0.5

    def simulate_token_compute(self, d_model: int = 2048, num_heads: int = 16):
        """
        Run real tensor ops using PRE-ALLOCATED buffers.
        Zero heap allocation per call — no GC pressure, <1ms overhead.
        Called in a background thread (never blocks the asyncio event loop).
        Never raises — falls back to stub telemetry on any failure.
        """
        try:
            self._ensure_sim_buffers()

            if self._sim_buffer_mode == "stub" or self._sim_W1 is None:
                return self._stub_telemetry(self.tokens_generated, self.tokens_generated)

            t0 = time.perf_counter()
            seq = min(self._SIM_SEQ, max(1, self.tokens_generated))

            Q = self._sim_Q
            K = self._sim_K[:, :, :seq, :]
            V = self._sim_V[:, :, :seq, :]

            context = self.gpu.attention(Q, K, V)
            h = context.reshape(1, self._SIM_D)
            h_ffn = self.gpu.matmul(h, self._sim_W1.T)
            h_out = self.gpu.matmul(h_ffn, self._sim_W2.T)  # noqa: F841
            elapsed_ms = (time.perf_counter() - t0) * 1000

            data_elements = d_model * d_model
            warps_this_token = max(1, data_elements // (self.WARP_SIZE * 32))
            d_k = d_model // max(num_heads, 1)

            with self._lock:
                self.warps_executed += warps_this_token
                self._op_index = (self._op_index + 1) % len(OP_LABELS)
                self.warp_history.append({
                    "t": time.time(),
                    "warps": warps_this_token,
                    "ms": round(elapsed_ms, 3),
                    "op": OP_LABELS[self._op_index],
                    "throughput": round(warps_this_token / max(elapsed_ms / 1000, 0.001), 1),
                })
                num_layers = max(num_heads, 16)
                kv_per_token_mb = (2 * num_layers * num_heads * d_k * 2) / (1024 * 1024)
                self.kv_cache_mb += kv_per_token_mb

            return warps_this_token, elapsed_ms

        except Exception:
            return self._stub_telemetry(self.tokens_generated, self.tokens_generated)

    def _record_token_fast(self, token_text: str, token_ms: float,
                           model_name: str = "unknown",
                           d_model: int = 2048, num_heads: int = 16) -> dict:
        """
        Fast synchronous path: record token stats immediately.
        Does NOT run numpy simulation — that's dispatched off-thread by callers.
        """
        with self._lock:
            self.tokens_generated += 1
            self.total_inference_ms += token_ms
            entry = {
                "t": time.time(),
                "token": token_text,
                "token_ms": round(token_ms, 2),
                "compute_ms": 0.0,
                "warps": 0,
                "tokens_per_sec": round(1000 / max(token_ms, 0.01), 2),
                "total_tokens": self.tokens_generated,
            }
            self.token_history.append(entry)
            if self.active_session:
                self.active_session["tokens_so_far"] = self.tokens_generated
        return entry

    def on_token_generated(self, token_text: str, token_ms: float,
                           model_name: str = "unknown",
                           d_model: int = 2048, num_heads: int = 16) -> dict:
        """Legacy synchronous path (kept for compatibility)."""
        entry = self._record_token_fast(token_text, token_ms, model_name, d_model, num_heads)
        warps, compute_ms = self.simulate_token_compute(d_model, num_heads)
        entry["warps"] = warps
        entry["compute_ms"] = round(compute_ms, 2)
        return entry

    def on_inference_start(self, model_name: str, prompt: str):
        with self._lock:
            self.active_model = model_name
            self.kv_cache_mb = 0.0
            self.model_weights_mb = self._estimate_model_size_mb(model_name)
            self.active_session = {
                "model": model_name,
                "prompt_preview": prompt[:80] + ("..." if len(prompt) > 80 else ""),
                "started_at": time.time(),
                "tokens_so_far": 0,
            }
        print(f"[SynthGPU] Inference started: {model_name}, "
              f"~{self.model_weights_mb:.0f}MB vRAM")

    def on_inference_complete(self, total_tokens: int, total_ms: float):
        with self._lock:
            if self.active_session:
                entry = {
                    "model": self.active_model,
                    "prompt_preview": self.active_session.get("prompt_preview", ""),
                    "tokens": total_tokens,
                    "avg_tps": round(total_tokens / max(total_ms / 1000, 0.001), 2)
                              if total_ms > 0 else 0,
                    "completed_at": time.time(),
                }
                self.session_history.append(entry)
                if len(self.session_history) > 20:
                    self.session_history = self.session_history[-20:]
            self.active_session = None

    def _estimate_model_size_mb(self, model_name: str) -> float:
        name = model_name.lower()
        if   "tinyllama" in name: return 600
        elif "70b"   in name: return 35000
        elif "34b"   in name: return 17000
        elif "13b"   in name: return 6500
        elif "8b"    in name: return 4700
        elif "7b"    in name: return 3800
        elif "3b"    in name: return 1800
        elif "1b"    in name: return 600
        elif "phi"   in name: return 1500
        elif "gemma" in name: return 2000
        elif "mistral" in name: return 4100
        else:                  return 2000

    def _estimate_model_dimensions(self, model_name: str) -> tuple:
        name = model_name.lower()
        if   "70b" in name: return (8192, 64)
        elif "13b" in name: return (5120, 40)
        elif "8b"  in name: return (4096, 32)
        elif "7b"  in name: return (4096, 32)
        elif "3b"  in name: return (3200, 32)
        elif "1b"  in name: return (2048, 16)
        elif "phi" in name: return (2560, 32)
        else:                return (2048, 16)

    def get_inference_telemetry(self) -> dict:
        with self._lock:
            recent_tokens = list(self.token_history)[-10:]
            avg_tps, current_tps = 0.0, 0.0
            if recent_tokens:
                vals = [t["tokens_per_sec"] for t in recent_tokens]
                avg_tps = sum(vals) / len(vals)
                current_tps = vals[-1]
            return {
                "active": self.active_session is not None,
                "backend": _inference_state["backend_name"],
                "backend_type": _inference_state["backend_type"],
                "backend_url": _inference_state["backend_url"],
                "backend_status": _inference_state["backend_status"],
                "active_model": self.active_model,
                "active_session": self.active_session,
                "tokens_generated_total": self.tokens_generated,
                "avg_tokens_per_sec": round(avg_tps, 2),
                "current_tokens_per_sec": round(current_tps, 2),
                "recent_tokens": recent_tokens,
                "available_models": _inference_state["available_models"],
                "session_history": self.session_history[-5:],
            }

    def get_memory_extension(self) -> dict:
        with self._lock:
            return {
                "kv_cache_mb": round(self.kv_cache_mb, 2),
                "model_weights_mb": round(self.model_weights_mb, 1),
                "source": "System RAM — NOT hard drive",
            }


class InferenceProxyRouter:
    """APIRouter wrapping all Ollama/LM Studio proxy routes."""

    def __init__(self, engine: SynthGPUTelemetryEngine,
                 token_connections: Set[WebSocket] = None,
                 broadcast_fn: Callable = None):
        self.engine = engine
        self.token_connections = token_connections if token_connections is not None else set()
        self.broadcast_fn = broadcast_fn
        self.router = APIRouter()
        self._register_routes()

    def set_broadcast(self, token_connections: Set[WebSocket], broadcast_fn: Callable):
        self.token_connections = token_connections
        self.broadcast_fn = broadcast_fn

    def _register_routes(self):
        router = self.router
        engine = self.engine
        state = _inference_state

        # ── Inference Management ──────────────────────────────────

        @router.get("/api/inference/status")
        async def inference_status():
            telem = engine.get_inference_telemetry()
            return {
                "backend": state["backend_name"],
                "backend_type": state["backend_type"],
                "backend_url": state["backend_url"],
                "backend_status": state["backend_status"],
                "backend_connected": state["backend_status"] == "connected",
                "active_model": engine.active_model,
                "active": engine.active_session is not None,
                "tokens_generated": engine.tokens_generated,
                "avg_tps": telem["avg_tokens_per_sec"],
                "available_models": state["available_models"],
                "retry_message": (
                    "Run 'ollama serve' then click Test Connection"
                    if state["backend_status"] != "connected" else None
                ),
            }

        @router.get("/api/inference/config")
        async def inference_config():
            return {
                "endpoints": {
                    "ollama": _normalize_backend_url(OLLAMA_URL),
                    "lmstudio": _normalize_backend_url(LM_STUDIO_URL),
                },
                "custom_default": (
                    _normalize_backend_url(CUSTOM_OLLAMA_URL)
                    if CUSTOM_OLLAMA_URL else ""
                ),
                "active": {
                    "type": state["backend_type"],
                    "name": state["backend_name"],
                    "url": state["backend_url"],
                    "connected": state["backend_status"] == "connected",
                },
            }

        @router.get("/api/inference/models")
        async def list_models_endpoint():
            if state["backend_status"] != "connected":
                await detect_backend(state)
            return {
                "connected": state["backend_status"] == "connected",
                "backend": state["backend_name"],
                "models": state["available_models"],
            }

        @router.post("/api/inference/connect")
        async def connect_backend(request: Request):
            body = await request.json()
            try:
                target = _resolve_backend_target(
                    body.get("backend", "ollama"), body.get("url", "")
                )
            except ValueError as exc:
                return JSONResponse(
                    {"connected": False, "error": str(exc), "message": str(exc)},
                    status_code=400,
                )

            test_url = target["url"]
            is_ollama = target["type"] != "lmstudio"
            full_url = f"{test_url}{target['models_path']}"

            loop = asyncio.get_event_loop()

            def _do_connect():
                try:
                    req = urllib.request.Request(
                        full_url,
                        headers={"Origin": "http://localhost:8000"},
                    )
                    with _urlopen(req, timeout=5) as resp:
                        if resp.status == 200:
                            return ("ok", json.loads(resp.read().decode()))
                        return ("http_error", resp.status)
                except urllib.error.URLError as e:
                    reason = str(e.reason) if hasattr(e, 'reason') else str(e)
                    return ("connect_error", reason)
                except Exception as e:
                    return ("error", str(e))

            result, payload = await loop.run_in_executor(None, _do_connect)

            if result == "ok":
                data = payload
                backend_name = "Ollama" if is_ollama else "LM Studio"
                models = (
                    _parse_models_ollama(data) if is_ollama
                    else _parse_models_lmstudio(data)
                )
                state["backend_url"] = test_url
                state["backend_name"] = backend_name
                state["backend_type"] = target["type"]
                state["backend_status"] = "connected"
                state["available_models"] = models
                print(f"[SynthGPU] ✓ {backend_name} connected at {test_url} "
                      f"({len(models)} models)")
                return {
                    "connected": True,
                    "backend": backend_name,
                    "backend_type": target["type"],
                    "backend_url": test_url,
                    "backend_status": "connected",
                    "models": models,
                    "message": (
                        f"Connected to {target['name']} at {test_url}. "
                        f"Found {len(models)} model(s)."
                    ),
                }

            if result == "connect_error":
                reason = payload
                connection_refused = "refused" in str(reason).lower()
                return {
                    "connected": False,
                    "error": reason,
                    "backend_type": target["type"],
                    "backend_url": test_url,
                    "message": f"Cannot reach {target['name']} at {test_url}: {reason}",
                    "fix": "ollama serve" if connection_refused else None,
                    "fix_windows": (
                        "Or double-click the Ollama icon in the system tray"
                        if connection_refused else None
                    ),
                }

            return {
                "connected": False,
                "error": str(payload),
                "backend_type": target["type"],
                "backend_url": test_url,
                "message": f"{target['name']} at {test_url} returned an error ({payload})",
                "fix": "ollama serve" if is_ollama else None,
            }

        @router.post("/api/inference/disconnect")
        async def disconnect_backend():
            state["backend_type"] = None
            state["backend_url"] = None
            state["backend_name"] = None
            state["backend_status"] = "disconnected"
            state["available_models"] = []
            return {"disconnected": True}

        @router.post("/api/inference/run")
        async def run_inference(request: Request):
            body = await request.json()
            model = body.get("model", "llama3.2:1b")
            prompt = body.get("prompt", "")
            try:
                requested_max_tokens = int(body.get("max_tokens", 1024))
            except (TypeError, ValueError):
                requested_max_tokens = 1024
            requested_max_tokens = max(1, min(4096, requested_max_tokens))
            free_mb = psutil.virtual_memory().available / (1024 * 1024)
            max_tokens = engine._get_safe_max_tokens(requested_max_tokens, free_mb)
            if not state["backend_url"]:
                return JSONResponse({"error": "No LLM backend connected"}, status_code=503)

            engine.on_inference_start(model, prompt)
            d_model, num_heads = engine._estimate_model_dimensions(model)
            broadcast_fn = self.broadcast_fn
            token_connections = self.token_connections
            loop = asyncio.get_event_loop()

            # Build Ollama request with performance options to reduce RAM pressure
            ollama_body = {
                "model": model,
                "prompt": prompt,
                "stream": True,
                "options": {
                    **engine._build_ollama_options(model),
                    "num_predict": max_tokens,
                },
            }

            async def _run():
                total_tokens = 0
                start_ms = time.perf_counter() * 1000
                token_start = time.perf_counter()
                try:
                    async with _httpx_client(timeout=180.0) as client:
                        async with client.stream(
                            "POST",
                            f"{state['backend_url']}/api/generate",
                            json=ollama_body,
                        ) as resp:
                            async for line in resp.aiter_lines():
                                if not line.strip():
                                    continue
                                try:
                                    data = json.loads(line)
                                except Exception:
                                    continue
                                token_text = data.get("response", "")
                                token_ms = (time.perf_counter() - token_start) * 1000
                                token_start = time.perf_counter()
                                if token_text:
                                    # Fast path: record stats immediately (non-blocking)
                                    tok = engine._record_token_fast(
                                        token_text, token_ms, model, d_model, num_heads
                                    )
                                    total_tokens += 1
                                    # Broadcast immediately — no simulation blocking
                                    if broadcast_fn and token_connections:
                                        await broadcast_fn(token_connections, {
                                            "type": "token",
                                            "token": token_text,
                                            "token_ms": round(token_ms, 2),
                                            "tokens_per_sec": tok["tokens_per_sec"],
                                            "total_tokens": total_tokens,
                                            "warps": tok["warps"],
                                            "step": total_tokens - 1,
                                            "pct_complete": 0,
                                        })
                                    # Fire-and-forget simulation in background thread
                                    fut = loop.run_in_executor(
                                        None, engine.simulate_token_compute, d_model, num_heads
                                    )
                                    fut.add_done_callback(lambda f: f.exception())
                                if data.get("done"):
                                    total_elapsed = time.perf_counter() * 1000 - start_ms
                                    done_reason = data.get("done_reason", "stop")
                                    engine.on_inference_complete(total_tokens, total_elapsed)
                                    if broadcast_fn and token_connections:
                                        await broadcast_fn(token_connections, {
                                            "type": "done",
                                            "total_tokens": total_tokens,
                                            "total_ms": round(total_elapsed, 1),
                                            "done_reason": done_reason,
                                        })
                                    break
                except Exception as exc:
                    engine.on_inference_complete(total_tokens, 0)
                    if broadcast_fn and token_connections:
                        await broadcast_fn(token_connections, {
                            "type": "error",
                            "message": (
                                f"{state['backend_name']} request to "
                                f"{state['backend_url']} failed: "
                                f"{type(exc).__name__}: {exc}"
                            ),
                        })

            asyncio.create_task(_run())
            return {
                "status": "streaming",
                "model": model,
                "requested_max_tokens": requested_max_tokens,
                "max_tokens": max_tokens,
            }

        @router.get("/api/inference/memory")
        async def inference_memory():
            ram = psutil.virtual_memory()
            used = engine.model_weights_mb + engine.kv_cache_mb
            return {
                "virtual_vram": {
                    "source": "System RAM — NOT hard drive",
                    "explanation": (
                        "SynthGPU allocates virtual VRAM from system RAM, "
                        "identical to how a real GPU's VRAM is on-chip RAM. "
                        "Hard drive / SSD is NOT used as VRAM."
                    ),
                    "total_mb": round(engine.vram_total_mb, 0),
                    "used_mb": round(used, 1),
                    "model_weights_mb": round(engine.model_weights_mb, 1),
                    "kv_cache_mb": round(engine.kv_cache_mb, 2),
                    "free_mb": max(0.0, round(engine.vram_total_mb - used, 1)),
                },
                "system_ram": {
                    "total_gb": round(ram.total / 1e9, 1),
                    "available_gb": round(ram.available / 1e9, 1),
                    "utilization_pct": ram.percent,
                    "total_mb": round(ram.total / 1e6, 0),
                    "available_mb": round(ram.available / 1e6, 0),
                },
            }

        @router.post("/api/inference/pull")
        async def pull_model(request: Request):
            body = await request.json()
            model = body.get("model", "")
            if not state["backend_url"] or state["backend_name"] != "Ollama":
                return JSONResponse({"error": "Ollama not connected"}, status_code=503)

            async def _stream_pull() -> AsyncIterator[bytes]:
                async with _httpx_client(timeout=3600.0) as client:
                    async with client.stream(
                        "POST",
                        f"{state['backend_url']}/api/pull",
                        json={"name": model},
                    ) as resp:
                        async for line in resp.aiter_lines():
                            if line.strip():
                                yield line.encode() + b"\n"

            return StreamingResponse(_stream_pull(), media_type="application/x-ndjson")

        # ── Ollama API Proxy ──────────────────────────────────────

        @router.post("/api/generate")
        async def proxy_generate(request: Request):
            body = await request.json()
            model = body.get("model", "unknown")
            prompt = body.get("prompt", "")
            if not state["backend_url"]:
                return JSONResponse({"error": "No backend detected"}, status_code=503)
            d_model, num_heads = engine._estimate_model_dimensions(model)
            engine.on_inference_start(model, prompt)
            broadcast_fn = self.broadcast_fn
            token_connections = self.token_connections
            loop = asyncio.get_event_loop()

            # Inject performance options (only if not already set by caller)
            if "options" not in body:
                body = dict(body)
                body["options"] = engine._build_ollama_options(model)

            async def _stream() -> AsyncIterator[bytes]:
                total_tokens = 0
                start_ms = time.perf_counter() * 1000
                token_start = time.perf_counter()
                async with _httpx_client(timeout=120.0) as client:
                    async with client.stream(
                        "POST", f"{state['backend_url']}/api/generate", json=body
                    ) as resp:
                        async for line in resp.aiter_lines():
                            if not line.strip():
                                continue
                            try:
                                data = json.loads(line)
                            except Exception:
                                yield line.encode() + b"\n"
                                continue
                            token_text = data.get("response", "")
                            token_ms = (time.perf_counter() - token_start) * 1000
                            token_start = time.perf_counter()
                            if token_text:
                                tok = engine._record_token_fast(
                                    token_text, token_ms, model, d_model, num_heads
                                )
                                total_tokens += 1
                                if broadcast_fn and token_connections:
                                    await broadcast_fn(token_connections, {
                                        "type": "token",
                                        "token": token_text,
                                        "token_ms": round(token_ms, 2),
                                        "tokens_per_sec": tok["tokens_per_sec"],
                                        "total_tokens": total_tokens,
                                        "step": total_tokens - 1,
                                        "pct_complete": 0,
                                    })
                                loop.run_in_executor(
                                    None, engine.simulate_token_compute, d_model, num_heads
                                ).add_done_callback(lambda f: f.exception())
                            data["synthgpu"] = {
                                "device": "SynthGPU Virtual Accelerator",
                                "warps_executed": engine.warps_executed,
                                "vram_used_mb": round(engine.model_weights_mb + engine.kv_cache_mb, 1),
                                "no_physical_gpu": True,
                            }
                            if data.get("done"):
                                elapsed = time.perf_counter() * 1000 - start_ms
                                engine.on_inference_complete(total_tokens, elapsed)
                                data["synthgpu"]["total_tokens"] = total_tokens
                                data["synthgpu"]["tokens_per_sec"] = round(
                                    total_tokens / max(elapsed / 1000, 0.001), 2)
                            yield json.dumps(data).encode() + b"\n"

            return StreamingResponse(_stream(), media_type="application/x-ndjson")

        @router.post("/api/chat")
        async def proxy_chat(request: Request):
            body = await request.json()
            model = body.get("model", "unknown")
            messages = body.get("messages", [])
            prompt = " ".join(m.get("content", "") for m in messages[-3:])
            if not state["backend_url"]:
                return JSONResponse({"error": "No backend detected"}, status_code=503)
            d_model, num_heads = engine._estimate_model_dimensions(model)
            engine.on_inference_start(model, prompt)

            async def _stream() -> AsyncIterator[bytes]:
                token_start = time.perf_counter()
                total_tokens = 0
                start_ms = time.perf_counter() * 1000
                async with _httpx_client(timeout=120.0) as client:
                    async with client.stream(
                        "POST", f"{state['backend_url']}/api/chat", json=body
                    ) as resp:
                        async for line in resp.aiter_lines():
                            if not line.strip():
                                continue
                            try:
                                data = json.loads(line)
                            except Exception:
                                yield line.encode() + b"\n"
                                continue
                            token_text = data.get("message", {}).get("content", "")
                            token_ms = (time.perf_counter() - token_start) * 1000
                            token_start = time.perf_counter()
                            if token_text:
                                engine.on_token_generated(
                                    token_text, token_ms, model, d_model, num_heads
                                )
                                total_tokens += 1
                            data["synthgpu"] = {
                                "device": "SynthGPU Virtual Accelerator",
                                "no_physical_gpu": True,
                            }
                            if data.get("done"):
                                engine.on_inference_complete(
                                    total_tokens, time.perf_counter() * 1000 - start_ms
                                )
                            yield json.dumps(data).encode() + b"\n"

            return StreamingResponse(_stream(), media_type="application/x-ndjson")

        @router.post("/v1/chat/completions")
        async def proxy_openai_chat(request: Request):
            body = await request.json()
            model = body.get("model", "unknown")
            messages = body.get("messages", [])
            prompt = " ".join(m.get("content", "") for m in messages[-3:])
            stream = body.get("stream", False)
            if not state["backend_url"]:
                return JSONResponse({"error": "No backend detected"}, status_code=503)
            d_model, num_heads = engine._estimate_model_dimensions(model)
            engine.on_inference_start(model, prompt)
            target = f"{state['backend_url']}/v1/chat/completions"

            async def _stream() -> AsyncIterator[bytes]:
                token_start = time.perf_counter()
                total_tokens = 0
                async with _httpx_client(timeout=120.0) as client:
                    async with client.stream("POST", target, json=body) as resp:
                        async for line in resp.aiter_lines():
                            if not line.strip():
                                continue
                            if line.startswith("data: "):
                                data_str = line[6:]
                                if data_str == "[DONE]":
                                    engine.on_inference_complete(total_tokens, 0)
                                    yield b"data: [DONE]\n\n"
                                    continue
                                try:
                                    data = json.loads(data_str)
                                    delta = (data.get("choices") or [{}])[0].get("delta", {})
                                    token_text = delta.get("content", "")
                                    token_ms = (time.perf_counter() - token_start) * 1000
                                    token_start = time.perf_counter()
                                    if token_text:
                                        engine.on_token_generated(
                                            token_text, token_ms, model, d_model, num_heads
                                        )
                                        total_tokens += 1
                                    data["synthgpu_device"] = "SynthGPU Virtual Accelerator"
                                    data["no_physical_gpu"] = True
                                    yield f"data: {json.dumps(data)}\n\n".encode()
                                except Exception:
                                    yield line.encode() + b"\n"
                            else:
                                yield line.encode() + b"\n"

            if stream:
                return StreamingResponse(_stream(), media_type="text/event-stream")
            else:
                async with _httpx_client(timeout=120.0) as client:
                    resp = await client.post(target, json=body)
                    data = resp.json()
                    usage = data.get("usage", {})
                    engine.on_inference_complete(usage.get("completion_tokens", 0), 0)
                    data["synthgpu_device"] = "SynthGPU Virtual Accelerator"
                    data["no_physical_gpu"] = True
                    return JSONResponse(data)

        @router.get("/v1/models")
        async def list_openai_models():
            if not state["backend_url"]:
                return JSONResponse({"data": []})
            async with _httpx_client(timeout=5.0) as client:
                try:
                    if state["backend_name"] == "Ollama":
                        resp = await client.get(f"{state['backend_url']}/api/tags")
                        models = resp.json().get("models", [])
                        return JSONResponse({
                            "data": [{"id": m["name"], "object": "model"} for m in models],
                            "synthgpu_device": "SynthGPU Virtual Accelerator",
                        })
                    else:
                        resp = await client.get(f"{state['backend_url']}/v1/models")
                        return JSONResponse(resp.json())
                except Exception as e:
                    return JSONResponse({"error": str(e)}, status_code=503)

        @router.get("/api/tags")
        async def list_tags():
            if not state["backend_url"]:
                return JSONResponse({"models": []})
            async with _httpx_client(timeout=5.0) as client:
                try:
                    resp = await client.get(f"{state['backend_url']}/api/tags")
                    return JSONResponse(resp.json())
                except Exception:
                    return JSONResponse({"models": []})

        @router.get("/synthgpu/status")
        async def synthgpu_status():
            telem = engine.get_inference_telemetry()
            return JSONResponse({
                "status": "active",
                "device": "SynthGPU Virtual Accelerator",
                "version": __version__,
                "no_physical_gpu": True,
                "backend": state["backend_name"],
                "backend_url": state["backend_url"],
                **telem,
            })

        @router.get("/synthgpu/memory")
        async def memory_info():
            ram = psutil.virtual_memory()
            used = engine.model_weights_mb + engine.kv_cache_mb
            return JSONResponse({
                "virtual_vram": {
                    "source": "System RAM — NOT hard drive",
                    "total_mb": round(engine.vram_total_mb, 0),
                    "used_mb": round(used, 1),
                    "model_weights_mb": round(engine.model_weights_mb, 1),
                    "kv_cache_mb": round(engine.kv_cache_mb, 2),
                    "free_mb": max(0.0, round(engine.vram_total_mb - used, 1)),
                },
                "system_ram": {
                    "total_gb": round(ram.total / 1e9, 1),
                    "available_gb": round(ram.available / 1e9, 1),
                    "utilization_pct": ram.percent,
                },
                "roadmap": {
                    "v0.3": "Memory-mapped overflow for models larger than RAM (same as llama.cpp)"
                },
            })

        @router.get("/health")
        async def health():
            return {"status": "ok", "synthgpu": "active", "no_physical_gpu": True}

        @router.get("/api/debug/routes")
        async def debug_routes():
            return {
                "ollama_connected": state["backend_status"] == "connected",
                "ollama_url": state["backend_url"],
                "ollama_name": state["backend_name"],
                "available_models": [m["name"] for m in state["available_models"]],
                "routes": [
                    "/api/inference/status",
                    "/api/inference/models",
                    "/api/inference/connect",
                    "/api/inference/run",
                    "/api/inference/memory",
                    "/api/inference/pull",
                    "/api/generate",
                    "/api/chat",
                    "/api/tags",
                    "/v1/chat/completions",
                    "/v1/models",
                    "/synthgpu/status",
                    "/synthgpu/memory",
                    "/health",
                ],
            }
