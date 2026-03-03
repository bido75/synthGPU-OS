@echo off
echo ================================================
echo   SynthGPU ^— Starting Ollama (Optimized)
echo ================================================
echo.

REM ── CORS: Allow SynthGPU backend to connect ──────────────────
set OLLAMA_ORIGINS=*
set OLLAMA_HOST=0.0.0.0:11434

REM ── Performance: Maximize throughput ─────────────────────────
REM One request at a time — better latency than juggling multiple
set OLLAMA_NUM_PARALLEL=1
REM Keep model in memory between requests (avoids 30s reload delay)
set OLLAMA_MAX_LOADED_MODELS=1
REM Keep model loaded for 60 minutes of inactivity
set OLLAMA_KEEP_ALIVE=60m

REM ── Memory: Reduce KV cache pressure ─────────────────────────
REM Quantize KV cache to fp8 — halves KV memory (640MB -> ~320MB)
set OLLAMA_KV_CACHE_TYPE=q8_0

REM ── No GPU offload attempts (CPU-only machine) ─────────────────
set CUDA_VISIBLE_DEVICES=
set HIP_VISIBLE_DEVICES=

echo OLLAMA_ORIGINS      = %OLLAMA_ORIGINS%
echo OLLAMA_HOST         = %OLLAMA_HOST%
echo OLLAMA_NUM_PARALLEL = %OLLAMA_NUM_PARALLEL%
echo OLLAMA_KEEP_ALIVE   = %OLLAMA_KEEP_ALIVE%
echo OLLAMA_KV_CACHE_TYPE= %OLLAMA_KV_CACHE_TYPE%
echo.
echo Starting Ollama server...
echo (Keep this window open while using SynthGPU)
echo.
ollama serve
