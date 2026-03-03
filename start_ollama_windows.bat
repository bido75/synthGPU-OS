@echo off
echo ================================================
echo   SynthGPU ^— Ollama Optimized for 2-Core Machine
echo ================================================
echo.
echo Hardware profile:
echo   CPU cores: 2 physical, 4 logical (hyperthreaded)
echo   Free RAM:  ~1.5 GB (8GB total, 6.5GB used by OS)
echo   SIMD:      SSE4.2 (no AVX2)
echo.
echo Recommended model: tinyllama:latest (638MB)
echo   - Fits entirely in RAM — no disk swapping
echo   - Expected speed: 2-4 tokens/sec
echo.
echo Avoid on this machine (will swap to disk, 0.3-0.5 tok/sec):
echo   - phi:latest (1.6GB)
echo   - Any model over 1GB
echo.

REM ── CORS: Allow SynthGPU backend to connect ──────────────────
set OLLAMA_ORIGINS=*
set OLLAMA_HOST=0.0.0.0:11434

REM ── Threads: Use ALL 4 logical cores (was defaulting to 2) ────
set OLLAMA_NUM_THREAD=4

REM ── Memory: Keep model loaded for 60 min (no reload delay) ────
set OLLAMA_KEEP_ALIVE=60m

REM ── Performance: Single model, no parallel requests ──────────
set OLLAMA_NUM_PARALLEL=1
set OLLAMA_MAX_LOADED_MODELS=1

REM ── KV cache: Quantize to q8_0 (halves KV memory) ────────────
set OLLAMA_KV_CACHE_TYPE=q8_0

REM ── No GPU detection (CPU-only machine) ──────────────────────
set CUDA_VISIBLE_DEVICES=
set ROCR_VISIBLE_DEVICES=

echo Settings applied:
echo   OLLAMA_ORIGINS      = %OLLAMA_ORIGINS%
echo   OLLAMA_NUM_THREAD   = %OLLAMA_NUM_THREAD%
echo   OLLAMA_KEEP_ALIVE   = %OLLAMA_KEEP_ALIVE%
echo   OLLAMA_NUM_PARALLEL = %OLLAMA_NUM_PARALLEL%
echo   OLLAMA_KV_CACHE_TYPE= %OLLAMA_KV_CACHE_TYPE%
echo.
echo TIP: Run this FIRST, then start SynthGPU backend
echo TIP: Use tinyllama:latest for fastest demo responses
echo.
echo Starting Ollama...
ollama serve
