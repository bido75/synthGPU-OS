@echo off
echo ================================================
echo   SynthGPU Full Stack ^— Optimized Startup
echo ================================================
echo.
echo This will open 3 windows:
echo   1. Ollama Server    (CORS + KV cache + keep-alive)
echo   2. SynthGPU Backend (BLAS threads + simulation fix)
echo   3. SynthGPU Frontend
echo.
echo Press any key to start all three services...
pause > nul

echo.
echo [1/3] Starting Ollama (optimized)...
start "Ollama - SynthGPU" cmd /k "set OLLAMA_ORIGINS=* && set OLLAMA_HOST=0.0.0.0:11434 && set OLLAMA_NUM_PARALLEL=1 && set OLLAMA_MAX_LOADED_MODELS=1 && set OLLAMA_KEEP_ALIVE=60m && set OLLAMA_KV_CACHE_TYPE=q8_0 && set CUDA_VISIBLE_DEVICES= && echo Ollama starting... && ollama serve"
timeout /t 5 /nobreak > nul

echo [2/3] Starting SynthGPU Backend (optimized)...
start "SynthGPU Backend" cmd /k "set OPENBLAS_NUM_THREADS=%NUMBER_OF_PROCESSORS% && set OMP_NUM_THREADS=%NUMBER_OF_PROCESSORS% && set MKL_NUM_THREADS=%NUMBER_OF_PROCESSORS% && set BLAS_NUM_THREADS=%NUMBER_OF_PROCESSORS% && set PYTHONOPTIMIZE=1 && cd /d "%~dp0backend" && python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload"
timeout /t 4 /nobreak > nul

echo [3/3] Starting SynthGPU Frontend...
start "SynthGPU Frontend" cmd /k "cd /d "%~dp0frontend" && npm run dev"
timeout /t 3 /nobreak > nul

echo.
echo ================================================
echo   All services started!
echo.
echo   Dashboard:  http://localhost:5173
echo   Backend:    http://localhost:8000
echo   Ollama:     http://localhost:11434
echo.
echo   Recommended model: tinyllama (fastest)
echo   No physical GPU required.
echo ================================================
echo.
echo Performance tips:
echo   - tinyllama is fastest (best for live demo)
echo   - phi/deepseek-r1:1.5b = smarter, same hardware
echo   - Keep prompts under 200 chars for fastest first token
echo   - Model stays loaded for 60min (no reload delay)
echo.
pause
