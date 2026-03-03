@echo off
echo ========================================
echo   SynthGPU Full Stack — Windows Startup
echo ========================================
echo.
echo This will open 3 windows:
echo   1. Ollama (with CORS enabled)
echo   2. SynthGPU Backend  (port 8000)
echo   3. SynthGPU Frontend (port 5173)
echo.

REM ── Step 1: Ollama ──────────────────────────────────────────
echo [1/3] Starting Ollama with CORS enabled...
start "Ollama Server" cmd /k "set OLLAMA_ORIGINS=* && set OLLAMA_HOST=0.0.0.0:11434 && echo Ollama starting... && ollama serve"
timeout /t 3 /nobreak > nul

REM ── Step 2: Backend ─────────────────────────────────────────
echo [2/3] Starting SynthGPU Backend on port 8000...
start "SynthGPU Backend" cmd /k "cd /d %~dp0backend && python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload"
timeout /t 3 /nobreak > nul

REM ── Step 3: Frontend ────────────────────────────────────────
echo [3/3] Starting SynthGPU Frontend on port 5173...
start "SynthGPU Frontend" cmd /k "cd /d %~dp0frontend && npm run dev"
timeout /t 4 /nobreak > nul

echo.
echo ========================================
echo   All services started!
echo.
echo   Dashboard : http://localhost:5173
echo   Backend   : http://localhost:8000
echo   Ollama    : http://localhost:11434
echo.
echo   Models available: tinyllama, phi
echo   No physical GPU required.
echo ========================================
echo.
echo Press any key to close this launcher window.
echo (The 3 service windows stay open)
pause > nul
