@echo off
echo ================================================
echo   SynthGPU ^— Starting Backend (Optimized)
echo ================================================
echo.

REM ── Python performance ────────────────────────────────────────
set PYTHONOPTIMIZE=1
set PYTHONDONTWRITEBYTECODE=1

REM ── BLAS/numpy thread optimization ───────────────────────────
REM Must be set here too (in case uvicorn restarts the process)
set OPENBLAS_NUM_THREADS=%NUMBER_OF_PROCESSORS%
set OMP_NUM_THREADS=%NUMBER_OF_PROCESSORS%
set MKL_NUM_THREADS=%NUMBER_OF_PROCESSORS%
set NUMEXPR_NUM_THREADS=%NUMBER_OF_PROCESSORS%
set BLAS_NUM_THREADS=%NUMBER_OF_PROCESSORS%
set NPY_DISABLE_CPU_FEATURES=

echo CPU Cores:    %NUMBER_OF_PROCESSORS%
echo BLAS Threads: %NUMBER_OF_PROCESSORS%
echo.
echo Starting SynthGPU backend on http://localhost:8000...
echo.

cd /d "%~dp0backend"
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
