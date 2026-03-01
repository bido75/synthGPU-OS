@echo off
echo ============================================================
echo   SynthGPU Beta v0.2 -- Starting
echo ============================================================

REM --- Install backend deps ---
echo [1/4] Installing backend dependencies...
cd /d "%~dp0backend"
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: pip install failed. Make sure Python is in PATH.
    pause & exit /b 1
)
cd /d "%~dp0"

REM --- Build frontend ---
echo [2/4] Installing and building frontend...
cd /d "%~dp0frontend"
call npm install
if errorlevel 1 ( echo ERROR: npm install failed. & pause & exit /b 1 )
call npm run build
if errorlevel 1 ( echo ERROR: npm build failed. & pause & exit /b 1 )
cd /d "%~dp0"

REM --- Start backend in separate window ---
echo [3/4] Starting backend server...
start "SynthGPU Backend - http://localhost:8000" cmd /k "cd /d "%~dp0backend" && uvicorn main:app --host 0.0.0.0 --port 8000"

REM --- Open browser ---
echo [4/4] Opening browser in 4 seconds...
timeout /t 4 /nobreak >nul
start http://localhost:8000

echo ============================================================
echo   SynthGPU running at: http://localhost:8000
echo   Close the "SynthGPU Backend" window to stop.
echo ============================================================
