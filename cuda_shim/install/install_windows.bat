@echo off
REM =================================================================
REM SynthGPU CUDA Shim — Windows Installer
REM =================================================================
REM Usage: install_windows.bat [--dev]
REM =================================================================
setlocal enabledelayedexpansion

set "SCRIPT_DIR=%~dp0"
set "SHIM_DIR=%SCRIPT_DIR%.."
set "BUILD_DIR=%SHIM_DIR%\build"
set "PREFIX=%SHIM_DIR%\dist"
set "DEV_MODE=0"

:parse
if "%~1"=="--dev" ( set "DEV_MODE=1" & shift & goto parse )

echo.
echo ===================================================
echo   SynthGPU CUDA Shim -- Windows Installer
echo ===================================================
echo   Shim dir : %SHIM_DIR%
echo   Prefix   : %PREFIX%
echo.

REM ── Check prerequisites ──────────────────────────────────────
echo [0/4] Checking prerequisites...
where cmake  >nul 2>&1 || (echo ERROR: cmake not found. Install from https://cmake.org & exit /b 1)
where python >nul 2>&1 || (echo ERROR: python not found. Install from https://python.org & exit /b 1)
where pip    >nul 2>&1 || (echo ERROR: pip not found & exit /b 1)
echo    All tools present.

REM ── Python kernel layer ───────────────────────────────────────
echo.
echo [1/4] Installing Python kernel layer...
if "%DEV_MODE%"=="1" (
    pip install -e "%SHIM_DIR%" --quiet
    echo    Installed in editable (dev) mode.
) else (
    pip install "%SHIM_DIR%" --quiet
    echo    Installed.
)

REM ── Build C library ───────────────────────────────────────────
echo.
echo [2/4] Building SynthGPUCUDA.dll...
if not exist "%BUILD_DIR%" mkdir "%BUILD_DIR%"
cmake -S "%SHIM_DIR%" -B "%BUILD_DIR%" ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DCMAKE_INSTALL_PREFIX="%PREFIX%"
if errorlevel 1 ( echo CMake configure failed & exit /b 1 )
cmake --build "%BUILD_DIR%" --config Release
if errorlevel 1 ( echo Build failed & exit /b 1 )

REM ── Install ───────────────────────────────────────────────────
echo.
echo [3/4] Installing to %PREFIX%...
cmake --install "%BUILD_DIR%" --config Release
if errorlevel 1 ( echo Install failed & exit /b 1 )

REM ── Smoke test ────────────────────────────────────────────────
echo.
echo [4/4] Running Python smoke test...
python -c "from cuda_shim.kernels.bridge_api import get_telemetry; t=get_telemetry(); print('  bridge OK — shim_active:', t['shim_active'])"
if errorlevel 1 ( echo WARNING: Python smoke test failed. Check install. )

echo.
echo ===================================================
echo   Installation complete!
echo.
echo   DLL location: %PREFIX%\bin\SynthGPUCUDA.dll
echo.
echo   To activate (add DLL dir to PATH):
echo     set PATH=%PREFIX%\bin;%%PATH%%
echo.
echo   To run tests:
echo     pytest ..\tests\ -v
echo ===================================================
endlocal
