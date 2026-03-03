@echo off
REM ============================================================
REM SynthGPU CUDA Shim — Windows install script
REM Usage: install.bat [--dev]
REM ============================================================
setlocal enabledelayedexpansion

set "SHIM_DIR=%~dp0"
set "BUILD_DIR=%SHIM_DIR%build"
set "PREFIX=%SHIM_DIR%dist"
set "DEV_MODE=0"

:parse_args
if "%~1"=="--dev" (set "DEV_MODE=1" & shift & goto parse_args)

echo.
echo ====================================================
echo   SynthGPU CUDA Shim -- Windows Installer
echo ====================================================
echo   Shim dir : %SHIM_DIR%
echo   Build dir: %BUILD_DIR%
echo   Prefix   : %PREFIX%
echo.

REM ── Dependency check ─────────────────────────────────
echo =^> Checking dependencies...
where cmake  >nul 2>&1 || (echo ERROR: cmake not found. Install from https://cmake.org & exit /b 1)
where python >nul 2>&1 || (echo ERROR: python not found. Install from https://python.org & exit /b 1)
where pip    >nul 2>&1 || (echo ERROR: pip not found & exit /b 1)
echo    All required tools present.

REM ── Python package ────────────────────────────────────
echo.
echo =^> Installing Python kernel layer...
if "%DEV_MODE%"=="1" (
    pip install -e "%SHIM_DIR%" --quiet
    echo    Installed in editable (dev) mode.
) else (
    pip install "%SHIM_DIR%" --quiet
    echo    Installed.
)

REM ── C library build ───────────────────────────────────
echo.
echo =^> Building C shared library...
if not exist "%BUILD_DIR%" mkdir "%BUILD_DIR%"

cmake -S "%SHIM_DIR%" -B "%BUILD_DIR%" ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DCMAKE_INSTALL_PREFIX="%PREFIX%"
if errorlevel 1 (echo CMake configure failed & exit /b 1)

cmake --build "%BUILD_DIR%" --config Release
if errorlevel 1 (echo CMake build failed & exit /b 1)

cmake --install "%BUILD_DIR%" --config Release
if errorlevel 1 (echo CMake install failed & exit /b 1)

echo.
echo ====================================================
echo   Install complete!
echo ====================================================
echo.
echo   DLL location: %PREFIX%\bin\SynthGPUCUDA.dll
echo.
echo   To run tests:
echo     pytest ..\tests\ -v
echo.
endlocal
