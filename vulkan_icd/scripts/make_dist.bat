@echo off
:: ============================================================
:: SynthGPU — Build distributable package
:: Run this AFTER cmake --build to create a self-contained
:: folder that can be zipped and sent to any Windows machine.
:: ============================================================
setlocal

set SCRIPT_DIR=%~dp0
set ROOT=%SCRIPT_DIR%..
set DLL_SRC=%ROOT%\build\Release\synthgpu_vulkan_icd.dll
set DIST_DIR=%ROOT%\dist

if not exist "%DLL_SRC%" (
    echo [ERROR] DLL not found. Build first:
    echo   cmake --build vulkan_icd\build --config Release
    exit /b 1
)

echo [SynthGPU] Creating distribution package...

:: Create dist folder
if exist "%DIST_DIR%" rmdir /s /q "%DIST_DIR%"
mkdir "%DIST_DIR%"

:: Copy DLL
copy /Y "%DLL_SRC%" "%DIST_DIR%\synthgpu_vulkan_icd.dll" >nul

:: Copy manifest
copy /Y "%ROOT%\manifests\synthgpu_icd_win64.json" "%DIST_DIR%\synthgpu_icd_win64.json" >nul

:: Copy installer scripts (self-contained — DLL is next to the .bat)
copy /Y "%SCRIPT_DIR%install_windows.bat" "%DIST_DIR%\install_windows.bat" >nul
copy /Y "%SCRIPT_DIR%uninstall.bat" "%DIST_DIR%\uninstall.bat" >nul

echo.
echo [SynthGPU] Package created: %DIST_DIR%
echo [SynthGPU] Contents:
dir /b "%DIST_DIR%"
echo.
echo [SynthGPU] To install on any Windows machine:
echo   1. Copy the dist\ folder to target machine
echo   2. Right-click install_windows.bat
echo   3. "Run as administrator"
echo.
endlocal
