@echo off
setlocal EnableDelayedExpansion

set ICD_DIR=%~dp0..
set BUILD_DIR=%ICD_DIR%\build
set DLL_PATH=%BUILD_DIR%\Release\synthgpu_vulkan_icd.dll
set JSON_SRC=%ICD_DIR%\manifests\synthgpu_icd_win64.json
set JSON_DEST=%BUILD_DIR%\synthgpu_icd_win64_installed.json

echo [SynthGPU] Installing Vulkan ICD for Windows...
echo.

:: Check DLL exists
if not exist "%DLL_PATH%" (
    echo [ERROR] DLL not found: %DLL_PATH%
    echo [ERROR] Build first: cd build ^&^& cmake --build . --config Release
    exit /b 1
)

:: Write manifest with absolute DLL path
powershell -NoProfile -Command ^
    "(Get-Content '%JSON_SRC%') -replace '\.\\\\synthgpu_vulkan_icd\.dll', '%DLL_PATH:\=\\%' | Set-Content '%JSON_DEST%'"

if %ERRORLEVEL% neq 0 (
    echo [ERROR] Failed to write manifest. Run as Administrator.
    exit /b 1
)

:: Register in Vulkan Loader registry (requires Administrator)
reg add "HKLM\SOFTWARE\Khronos\Vulkan\Drivers" /v "%JSON_DEST%" /t REG_DWORD /d 0 /f
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Registry write failed. Run this script as Administrator.
    exit /b 1
)

echo [SynthGPU] Registered manifest: %JSON_DEST%
echo [SynthGPU] DLL path:            %DLL_PATH%
echo.
echo [SynthGPU] Verifying installation...
vulkaninfo --summary 2>nul | findstr /i "SynthGPU"
if %ERRORLEVEL% == 0 (
    echo [SynthGPU] SUCCESS — SynthGPU Virtual Accelerator detected by Vulkan!
) else (
    echo [SynthGPU] Installed. Run 'vulkaninfo --summary' to verify.
    echo [SynthGPU] Note: vulkaninfo may need to be run separately.
)

echo.
echo [SynthGPU] Done. Restart any Vulkan applications to pick up the new ICD.
endlocal
