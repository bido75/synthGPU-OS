@echo off
setlocal

set ICD_DIR=%~dp0..

:: Remove registry entry
reg delete "HKLM\SOFTWARE\Khronos\Vulkan\Drivers" /f 2>nul
echo [SynthGPU] Removed Vulkan ICD registry entries.

:: Remove installed manifest
set JSON_DEST=%ICD_DIR%\build\synthgpu_icd_win64_installed.json
if exist "%JSON_DEST%" (
    del "%JSON_DEST%"
    echo [SynthGPU] Removed manifest: %JSON_DEST%
)

echo [SynthGPU] Uninstall complete.
endlocal
