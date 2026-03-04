@echo off
setlocal EnableDelayedExpansion

:: ============================================================
:: SynthGPU Vulkan ICD — Windows Uninstaller
:: Must be run as Administrator.
:: ============================================================

set INSTALL_DIR=%ProgramFiles%\SynthGPU
set JSON_DEST=%INSTALL_DIR%\synthgpu_icd_win64.json

echo.
echo  =====================================================
echo   SynthGPU Vulkan ICD Uninstaller
echo  =====================================================
echo.

net session >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Must be run as Administrator.
    pause
    exit /b 1
)

:: Remove registry entries
echo [SynthGPU] Removing Vulkan loader registry entry...
reg delete "HKLM\SOFTWARE\Khronos\Vulkan\Drivers" /v "%JSON_DEST%" /f >nul 2>&1

echo [SynthGPU] Removing uninstall entry...
reg delete "HKLM\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\SynthGPU-VulkanICD" /f >nul 2>&1

:: Remove installed files
echo [SynthGPU] Removing files from %INSTALL_DIR%...
if exist "%INSTALL_DIR%" (
    rmdir /s /q "%INSTALL_DIR%"
)

echo [SynthGPU] Uninstall complete.
echo.
endlocal
