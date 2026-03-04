@echo off
setlocal EnableDelayedExpansion

:: ============================================================
:: SynthGPU Vulkan ICD — Windows Installer
:: Works from the repo (dev) or from a distributed package.
:: Must be run as Administrator.
:: ============================================================

set PRODUCT_NAME=SynthGPU
set INSTALL_DIR=%ProgramFiles%\SynthGPU
set DLL_NAME=synthgpu_vulkan_icd.dll
set JSON_NAME=synthgpu_icd_win64.json

:: Locate the DLL — check distributed package first, then dev build
set SCRIPT_DIR=%~dp0
set DLL_SRC=

if exist "%SCRIPT_DIR%%DLL_NAME%" (
    set DLL_SRC=%SCRIPT_DIR%%DLL_NAME%
) else if exist "%SCRIPT_DIR%..\build\Release\%DLL_NAME%" (
    set DLL_SRC=%SCRIPT_DIR%..\build\Release\%DLL_NAME%
) else if exist "%SCRIPT_DIR%..\bin\%DLL_NAME%" (
    set DLL_SRC=%SCRIPT_DIR%..\bin\%DLL_NAME%
)

if "%DLL_SRC%"=="" (
    echo [ERROR] Cannot find %DLL_NAME%
    echo [ERROR] Expected locations:
    echo [ERROR]   %SCRIPT_DIR%%DLL_NAME%
    echo [ERROR]   %SCRIPT_DIR%..\build\Release\%DLL_NAME%
    echo [ERROR] Build first: cmake --build vulkan_icd\build --config Release
    exit /b 1
)

:: Locate the manifest template
set JSON_SRC=
if exist "%SCRIPT_DIR%%JSON_NAME%" (
    set JSON_SRC=%SCRIPT_DIR%%JSON_NAME%
) else if exist "%SCRIPT_DIR%..\manifests\%JSON_NAME%" (
    set JSON_SRC=%SCRIPT_DIR%..\manifests\%JSON_NAME%
)

if "%JSON_SRC%"=="" (
    echo [ERROR] Cannot find manifest %JSON_NAME%
    exit /b 1
)

echo.
echo  =====================================================
echo   SynthGPU Vulkan ICD Installer
echo  =====================================================
echo   DLL source : %DLL_SRC%
echo   Install to : %INSTALL_DIR%
echo  =====================================================
echo.

:: Check for Administrator rights
net session >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ERROR] This installer must be run as Administrator.
    echo [ERROR] Right-click install_windows.bat and choose "Run as administrator"
    pause
    exit /b 1
)

:: Create install directory
if not exist "%INSTALL_DIR%" (
    mkdir "%INSTALL_DIR%"
    if %ERRORLEVEL% neq 0 (
        echo [ERROR] Failed to create %INSTALL_DIR%
        exit /b 1
    )
)

:: Copy DLL to install directory
echo [SynthGPU] Copying DLL...
copy /Y "%DLL_SRC%" "%INSTALL_DIR%\%DLL_NAME%" >nul
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Failed to copy DLL to %INSTALL_DIR%
    exit /b 1
)

:: Write manifest with the installed DLL absolute path
set DLL_DEST=%INSTALL_DIR%\%DLL_NAME%
set JSON_DEST=%INSTALL_DIR%\%JSON_NAME%
set DLL_DEST_JSON=%DLL_DEST:\=\\%

echo [SynthGPU] Writing manifest...
powershell -NoProfile -Command ^
    "(Get-Content '%JSON_SRC%') -replace '\.[\\/\\\\]*synthgpu_vulkan_icd\.dll', '%DLL_DEST_JSON%' | Set-Content '%JSON_DEST%'"
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Failed to write manifest to %JSON_DEST%
    exit /b 1
)

:: Register in Vulkan Loader registry
echo [SynthGPU] Registering with Vulkan loader...
reg add "HKLM\SOFTWARE\Khronos\Vulkan\Drivers" /v "%JSON_DEST%" /t REG_DWORD /d 0 /f >nul
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Registry write failed.
    exit /b 1
)

:: Save uninstall info to registry
reg add "HKLM\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\SynthGPU-VulkanICD" /v "DisplayName" /t REG_SZ /d "SynthGPU Vulkan ICD v0.3" /f >nul
reg add "HKLM\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\SynthGPU-VulkanICD" /v "InstallLocation" /t REG_SZ /d "%INSTALL_DIR%" /f >nul
reg add "HKLM\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\SynthGPU-VulkanICD" /v "DisplayVersion" /t REG_SZ /d "0.3.0" /f >nul
reg add "HKLM\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\SynthGPU-VulkanICD" /v "Publisher" /t REG_SZ /d "SynthGPU" /f >nul

echo.
echo [SynthGPU] Installation complete!
echo [SynthGPU]   DLL    : %INSTALL_DIR%\%DLL_NAME%
echo [SynthGPU]   Manifest: %JSON_DEST%
echo.
echo [SynthGPU] Verifying...
vulkaninfo --summary 2>nul | findstr /i "SynthGPU"
if %ERRORLEVEL% == 0 (
    echo [SynthGPU] SUCCESS - SynthGPU Virtual Accelerator detected by Vulkan!
) else (
    echo [SynthGPU] Registered. Open a new terminal and run:
    echo [SynthGPU]   vulkaninfo --summary
    echo [SynthGPU] You should see: deviceName = SynthGPU Virtual Accelerator v0.3
)
echo.
endlocal
