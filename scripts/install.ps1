[CmdletBinding()]
param(
    [switch]$Uninstall,
    [switch]$NoOpen,
    [string]$Distro = "Ubuntu-24.04",
    [string]$RepoUrl = "https://github.com/bido75/SynthGPU.git",
    [string]$Branch = "main"
)

$ErrorActionPreference = "Stop"
$ProjectName = "synthgpu"

function Invoke-Checked {
    param([string]$Description, [scriptblock]$Action)
    Write-Host "==> $Description"
    & $Action
    if ($LASTEXITCODE -ne 0) {
        throw "$Description failed with exit code $LASTEXITCODE"
    }
}

function Get-WslDistros {
    $output = (& wsl.exe --list --quiet 2>$null) -join "`n"
    $cleanOutput = $output -replace "`0", ""
    return @(($cleanOutput -split "`r?`n") | Where-Object { $_.Trim() })
}

function Invoke-WslRoot {
    param([string]$Command)
    & wsl.exe -d $Distro -u root -- bash -lc $Command
    if ($LASTEXITCODE -ne 0) {
        throw "WSL command failed with exit code $LASTEXITCODE"
    }
}

function Get-LinuxInstallContext {
    $command = 'u=$(getent passwd 1000 | cut -d: -f1); if [ -n "$u" ]; then h=$(getent passwd "$u" | cut -d: -f6); else u=root; h=/root; fi; printf "%s|%s" "$u" "$h"'
    $context = & wsl.exe -d $Distro -u root -- bash -lc $command
    if ($LASTEXITCODE -ne 0 -or -not $context) {
        throw "Could not determine the Ubuntu install user"
    }
    $parts = (($context -join "") -replace "`0", "").Split("|", 2)
    return @{ User = $parts[0]; Home = $parts[1]; Repo = "$($parts[1])/synthgpu" }
}

if ($Uninstall) {
    $distros = Get-WslDistros
    if ($Distro -notin $distros) {
        Write-Host "Nothing to uninstall: WSL distro '$Distro' is not installed."
        exit 0
    }
    $context = Get-LinuxInstallContext
    $repo = $context.Repo
    Invoke-WslRoot "if [ -f '$repo/docker-compose.yml' ]; then cd '$repo' && docker compose -p '$ProjectName' down --rmi local --volumes --remove-orphans; else echo 'SynthGPU checkout not found; nothing to remove.'; fi"
    Write-Host "SynthGPU containers, local images, and project volumes removed."
    Write-Host "The '$Distro' WSL distribution and source checkout were preserved."
    exit 0
}

$identity = [Security.Principal.WindowsIdentity]::GetCurrent()
$principal = [Security.Principal.WindowsPrincipal]::new($identity)
$isAdministrator = $principal.IsInRole(
    [Security.Principal.WindowsBuiltInRole]::Administrator
)
if (-not $isAdministrator) {
    throw "Run this installer from an Administrator PowerShell session."
}

$computer = Get-CimInstance Win32_ComputerSystem
$processors = Get-CimInstance Win32_Processor
$firmwareVirtualization = @($processors | Where-Object VirtualizationFirmwareEnabled).Count -gt 0
if (-not $computer.HypervisorPresent -and -not $firmwareVirtualization) {
    throw "Hardware virtualization is disabled. Enable Intel VT-x/AMD-V in firmware first."
}

$rebootRequired = $false
foreach ($featureName in @("Microsoft-Windows-Subsystem-Linux", "VirtualMachinePlatform")) {
    $feature = Get-WindowsOptionalFeature -Online -FeatureName $featureName
    if ($feature.State -ne "Enabled") {
        Write-Host "==> Enabling Windows feature: $featureName"
        $result = Enable-WindowsOptionalFeature -Online -FeatureName $featureName -All -NoRestart
        if ($result.RestartNeeded) { $rebootRequired = $true }
    }
}
if ($rebootRequired) {
    Write-Warning "Windows features were enabled. Restart Windows, then rerun this script."
    exit 3010
}

Invoke-Checked "Checking WSL" { wsl.exe --version }
$distros = Get-WslDistros
if ($Distro -notin $distros) {
    Invoke-Checked "Installing $Distro" {
        wsl.exe --install --distribution $Distro --no-launch
    }
    Write-Host "The distro was installed. Initializing it now; Windows may show a first-run prompt."
}

Invoke-WslRoot "echo 'WSL2 alive'"
$context = Get-LinuxInstallContext
$linuxUser = $context.User
$repo = $context.Repo

Invoke-WslRoot "apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y ca-certificates curl git"
Invoke-WslRoot "if ! command -v docker >/dev/null 2>&1; then curl -fsSL https://get.docker.com -o /tmp/get-docker.sh && sh /tmp/get-docker.sh && rm -f /tmp/get-docker.sh; fi"
if ($linuxUser -ne "root") {
    Invoke-WslRoot "usermod -aG docker '$linuxUser'"
}
Invoke-WslRoot "if command -v systemctl >/dev/null 2>&1 && systemctl is-system-running >/dev/null 2>&1; then systemctl enable --now docker; else service docker start; fi"

Invoke-WslRoot "if [ -d '$repo/.git' ]; then cd '$repo' && git fetch origin '$Branch' && git checkout '$Branch' && git merge --ff-only 'origin/$Branch'; elif [ -e '$repo' ]; then echo 'Install path exists but is not a Git checkout: $repo' >&2; exit 1; else git clone --branch '$Branch' --single-branch '$RepoUrl' '$repo'; fi"
if ($linuxUser -ne "root") {
    Invoke-WslRoot "chown -R '${linuxUser}:${linuxUser}' '$repo'"
}
$windowsHostIp = (& wsl.exe -d $Distro -u root -- bash -lc "ip route show default | awk '{print `$3; exit}'") -join ""
$windowsHostIp = ($windowsHostIp -replace "`0", "").Trim()
if (-not $windowsHostIp) {
    throw "Could not determine the Windows host address from WSL2"
}
Invoke-WslRoot "cd '$repo' && touch .env && if grep -q '^SYNTHGPU_OLLAMA_URL=' .env; then sed -i 's|^SYNTHGPU_OLLAMA_URL=.*|SYNTHGPU_OLLAMA_URL=http://$windowsHostIp`:11434|' .env; else printf '\nSYNTHGPU_OLLAMA_URL=http://$windowsHostIp`:11434\n' >> .env; fi"
Invoke-WslRoot "docker compose version >/dev/null"
Invoke-WslRoot "deadline=`$((`$(date +%s) + 30)); until docker version >/dev/null 2>&1; do if [ `$(date +%s) -ge `$deadline ]; then echo 'Docker Engine did not become ready within 30 seconds.' >&2; exit 1; fi; sleep 3; done"
Invoke-WslRoot "cd '$repo' && docker compose -p '$ProjectName' up -d --build"
Invoke-WslRoot "cd '$repo' && docker compose -p '$ProjectName' ps"

Write-Host "SynthGPU is running at http://localhost:8000"
Write-Host "Checkout: $Distro`:$repo"
if (-not $NoOpen) {
    Start-Process "http://localhost:8000"
}
