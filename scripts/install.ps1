[CmdletBinding()]
param(
    [switch]$Uninstall,
    [switch]$NoOpen,
    [string]$Distro = "Ubuntu-24.04",
    [string]$RepoUrl = "https://github.com/bido75/synthGPU-OS.git",
    [string]$Branch = "main"
)

$ErrorActionPreference = "Stop"
$ProjectName = "synthgpu"
$ComposeFiles = "-f docker-compose.yml -f docker-compose.wsl.yml"

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

function Get-WslVersion {
    $output = ((& wsl.exe --list --verbose 2>$null) -join "`n") -replace "`0", ""
    $escapedDistro = [regex]::Escape($Distro)
    $match = [regex]::Match($output, "(?m)^\s*\*?\s*$escapedDistro\s+\S+\s+(\d+)\s*$")
    if (-not $match.Success) { return $null }
    return [int]$match.Groups[1].Value
}

function Invoke-WslRoot {
    param([string]$Command)
    # Windows PowerShell 5.1 rewrites quotes in native-command arguments.
    # Transport the script as base64 so Bash receives redirects, pipes, and
    # nested quotes exactly as authored.
    $encodedCommand = [Convert]::ToBase64String(
        [Text.Encoding]::UTF8.GetBytes($Command)
    )
    $previousErrorAction = $ErrorActionPreference
    try {
        # Native programs commonly use stderr for progress and informational
        # messages. Judge success by their process exit code instead.
        $ErrorActionPreference = "Continue"
        & wsl.exe -d $Distro -u root -- bash -lc "echo $encodedCommand|base64 -d|env PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin bash"
        $wslExitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $previousErrorAction
    }
    if ($wslExitCode -ne 0) {
        throw "WSL command failed with exit code $wslExitCode"
    }
}

function Invoke-WslRootOutput {
    param([string]$Command)
    $encodedCommand = [Convert]::ToBase64String(
        [Text.Encoding]::UTF8.GetBytes($Command)
    )
    $previousErrorAction = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        $output = & wsl.exe -d $Distro -u root -- bash -lc "echo $encodedCommand|base64 -d|env PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin bash"
        $wslExitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $previousErrorAction
    }
    if ($wslExitCode -ne 0) {
        throw "WSL command failed with exit code $wslExitCode"
    }
    return (($output -join "`n") -replace "`0", "").Trim()
}

function Get-LinuxInstallContext {
    # Avoid a pipe delimiter here: Windows PowerShell 5.1 can remove the
    # surrounding quotes when forwarding this command through wsl.exe.
    $command = 'u=$(getent passwd 1000 | cut -d: -f1); if [ -n "$u" ]; then h=$(getent passwd "$u" | cut -d: -f6); else u=root; h=/root; fi; printf "%s:%s" "$u" "$h"'
    $encodedCommand = [Convert]::ToBase64String(
        [Text.Encoding]::UTF8.GetBytes($command)
    )
    $context = & wsl.exe -d $Distro -u root -- bash -lc "echo $encodedCommand|base64 -d|env PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin bash"
    if ($LASTEXITCODE -ne 0 -or -not $context) {
        throw "Could not determine the Ubuntu install user"
    }
    $parts = (($context -join "") -replace "`0", "").Split(":", 2)
    if ($parts.Count -ne 2 -or -not $parts[0] -or -not $parts[1]) {
        throw "Ubuntu install user context was malformed: $context"
    }
    return @{ User = $parts[0]; Home = $parts[1]; Repo = "$($parts[1])/synthgpu" }
}

function Set-LocalhostPortProxy {
    param(
        [int]$ListenPort,
        [int]$ConnectPort,
        [string]$ConnectAddress
    )
    if ($ConnectAddress -notmatch '^\d+\.\d+\.\d+\.\d+$') {
        throw "Refusing to configure port proxy with invalid WSL2 IP: $ConnectAddress"
    }
    & netsh.exe interface portproxy delete v4tov4 listenport=$ListenPort listenaddress=127.0.0.1 2>$null | Out-Null
    & netsh.exe interface portproxy add v4tov4 listenport=$ListenPort listenaddress=127.0.0.1 connectport=$ConnectPort connectaddress=$ConnectAddress
    if ($LASTEXITCODE -ne 0) {
        throw "Could not configure localhost:$ListenPort -> $ConnectAddress`:$ConnectPort port proxy"
    }
}

if ($Uninstall) {
    $distros = Get-WslDistros
    if ($Distro -notin $distros) {
        Write-Host "Nothing to uninstall: WSL distro '$Distro' is not installed."
        exit 0
    }
    $context = Get-LinuxInstallContext
    $repo = $context.Repo
    Invoke-WslRoot "if [ -f '$repo/docker-compose.yml' ]; then cd '$repo' && docker compose $ComposeFiles -p '$ProjectName' down --rmi local --volumes --remove-orphans; else echo 'SynthGPU checkout not found; nothing to remove.'; fi"
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
Invoke-Checked "Setting WSL2 as the default" { wsl.exe --set-default-version 2 }
$distros = Get-WslDistros
if ($Distro -notin $distros) {
    Invoke-Checked "Installing $Distro" {
        wsl.exe --install --distribution $Distro --no-launch
    }
    Write-Host "The distro was installed. Initializing it now; Windows may show a first-run prompt."
}

if ((Get-WslVersion) -ne 2) {
    Invoke-Checked "Converting $Distro to WSL2" {
        wsl.exe --set-version $Distro 2
    }
}
Invoke-Checked "Setting $Distro as the default WSL distribution" {
    wsl.exe --set-default $Distro
}

Invoke-WslRoot "echo 'WSL2 alive'"
Invoke-WslRoot "if ip route show default | grep -q 'default via 10\.0\.0\.1 '; then ip route del default via 172.16.16.16 2>/dev/null || true; fi"
$context = Get-LinuxInstallContext
$linuxUser = $context.User
$repo = $context.Repo

Invoke-WslRoot "apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y ca-certificates curl git"
Invoke-WslRoot 'docker_path=$(command -v docker 2>/dev/null || true); if [ -z "$docker_path" ] || [[ "$docker_path" == /mnt/* ]]; then install -m 0755 -d /etc/apt/keyrings && curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc && chmod a+r /etc/apt/keyrings/docker.asc && . /etc/os-release && printf "deb [arch=%s signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu %s stable\n" "$(dpkg --print-architecture)" "$VERSION_CODENAME" > /etc/apt/sources.list.d/docker.list && apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin; fi'
if ($linuxUser -ne "root") {
    Invoke-WslRoot "usermod -aG docker '$linuxUser'"
}
Invoke-WslRoot "if command -v systemctl >/dev/null 2>&1 && systemctl is-system-running >/dev/null 2>&1; then systemctl enable --now docker; else service docker start; fi"

Invoke-WslRoot "if [ -d '$repo/.git' ]; then cd '$repo' && git fetch origin '$Branch' && git checkout '$Branch' && git merge --ff-only 'origin/$Branch'; elif [ -e '$repo' ]; then echo 'Install path exists but is not a Git checkout: $repo' >&2; exit 1; else git clone --branch '$Branch' --single-branch '$RepoUrl' '$repo'; fi"
if ($linuxUser -ne "root") {
    Invoke-WslRoot "chown -R '${linuxUser}:${linuxUser}' '$repo'"
}
Invoke-WslRoot "cd '$repo' && touch .env && if grep -q '^SYNTHGPU_OLLAMA_URL=' .env; then sed -i 's|^SYNTHGPU_OLLAMA_URL=.*|SYNTHGPU_OLLAMA_URL=http://localhost:11434|' .env; else printf '\nSYNTHGPU_OLLAMA_URL=http://localhost:11434\n' >> .env; fi"
Invoke-WslRoot "docker compose version >/dev/null"
Invoke-WslRoot "deadline=`$((`$(date +%s) + 30)); until docker version >/dev/null 2>&1; do if [ `$(date +%s) -ge `$deadline ]; then echo 'Docker Engine did not become ready within 30 seconds.' >&2; exit 1; fi; sleep 3; done"
Invoke-WslRoot "cd '$repo' && docker compose $ComposeFiles -p '$ProjectName' up -d --build"
Invoke-WslRoot "cd '$repo' && docker compose $ComposeFiles -p '$ProjectName' ps"
$wslIps = (Invoke-WslRootOutput "hostname -I") -split "\s+" | Where-Object {
    $_ -match '^\d+\.\d+\.\d+\.\d+$' -and
    $_ -notmatch '^127\.' -and
    $_ -notmatch '^169\.254\.' -and
    $_ -notmatch '^172\.17\.' -and
    $_ -notmatch '^172\.18\.'
}
$proxyTarget = @($wslIps | Select-Object -Skip 1 -First 1)[0]
if (-not $proxyTarget) {
    $proxyTarget = @($wslIps | Select-Object -First 1)[0]
}
if (-not $proxyTarget) {
    throw "Could not determine a WSL2 address for localhost forwarding"
}
Set-LocalhostPortProxy -ListenPort 8000 -ConnectPort 8000 -ConnectAddress $proxyTarget

Write-Host "SynthGPU is running at http://localhost:8000"
Write-Host "Checkout: $Distro`:$repo"
if (-not $NoOpen) {
    Start-Process "http://localhost:8000"
}
