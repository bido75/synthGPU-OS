[CmdletBinding()]
param(
    [ValidateRange(5, 300)]
    [int]$TimeoutSeconds = 30
)

$ErrorActionPreference = "Stop"

function Get-DockerServerVersion {
    $startInfo = [System.Diagnostics.ProcessStartInfo]::new()
    $startInfo.FileName = "docker"
    $startInfo.Arguments = 'version --format "{{.Server.Version}}"'
    $startInfo.UseShellExecute = $false
    $startInfo.RedirectStandardOutput = $true
    $startInfo.RedirectStandardError = $true
    $startInfo.CreateNoWindow = $true

    try {
        $process = [System.Diagnostics.Process]::Start($startInfo)
    } catch {
        return $null
    }
    if (-not $process.WaitForExit(5000)) {
        $process.Kill($true)
        $process.WaitForExit()
        return $null
    }
    if ($process.ExitCode -ne 0) { return $null }
    return $process.StandardOutput.ReadToEnd().Trim()
}

$deadline = (Get-Date).AddSeconds($TimeoutSeconds)
do {
    $version = Get-DockerServerVersion
    if ($version) {
        Write-Output "Docker engine: $version"
        exit 0
    }
    Start-Sleep -Seconds 3
} while ((Get-Date) -lt $deadline)

Write-Warning "Docker engine did not respond within $TimeoutSeconds seconds."
Write-Warning "Run 'wsl --version' and verify the selected Docker runtime is running."
exit 1
