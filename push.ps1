# SynthGPU — Quick Push Script
# Usage: .\push.ps1 "your commit message"
# Usage: .\push.ps1              (uses auto-generated message with timestamp)

param(
    [string]$msg = ""
)

$projectRoot = $PSScriptRoot
Set-Location $projectRoot

# Auto-generate message if none provided
if (-not $msg) {
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm"
    $msg = "update: $timestamp"
}

Write-Host ""
Write-Host "  SynthGPU Push" -ForegroundColor Cyan
Write-Host "  Commit: $msg" -ForegroundColor Gray
Write-Host ""

# Check for changes
$status = git status --porcelain
if (-not $status) {
    Write-Host "  Nothing to commit — working tree clean." -ForegroundColor Yellow
    exit 0
}

# Show what's changing
Write-Host "  Changed files:" -ForegroundColor DarkCyan
git status --short | ForEach-Object { Write-Host "    $_" -ForegroundColor Gray }
Write-Host ""

# Stage, commit, push
git add .
if ($LASTEXITCODE -ne 0) { Write-Host "  git add failed" -ForegroundColor Red; exit 1 }

git commit -m $msg
if ($LASTEXITCODE -ne 0) { Write-Host "  git commit failed" -ForegroundColor Red; exit 1 }

git push
if ($LASTEXITCODE -ne 0) { Write-Host "  git push failed — check your remote/credentials" -ForegroundColor Red; exit 1 }

Write-Host ""
Write-Host "  Pushed successfully. CI/CD pipeline now running on GitHub." -ForegroundColor Green
Write-Host ""
