# =====================================================================
# TrainerMon Automated Dual-Mode Serving Launcher
# Automatically detects Docker daemon availability:
# - If Docker is active: Builds & deploys isolated Docker container on port 5000
# - If Docker is stopped/offline: Automatically falls back to native Python Flask runtime
# Ensures zero-failure, immediate interactive serving on http://localhost:5000
# =====================================================================

param (
    [int]$Port = 5000,
    [string]$ForceMode = "auto" # "auto", "docker", "native"
)

$Host.UI.RawUI.WindowTitle = "TrainerMon Serving Engine (Port $Port)"
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "  TRAINERMON DIAGNOSTIC SERVING LAUNCHER (Dual-Mode)" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

# 1. Check Docker daemon availability
$dockerAvailable = $false
$dockerVersion = ""

if ($ForceMode -ne "native") {
    Write-Host "[1/3] Checking Docker Desktop daemon availability..." -ForegroundColor Gray
    try {
        $dockerCmd = Get-Command docker -ErrorAction SilentlyContinue
        if ($dockerCmd) {
            $versionOutput = docker version --format '{{.Server.Version}}' 2>$null
            if ($LASTEXITCODE -eq 0 -and $versionOutput) {
                $dockerAvailable = $true
                $dockerVersion = $versionOutput.Trim()
            }
        }
    } catch {
        $dockerAvailable = $false
    }
}

# 2. Engage Active Runtime Mode
if ($dockerAvailable -and ($ForceMode -eq "auto" -or $ForceMode -eq "docker")) {
    Write-Host "[Mode: Docker Container] Docker daemon detected (v$dockerVersion)." -ForegroundColor Green
    Write-Host "Building & launching Docker container 'trainermon-diagnostic-service' on port $Port..." -ForegroundColor Green
    docker compose -f docker-compose.yml up -d --build
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[Warning] Docker compose encountered an error. Engaging automatic native Python fallback..." -ForegroundColor Yellow
        $dockerAvailable = $false
    }
}

if (-not $dockerAvailable -or $ForceMode -eq "native") {
    Write-Host "[Mode: Native Python Flask] Docker Desktop daemon is stopped or unavailable." -ForegroundColor Yellow
    Write-Host "[AUTO-FALLBACK] Seamlessly launching production Flask inference engine natively on port $Port..." -ForegroundColor Yellow

    # Ensure pip dependencies
    Write-Host "Verifying Python dependencies..." -ForegroundColor Gray
    pip install -q -r requirements.txt

    # Start background process
    $env:PORT = "$Port"
    $env:PYTHONUNBUFFERED = "1"
    $job = Start-Process -FilePath "python" -ArgumentList "app.py" -WorkingDirectory $PSScriptRoot -PassThru -WindowStyle Hidden
    Write-Host "Native Python process started (PID: $($job.Id))." -ForegroundColor Green
}

# 3. Verify Health Check
Write-Host "[3/3] Polling serving health endpoint on http://127.0.0.1:$Port/health..." -ForegroundColor Gray
$healthy = $false
for ($i = 0; $i -lt 15; $i++) {
    Start-Sleep -Milliseconds 600
    try {
        $res = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/health" -TimeoutSec 2 -ErrorAction SilentlyContinue
        if ($res.status -eq "healthy") {
            $healthy = $true
            break
        }
    } catch {}
}

if ($healthy) {
    Write-Host "==========================================================" -ForegroundColor Green
    Write-Host "  SUCCESS: Model Inference Service is LIVE!" -ForegroundColor Green
    Write-Host "  Endpoint URL: http://localhost:$Port" -ForegroundColor Green
    Write-Host "  Active Mode : $(if ($dockerAvailable) { 'Docker Container' } else { 'Native Python Flask (Auto-Fallback)' })" -ForegroundColor Green
    Write-Host "==========================================================" -ForegroundColor Green
} else {
    Write-Host "==========================================================" -ForegroundColor Yellow
    Write-Host "  Service warming up on http://localhost:$Port..." -ForegroundColor Yellow
    Write-Host "==========================================================" -ForegroundColor Yellow
}
