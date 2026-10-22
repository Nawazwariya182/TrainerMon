# TrainerMon Automated Dual-Mode Serving Launcher (PowerShell)
param (
    [int]$Port = 5000,
    [string]$ForceMode = "auto"
)

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "  TRAINERMON DIAGNOSTIC SERVING LAUNCHER (Dual-Mode)" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

# 1. Check Docker Desktop daemon availability
$dockerAvailable = $false
$dockerVersion = ""
if ($ForceMode -ne "native") {
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
    pip install -q -r requirements.txt
    $env:PORT = "$Port"
    $env:PYTHONUNBUFFERED = "1"
    $job = Start-Process -FilePath "python" -ArgumentList "app.py" -WorkingDirectory $PSScriptRoot -PassThru -WindowStyle Hidden
    Write-Host "Native Python process started (PID: $($job.Id)) on http://127.0.0.1:$Port" -ForegroundColor Green
}
