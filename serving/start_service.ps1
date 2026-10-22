# PowerShell Service Launcher with Auto-Fallback
param([int]$Port = 5000, [string]$ForceMode = "")

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "  TRAINERMON DIAGNOSTIC SERVING LAUNCHER (Dual-Mode)" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

$dockerAvailable = $false
try {
    $null = docker version 2>&1
    if ($LASTEXITCODE -eq 0) { $dockerAvailable = $true }
} catch { $dockerAvailable = $false }

if ($dockerAvailable -and $ForceMode -ne "native") {
    Write-Host "[Mode: Docker Container] Launching container on port $Port..." -ForegroundColor Green
    docker compose -f docker-compose.yml up -d --build
    Write-Host "Serving Container started on http://localhost:$Port" -ForegroundColor Green
} else {
    Write-Host "[Mode: Native Python Flask] Launching on port $Port..." -ForegroundColor Yellow
    pip install -q -r requirements.txt
    $env:PORT = "$Port"
    $env:PYTHONUNBUFFERED = "1"
    $job = Start-Process -FilePath "python" -ArgumentList "app.py" -WorkingDirectory $PSScriptRoot -PassThru -WindowStyle Hidden
    Write-Host "Native Python process started (PID: $($job.Id)) on http://127.0.0.1:$Port" -ForegroundColor Green
}
