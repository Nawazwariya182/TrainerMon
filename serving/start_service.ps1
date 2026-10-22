# TrainerMon Dual-Mode Serving Launcher (Docker + Native Python Fallback)
$Port = 5000
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "  TRAINERMON DIAGNOSTIC SERVING LAUNCHER" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

# Check if Docker Desktop daemon is running
$dockerAvailable = $false
try {
    $dockerInfo = docker info 2>&1
    if ($LASTEXITCODE -eq 0) {
        $dockerAvailable = $true
    }
} catch {
    $dockerAvailable = $false
}

if ($dockerAvailable) {
    Write-Host "[Mode: Docker] Docker daemon is online. Launching container on port $Port..." -ForegroundColor Green
    docker compose -f docker-compose.yml up -d --build
    Write-Host "Serving Container started: http://localhost:$Port" -ForegroundColor Green
} else {
    Write-Host "[Mode: Native Flask] Docker daemon unavailable. Starting background Python Flask server on port $Port..." -ForegroundColor Yellow
    # Ensure dependencies
    pip install -r requirements.txt
    $job = Start-Process -FilePath "python" -ArgumentList "app.py" -WorkingDirectory $PSScriptRoot -PassThru -WindowStyle Hidden
    Write-Host "Flask Process started (PID: $($job.Id)) on http://127.0.0.1:$Port" -ForegroundColor Green
}
