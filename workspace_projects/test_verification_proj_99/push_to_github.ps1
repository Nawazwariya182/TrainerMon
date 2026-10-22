# One-Click GitHub Push Script (Windows PowerShell)
# Project: Brain Tumor Multi-Modal MRI Segmentation
param(
    [Parameter(Mandatory=$false)]
    [string]$RepoUrl = "",
    [Parameter(Mandatory=$false)]
    [string]$Branch = "main",
    [Parameter(Mandatory=$false)]
    [string]$CommitMessage = "Initial commit: Production pretraining pipeline for Brain Tumor Multi-Modal MRI Segmentation"
)

$ErrorActionPreference = "Stop"
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "  ML-TRAIN: Push Workspace to GitHub" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

# 1. Initialize Git if needed
if (-not (Test-Path ".git")) {
    Write-Host "[1/4] Initializing Git repository..." -ForegroundColor Yellow
    git init
    git branch -M $Branch
} else {
    Write-Host "[1/4] Git repository already initialized." -ForegroundColor Green
}

# 2. Configure identity if not set
git config user.name "ML-Train Studio"
git config user.email "studio@mltrain.local"

# 3. Stage files
Write-Host "[2/4] Staging files (respecting .gitignore)..." -ForegroundColor Yellow
git add -A

# 4. Commit
Write-Host "[3/4] Committing code artifacts..." -ForegroundColor Yellow
try {
    git commit -m $CommitMessage -q
} catch {
    Write-Host "  Working tree clean or nothing new to commit." -ForegroundColor Gray
}

# 5. Push if RepoUrl provided
if ($RepoUrl -ne "") {
    Write-Host "[4/4] Pushing to remote: $RepoUrl ($Branch branch)..." -ForegroundColor Yellow
    git remote remove origin 2>$null
    git remote add origin $RepoUrl
    git push -u origin $Branch --force
    Write-Host "==========================================================" -ForegroundColor Green
    Write-Host "  SUCCESS: Pushed to GitHub repository ($RepoUrl)!" -ForegroundColor Green
    Write-Host "==========================================================" -ForegroundColor Green
} else {
    Write-Host "[4/4] No -RepoUrl supplied. Local repository committed successfully!" -ForegroundColor Green
    Write-Host "  To push to GitHub, run:" -ForegroundColor Cyan
    Write-Host "    .\push_to_github.ps1 -RepoUrl 'https://github.com/your-user/your-repo.git'" -ForegroundColor Cyan
}
