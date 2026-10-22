# One-Click Pipeline Execution Runner (Windows PowerShell)
# Project: Conformer Speech Recognition
$ErrorActionPreference = "Stop"

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "  ML-TRAIN: Conformer Speech Recognition" -ForegroundColor Cyan
Write-Host "  Multi-Dataset Pretraining Pipeline Execution" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

# Step 1: Exploratory Data Analysis & Multi-Dataset Harmonization Check
Write-Host "[1/4] Running Exploratory Data Analysis & Multi-Dataset Harmonization..." -ForegroundColor Yellow
python eda_and_harmonization.py

# Step 2: Pretraining from Scratch
Write-Host "[2/4] Executing Model Pretraining Loop from Scratch..." -ForegroundColor Yellow
python train.py

# Step 3: Quantitative Scoring & Validation
Write-Host "[3/4] Evaluating Checkpoint Performance & Clinical Metrics..." -ForegroundColor Yellow
python evaluate_and_score.py

# Step 4: Visual Slices & Convergence Generation
Write-Host "[4/4] Rendering Orthogonal Slice Overlays & Inspection Plots..." -ForegroundColor Yellow
python visualize.py

Write-Host "==========================================================" -ForegroundColor Green
Write-Host "  SUCCESS: All Pipeline Stages Completed Cleanly!" -ForegroundColor Green
Write-Host "  Model Checkpoint: ./checkpoints/best_model.pt" -ForegroundColor Green
Write-Host "  Evaluation Report: ./evaluation_metrics.json" -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Green
