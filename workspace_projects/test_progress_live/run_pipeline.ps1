# One-Click Pipeline Execution Runner (Windows PowerShell)
# Project: Brain Tumor Live Progress
$ErrorActionPreference = "Stop"

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "  ML-TRAIN: Brain Tumor Live Progress" -ForegroundColor Cyan
Write-Host "  Multi-Dataset Pretraining Pipeline Execution" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

# Step 1: Download & Verify Datasets
Write-Host "[1/5] Ingesting & Verifying Multi-Dataset Sources..." -ForegroundColor Yellow
python download_datasets.py

# Step 2: Exploratory Data Analysis & Multi-Dataset Harmonization Check
Write-Host "[2/5] Running Exploratory Data Analysis & Multi-Dataset Harmonization..." -ForegroundColor Yellow
python eda_and_harmonization.py

# Step 3: Pretraining from Scratch
Write-Host "[3/5] Executing Model Pretraining Loop from Scratch..." -ForegroundColor Yellow
python train.py

# Step 4: Quantitative Scoring & Validation
Write-Host "[4/5] Evaluating Checkpoint Performance & Clinical Metrics..." -ForegroundColor Yellow
python evaluate_and_score.py

# Step 5: Visual Slices & Convergence Generation
Write-Host "[5/5] Rendering Inspection Overlays & Inspection Plots..." -ForegroundColor Yellow
python visualize.py

Write-Host "==========================================================" -ForegroundColor Green
Write-Host "  SUCCESS: All 5 Pipeline Stages Completed Cleanly!" -ForegroundColor Green
Write-Host "  Model Checkpoint: ./checkpoints/best_model.pt" -ForegroundColor Green
Write-Host "  Evaluation Report: ./evaluation_metrics.json" -ForegroundColor Green
Write-Host "  Execution Status: Updated in ./EXECUTION_STATUS.json" -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Green
