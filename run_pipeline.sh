#!/usr/bin/env bash

# One-Click Pipeline Execution Runner (POSIX Bash)

# Project: Iris Flower Classifier

set -e

cd "$(dirname "$0")"



echo "=========================================================="

echo "  ML-TRAIN: Iris Flower Classifier"

echo "  Multi-Dataset Pretraining Pipeline Execution"

echo "=========================================================="



echo "[1/5] Ingesting & Verifying Multi-Dataset Sources..."

python3 download_datasets.py



echo "[2/5] Running Exploratory Data Analysis & Multi-Dataset Harmonization..."

python3 eda_and_harmonization.py



echo "[3/5] Executing Model Pretraining Loop from Scratch..."

python3 train.py



echo "[4/5] Evaluating Checkpoint Performance & Clinical Metrics..."

python3 evaluate_and_score.py



echo "[5/5] Rendering Inspection Overlays & Inspection Plots..."

python3 visualize.py



echo "=========================================================="

echo "  SUCCESS: All 5 Pipeline Stages Completed Cleanly!"

echo "=========================================================="

