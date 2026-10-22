#!/usr/bin/env bash
# One-Click Pipeline Execution Runner (POSIX Bash)
# Project: Conformer Speech Recognition
set -e

echo "=========================================================="
echo "  ML-TRAIN: Conformer Speech Recognition"
echo "  Multi-Dataset Pretraining Pipeline Execution"
echo "=========================================================="

echo "[1/4] Running Exploratory Data Analysis & Multi-Dataset Harmonization..."
python3 eda_and_harmonization.py

echo "[2/4] Executing Model Pretraining Loop from Scratch..."
python3 train.py

echo "[3/4] Evaluating Checkpoint Performance & Clinical Metrics..."
python3 evaluate_and_score.py

echo "[4/4] Rendering Orthogonal Slice Overlays & Inspection Plots..."
python3 visualize.py

echo "=========================================================="
echo "  SUCCESS: All Pipeline Stages Completed Cleanly!"
echo "=========================================================="
