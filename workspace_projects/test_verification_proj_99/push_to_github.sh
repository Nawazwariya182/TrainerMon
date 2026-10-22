#!/usr/bin/env bash
# One-Click GitHub Push Script (POSIX Bash)
# Project: Brain Tumor Multi-Modal MRI Segmentation
set -e

REPO_URL="$1"
BRANCH="${2:-main}"
COMMIT_MSG="${3:-Initial commit: Production pretraining pipeline for Brain Tumor Multi-Modal MRI Segmentation}"

echo "=========================================================="
echo "  ML-TRAIN: Push Workspace to GitHub"
echo "=========================================================="

if [ ! -d ".git" ]; then
    echo "[1/4] Initializing Git repository..."
    git init
    git branch -M "$BRANCH"
else
    echo "[1/4] Git repository already initialized."
fi

git config user.name "ML-Train Studio"
git config user.email "studio@mltrain.local"

echo "[2/4] Staging files (respecting .gitignore)..."
git add -A

echo "[3/4] Committing code artifacts..."
git commit -m "$COMMIT_MSG" || echo "  Working tree clean or nothing new to commit."

if [ -n "$REPO_URL" ]; then
    echo "[4/4] Pushing to remote: $REPO_URL ($BRANCH branch)..."
    git remote remove origin 2>/dev/null || true
    git remote add origin "$REPO_URL"
    git push -u origin "$BRANCH" --force
    echo "=========================================================="
    echo "  SUCCESS: Pushed to GitHub repository ($REPO_URL)!"
    echo "=========================================================="
else
    echo "[4/4] No RepoUrl supplied. Local repository committed successfully!"
    echo "  To push to GitHub, run:"
    echo "    ./push_to_github.sh 'https://github.com/your-user/your-repo.git'"
fi
