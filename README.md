# Iris Flower Classifier



[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

[![PyTorch 2.2+](https://img.shields.io/badge/PyTorch-2.2+-ee4c2c.svg)](https://pytorch.org/)

[![Status](https://img.shields.io/badge/Status-Ready%20to%20Execute-emerald.svg)](./STATUS.md)



Production Machine Learning Pretraining Pipeline synthesized by **ML-Train Studio**.

Trained completely from scratch on randomly initialized weights with cross-dataset tensor harmonization and zero data leakage.



---



## 📁 Repository Structure

```text

c3a24f9f-1ca2-4259-8ba5-5ce5216a0292/

├── EXECUTION_STATUS.json       # Live machine-readable pipeline execution tracker

├── STATUS.md                   # Human-readable execution state banner

├── config.yaml                 # Structured hyperparameters, loss weights & paths

├── datasets_manifest.json      # Dataset sources, licenses, and format standards

├── requirements.txt            # Reproducible pip environment dependencies

├── download_datasets.py        # [Stage 1] Ingests and extracts multi-dataset sources

├── eda_and_harmonization.py    # [Stage 2] Reconciles heterogeneous formats into tensors

├── train.py                    # [Stage 3] Pretraining loop with AMP mixed precision

├── evaluate_and_score.py       # [Stage 4] Validation scoring & clinical benchmark metrics

├── visualize.py                # [Stage 5] Multi-planar prediction overlays & loss curves

├── test_model.py               # Standalone interactive CLI test tool for live inference

├── run_pipeline.ps1            # Windows PowerShell one-click runner script

├── run_pipeline.sh             # Linux / macOS Bash one-click runner script

├── push_to_github.ps1          # One-click GitHub publisher script (PowerShell)

├── push_to_github.sh           # One-click GitHub publisher script (Bash)

└── .gitignore                  # Clean ML ignore rules (excludes raw caches & weights)

```

