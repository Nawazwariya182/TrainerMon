"""
NLP SENTIMENT CONVERGENCE AND METRICS VISUALIZATION
"""

import os
import sys
import json
import matplotlib.pyplot as plt
from pathlib import Path

VIZ_DIR = Path(__file__).resolve().parent / "visualizations"
VIZ_DIR.mkdir(parents=True, exist_ok=True)
ROOT_DIR = Path(__file__).resolve().parent

def main():
    print("=================================================================")
    print("Stage 5: Generating Diagnostic Visualizations")
    print("=================================================================")

    hist_file = ROOT_DIR / "checkpoints" / "training_history.json"
    if hist_file.exists():
        with open(hist_file, "r") as f:
            hist = json.load(f)
        fig, ax1 = plt.subplots(figsize=(8, 4.5))
        ax1.plot(hist["train_loss"], label="Train Loss", color="#4f46e5", lw=2)
        ax1.plot(hist["val_loss"], label="Val Loss", color="#ec4899", lw=2, linestyle="--")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Cross Entropy Loss")
        ax1.set_title("BiLSTM Sentiment Model Pretraining Convergence", fontsize=12, fontweight="bold")
        ax1.legend(loc="upper right")
        ax1.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(VIZ_DIR / "training_convergence.png", dpi=150)
        plt.close()
        print("Saved visualizations/training_convergence.png")
    safe_exit(0)


def safe_exit(code=0):
    import sys
    sys.stdout.flush()
    sys.stderr.flush()
    try:
        import ctypes
        k = ctypes.windll.kernel32
        k.GetCurrentProcess.restype = ctypes.c_void_p
        k.TerminateProcess.argtypes = [ctypes.c_void_p, ctypes.c_uint]
        k.TerminateProcess.restype = ctypes.c_int
        h = k.GetCurrentProcess()
        k.TerminateProcess(h, int(code))
    except Exception:
        import os
        os._exit(int(code))


if __name__ == "__main__":
    main()
