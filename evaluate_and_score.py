"""
MOVIE REVIEW SENTIMENT QUANTITATIVE VALIDATION AND QUALITY GATE SCORER
"""

import sys
import json
import torch
import numpy as np
from pathlib import Path
from train import ProductionTextNet

HARMONIZED_DIR = Path(__file__).resolve().parent / "data" / "harmonized"
CHECKPOINTS_DIR = Path(__file__).resolve().parent / "checkpoints"
CLASSES = ["negative","positive"]

def main():
    print("=================================================================")
    print("Stage 4: Quantitative Validation & Quality Gate Evaluation")
    print("=================================================================")

    ckpt_path = CHECKPOINTS_DIR / "best_model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint {ckpt_path} not found. Run train.py first.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device)

    vocab_size = ckpt.get("vocab_size", 20000)
    model = ProductionTextNet(vocab_size=vocab_size, embed_dim=128, hidden_dim=128, num_classes=2, dropout=0.0).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    test_x = np.load(HARMONIZED_DIR / "test_x.npy")
    test_y = np.load(HARMONIZED_DIR / "test_y.npy")

    batch_size = 128
    all_preds = []
    with torch.no_grad():
        for i in range(0, len(test_x), batch_size):
            bx = torch.from_numpy(test_x[i:i+batch_size]).to(device)
            out = model(bx)
            preds = out.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)

    all_preds = np.array(all_preds)
    accuracy = float(np.mean(all_preds == test_y) * 100.0)

    # Precision, Recall, F1
    tp = np.sum((all_preds == 1) & (test_y == 1))
    fp = np.sum((all_preds == 1) & (test_y == 0))
    fn = np.sum((all_preds == 0) & (test_y == 1))
    precision = float(tp / max(1, tp + fp))
    recall = float(tp / max(1, tp + fn))
    f1 = float(2 * precision * recall / max(1e-6, precision + recall))

    passed = accuracy >= 85.0
    status_str = "PASSED" if passed else "FAILED"

    print("-----------------------------------------------------------------")
    print(f"Validation Accuracy: {accuracy:.2f}% | F1-Score: {f1:.4f} (Threshold: >= 85.00%)")
    print(f"[QUALITY_GATE] {status_str}: Accuracy = {accuracy:.2f}%")
    print("-----------------------------------------------------------------")

    metrics = {
        "accuracy": round(accuracy, 2),
        "overall_accuracy": round(accuracy, 2),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "quality_gate_passed": passed
    }

    with open(CHECKPOINTS_DIR.parent / "evaluation_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    safe_exit(0 if passed else 1)


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
