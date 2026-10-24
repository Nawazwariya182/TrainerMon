"""
QUANTITATIVE EVALUATION AND ACCURACY SCORING ENGINE (REAL DATA)
Project: Machine Learning Project
Evaluates the trained model checkpoint against real held-out validation samples.
Computes real Accuracy, Precision, Recall, F1-Score, and Confusion Matrix.
"""

import os
import sys
import json
import logging
from pathlib import Path
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from train import ProductionVisionNet, RealDatasetLoader, update_execution_status, print_progress_bar

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evaluation_scoring")

def evaluate_model_performance(checkpoint_path: str = "./checkpoints/best_model.pt") -> dict:
    ckpt_file = Path(__file__).resolve().parent / checkpoint_path
    logger.info("Loading trained model checkpoint from: %s", ckpt_file)
    update_execution_status("evaluate_and_score", 10.0, "Loading checkpoint and validation split", "IN_PROGRESS")

    if not ckpt_file.exists():
        raise RuntimeError(f"Missing checkpoint file {ckpt_file}. Run train.py first.")

    manifest_file = Path(__file__).resolve().parent / "harmonized_manifest.json"
    with open(manifest_file, "r") as f:
        manifest = json.load(f)

    val_samples = manifest.get("val_samples", [])
    classes = manifest.get("classes", ["class_0", "class_1"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_file, map_location=device)
    num_classes = ckpt.get("num_classes", len(classes))
    target_size = tuple(ckpt.get("target_size") or manifest.get("target_size", [128, 128]))
    mean = ckpt.get("mean") or manifest.get("channel_mean", [0.485, 0.456, 0.406])
    std = ckpt.get("std") or manifest.get("channel_std", [0.229, 0.224, 0.225])

    model = ProductionVisionNet(in_channels=3, num_classes=num_classes, base_ch=48).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    val_ds = RealDatasetLoader(val_samples, target_size=target_size, mean=mean, std=std, is_train=False)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)

    all_preds = []
    all_targets = []
    all_probs = []

    total_batches = max(1, len(val_loader))
    with torch.no_grad():
        for step, (x, y) in enumerate(val_loader):
            x = x.to(device)
            logits = model(x)
            probs = F.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            all_preds.extend(preds.tolist())
            all_targets.extend(y.numpy().tolist())
            all_probs.extend(probs.tolist())

            pct = 15.0 + ((step + 1) / total_batches) * 75.0
            print_progress_bar(step + 1, total_batches, prefix="  Evaluating Batches", suffix=f"({step + 1}/{total_batches})")
            update_execution_status("evaluate_and_score", pct, f"Evaluated {step + 1}/{total_batches} batches", "IN_PROGRESS")

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)

    accuracy = float(np.mean(y_true == y_pred)) if len(y_true) > 0 else 0.0

    class_metrics = {}
    for idx, cname in enumerate(classes):
        tp = int(np.sum((y_true == idx) & (y_pred == idx)))
        fp = int(np.sum((y_true != idx) & (y_pred == idx)))
        fn = int(np.sum((y_true == idx) & (y_pred != idx)))
        tn = int(np.sum((y_true != idx) & (y_pred != idx)))

        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        class_metrics[cname] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "support": int(np.sum(y_true == idx))
        }

    macro_f1 = float(np.mean([m["f1_score"] for m in class_metrics.values()])) if class_metrics else 0.0

    scores = {
        "checkpoint": str(ckpt_file),
        "total_validation_samples": len(y_true),
        "overall_accuracy": round(accuracy, 4),
        "accuracy": round(accuracy, 4),
        "macro_f1_score": round(macro_f1, 4),
        "class_breakdown": class_metrics,
        "quality_gate_passed": bool(accuracy >= 0.75),
        "target_threshold": 0.75,
        "evaluation_status": "VALIDATED_ON_REAL_DATA"
    }

    logger.info("Evaluation Complete. Accuracy: %.2f%% | Macro F1: %.4f", accuracy * 100, macro_f1)
    with open("evaluation_metrics.json", "w") as f:
        json.dump(scores, f, indent=2)
    with open("metrics.json", "w") as f:
        json.dump(scores, f, indent=2)

    status_label = "COMPLETED" if accuracy >= 0.75 else "QUALITY_GATE_FAILED"
    update_execution_status("evaluate_and_score", 100.0, f"Validation: Accuracy {accuracy * 100:.1f}%, F1 {macro_f1:.3f} (Passed: {accuracy >= 0.75})", status_label)
    return scores

if __name__ == "__main__":
    metrics = evaluate_model_performance()
    acc = metrics.get("overall_accuracy", 0.0)
    print("=" * 60)
    if acc < 0.75:
        print(f"[QUALITY_GATE] FAILED: Validation Accuracy {acc * 100:.2f}% is below required 75.00% baseline threshold.")
        print("[QUALITY_GATE] Autonomous tuning required: Tuning hyperparameters and training duration.")
        sys.exit(2)
    else:
        print(f"[QUALITY_GATE] PASSED: Validation Accuracy {acc * 100:.2f}% satisfies the pretraining benchmark (Target >= 75.00%).")
        print("[QUALITY_GATE] Model checkpoint validated and ready for export (.pth).")
        sys.exit(0)
