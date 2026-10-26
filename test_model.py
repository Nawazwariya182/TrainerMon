"""
NLP SENTIMENT STANDALONE INFERENCE TESTER
"""

import sys
import re
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from train import ProductionTextNet

CLASSES = ["negative","positive"]
CKPT_PATH = Path(__file__).resolve().parent / "checkpoints" / "best_model.pt"
VOCAB_PATH = Path(__file__).resolve().parent / "vocab.json"

def test_inference(review_text: str = "This movie was an absolute masterpiece with thrilling visuals!") -> dict:
    if not CKPT_PATH.exists():
        return {"status": "ERROR", "message": "Checkpoint best_model.pt not found. Run train.py first."}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(CKPT_PATH, map_location=device)

    with open(VOCAB_PATH, "r", encoding="utf-8") as vf:
        vocab = json.load(vf)

    model = ProductionTextNet(vocab_size=len(vocab), embed_dim=128, hidden_dim=128, num_classes=2, dropout=0.0).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    tokens = re.findall(r"[a-z0-9']+|[!?,.]", review_text.lower())[:128]
    ids = [vocab.get(w, 1) for w in tokens]
    if len(ids) < 128:
        ids = ids + [0] * (128 - len(ids))

    tensor_in = torch.tensor([ids], dtype=torch.long, device=device)

    with torch.no_grad():
        out = model(tensor_in)
        probs = F.softmax(out, dim=1).cpu().numpy()[0]
        pred_idx = int(np.argmax(probs))
        conf = float(probs[pred_idx])

    res = {
        "status": "SUCCESS",
        "domain": "nlp_text",
        "input_text": review_text,
        "predicted_class": CLASSES[pred_idx],
        "confidence": round(conf, 4),
        "confidence_pct": f"{conf * 100:.2f}%",
        "probabilities": {CLASSES[i]: round(float(probs[i]), 4) for i in range(2)},
        "device": str(device)
    }
    print(json.dumps(res, indent=2))
    return res


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, default="This movie was an absolute masterpiece with thrilling visuals!", help="Review text to classify")
    args = parser.parse_args()
    test_inference(args.text)
    safe_exit(0)
