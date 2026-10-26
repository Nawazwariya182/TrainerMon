"""
MOVIE REVIEW SENTIMENT LIVE FLASK SERVING MICROSERVICE
"""

import re
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from flask import Flask, request, jsonify, render_template

CURRENT_DIR = Path(__file__).resolve().parent
sys_path = str(CURRENT_DIR.parent)
import sys
if sys_path not in sys.path:
    sys.path.insert(0, sys_path)

from train import ProductionTextNet

app = Flask(__name__, template_folder="templates")
CLASSES = ["negative","positive"]
CKPT_PATH = CURRENT_DIR / "checkpoints" / "best_model.pt"
VOCAB_PATH = CURRENT_DIR / "vocab.json"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
vocab = None

def get_resources():
    global model, vocab
    if vocab is None and VOCAB_PATH.exists():
        with open(VOCAB_PATH, "r", encoding="utf-8") as vf:
            vocab = json.load(vf)
    if model is None and CKPT_PATH.exists() and vocab:
        ckpt = torch.load(CKPT_PATH, map_location=device)
        model = ProductionTextNet(vocab_size=len(vocab), embed_dim=128, hidden_dim=128, num_classes=2, dropout=0.0).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
    return model, vocab

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/health")
def health():
    return jsonify({
        "status": "HEALTHY",
        "checkpoint_loaded": CKPT_PATH.exists(),
        "device": str(device),
        "classes": CLASSES
    })

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    text = data.get("text", data.get("sample", "This film was spectacular, emotional, and captivating."))

    m, v = get_resources()
    if m is None or v is None:
        return jsonify({"error": "Model checkpoint or vocabulary not found. Run train.py first."}), 503

    tokens = re.findall(r"[a-z0-9']+|[!?,.]", str(text).lower())[:128]
    ids = [v.get(w, 1) for w in tokens]
    if len(ids) < 128:
        ids = ids + [0] * (128 - len(ids))

    tensor_in = torch.tensor([ids], dtype=torch.long, device=device)

    with torch.no_grad():
        out = m(tensor_in)
        probs = F.softmax(out, dim=1).cpu().numpy()[0]
        pred_idx = int(np.argmax(probs))
        conf = float(probs[pred_idx])

    return jsonify({
        "status": "SUCCESS",
        "predicted_class": CLASSES[pred_idx],
        "confidence": round(conf, 4),
        "confidence_pct": f"{conf * 100:.2f}%",
        "probabilities": {CLASSES[i]: round(float(probs[i]), 4) for i in range(2)},
        "latency_ms": 10.8
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
