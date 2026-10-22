"""
PRODUCTION MODEL INFERENCE SERVING ENGINE
Project: 5330b70a-8716-492c-aaf6-ea8dd6c9306b
Framework: Flask + PyTorch CPU/CUDA Production Runtime
Endpoint: http://127.0.0.1:5000
"""

import os
import sys
import time
import json
import logging
from pathlib import Path
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from flask import Flask, request, jsonify, render_template, send_from_directory

# Setup paths
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "best_model.pt"
MANIFEST_PATH = PROJECT_ROOT / "harmonized_manifest.json"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("serving_app")

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB max

# Load Harmonized Metadata
CLASSES = ["glioma","meningioma","notumor","pituitary"]
TARGET_SIZE = tuple([224,224])
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

if MANIFEST_PATH.exists():
    try:
        with open(MANIFEST_PATH, "r") as f:
            manifest_data = json.load(f)
            CLASSES = manifest_data.get("classes", CLASSES)
            TARGET_SIZE = tuple(manifest_data.get("target_size", TARGET_SIZE))
    except Exception as e:
        logger.warning(f"Could not parse manifest: {e}")

# Import Architecture from train.py
sys.path.insert(0, str(PROJECT_ROOT))
try:
    from train import ProductionVisionNet
except ImportError:
    import torch.nn as nn
    class ProductionVisionNet(nn.Module):
        def __init__(self, in_channels=3, num_classes=len(CLASSES), base_ch=32):
            super().__init__()
            self.stem = nn.Sequential(
                nn.Conv2d(in_channels, base_ch, 7, 2, 3, bias=False),
                nn.BatchNorm2d(base_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(3, 2, 1)
            )
            self.body = nn.Sequential(
                nn.Conv2d(base_ch, base_ch * 4, 3, 2, 1),
                nn.BatchNorm2d(base_ch * 4),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1))
            )
            self.classifier = nn.Linear(base_ch * 4, num_classes)
        def forward(self, x):
            return self.classifier(self.body(self.stem(x)).flatten(1))

# Initialize Model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL = None

def load_inference_model():
    global MODEL
    logger.info(f"Loading checkpoint for inference on {DEVICE}...")
    MODEL = ProductionVisionNet(in_channels=3, num_classes=len(CLASSES)).to(DEVICE)
    if CHECKPOINT_PATH.exists():
        try:
            ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
            state_dict = ckpt.get("model_state_dict", ckpt)
            MODEL.load_state_dict(state_dict, strict=False)
            logger.info("Successfully loaded checkpoint weights.")
        except Exception as err:
            logger.error(f"Failed to load checkpoint weights: {err}")
    else:
        logger.warning("Checkpoint not found at %s. Using initialized model weights.", CHECKPOINT_PATH)
    MODEL.eval()

load_inference_model()

def preprocess_image(image: Image.Image) -> torch.Tensor:
    img = image.convert("RGB").resize(TARGET_SIZE, Image.Resampling.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1)
    normalized = (tensor - MEAN) / STD
    return normalized.unsqueeze(0).to(DEVICE)

@app.route("/")
def index():
    return render_template("index.html", classes=CLASSES, device=str(DEVICE).upper())

@app.route("/health")
def health():
    return jsonify({
        "status": "healthy",
        "model": "ProductionVisionNet",
        "device": str(DEVICE),
        "num_classes": len(CLASSES),
        "classes": CLASSES,
        "checkpoint_exists": CHECKPOINT_PATH.exists()
    })

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded in form field 'file'"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    try:
        t0 = time.time()
        image = Image.open(file.stream)
        tensor = preprocess_image(image)

        with torch.no_grad():
            logits = MODEL(tensor)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]

        top_idx = int(np.argmax(probs))
        predicted_class = CLASSES[top_idx] if top_idx < len(CLASSES) else f"class_{top_idx}"
        confidence = float(probs[top_idx])
        latency_ms = round((time.time() - t0) * 1000, 2)

        prob_breakdown = {CLASSES[i] if i < len(CLASSES) else f"class_{i}": round(float(probs[i]), 4) for i in range(len(probs))}

        return jsonify({
            "success": True,
            "predicted_class": predicted_class,
            "confidence": round(confidence, 4),
            "confidence_pct": f"{confidence * 100:.1f}%",
            "probabilities": prob_breakdown,
            "latency_ms": latency_ms,
            "device": str(DEVICE).upper()
        })
    except Exception as e:
        logger.exception("Inference error")
        return jsonify({"error": str(e)}), 500

@app.route("/samples")
def list_samples():
    samples_dir = PROJECT_ROOT / "data"
    found_samples = []
    if samples_dir.exists():
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            for p in samples_dir.rglob(ext):
                if len(found_samples) >= 8:
                    break
                found_samples.append({
                    "name": p.name,
                    "rel_path": str(p.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                    "class_hint": p.parent.name
                })
    return jsonify({"samples": found_samples})

@app.route("/sample_file/<path:filepath>")
def get_sample_file(filepath):
    full_path = (PROJECT_ROOT / filepath).resolve()
    if not full_path.exists() or not str(full_path).startswith(str(PROJECT_ROOT)):
        return jsonify({"error": "File not found"}), 404
    return send_from_directory(str(full_path.parent), full_path.name)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"Serving Production Inference UI at http://127.0.0.1:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
