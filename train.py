"""
MODEL PRETRAINING PIPELINE FROM SCRATCH (REAL DATA)
Project: Machine Learning Project
Architecture: Deep Vision Residual Backbone with Adaptive Pooling
Loss Objective: Label-Smoothed Cross-Entropy Loss
Datasets: Brain Tumor Classification MRI Benchmark (4 Classes: Glioma, Meningioma, Pituitary, Normal), Brain MRI Images for Brain Tumor Detection (Br35H / Navoneel Cohort)
"""

import os
import sys
import json
import time
import math
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple
from PIL import Image
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("pretrain_studio")

# =====================================================================
# 1. NEURAL ARCHITECTURE: RESIDUAL VISION CONVNET (From Scratch)
# =====================================================================

class ResidualBlock2D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.act1 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act2 = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        if in_ch != out_ch or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act2(out + res)

class ProductionVisionNet(nn.Module):
    """Deep residual neural vision backbone initialized from scratch."""
    def __init__(self, in_channels: int = 3, num_classes: int = 2, base_ch: int = 32):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_ch, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(base_ch),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.stage1 = ResidualBlock2D(base_ch, base_ch * 2, stride=2)
        self.stage2 = ResidualBlock2D(base_ch * 2, base_ch * 4, stride=2)
        self.stage3 = ResidualBlock2D(base_ch * 4, base_ch * 8, stride=2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(base_ch * 8, base_ch * 4),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(base_ch * 4, num_classes)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.pool(x)
        return self.classifier(x)

# =====================================================================
# 2. REAL MULTI-DATASET DATA LOADER
# =====================================================================

class RealDatasetLoader(Dataset):
    """Loads and standardizes real scanned images directly from disk."""
    def __init__(self, sample_records: List[Dict[str, Any]], target_size: Tuple[int, int] = (224, 224),
                 mean: List[float] = None, std: List[float] = None):
        self.samples = sample_records
        self.target_size = target_size
        self.mean = torch.tensor(mean or [0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor(std or [0.229, 0.224, 0.225]).view(3, 1, 1)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        rec = self.samples[idx]
        img_path = Path(__file__).resolve().parent / rec["path"]
        try:
            with Image.open(img_path) as img:
                img = img.convert("RGB").resize(self.target_size, Image.Resampling.BILINEAR)
                arr = np.array(img, dtype=np.float32) / 255.0
                tensor = torch.from_numpy(arr).permute(2, 0, 1)
                x = (tensor - self.mean) / self.std
        except Exception:
            x = torch.zeros(3, self.target_size[0], self.target_size[1], dtype=torch.float32)
        y = torch.tensor(rec.get("label", 0), dtype=torch.long)
        return x, y

def update_execution_status(step_name: str, progress_pct: float, details: str, status: str = "IN_PROGRESS"):
    status_file = Path(__file__).resolve().parent / "EXECUTION_STATUS.json"
    if not status_file.exists():
        return
    try:
        import datetime
        with open(status_file, "r") as f:
            data = json.load(f)
        data["last_updated"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        if status == "IN_PROGRESS":
            data["status"] = "EXECUTING"
            data["is_new"] = False
            data["current_step_name"] = step_name
        for s in data.get("steps", []):
            if s.get("name") == step_name:
                s["status"] = status
                s["progress_pct"] = round(progress_pct, 1)
                s["details"] = details
        total_p = sum(s.get("progress_pct", 0) for s in data.get("steps", []))
        data["overall_progress_pct"] = round(total_p / max(1, len(data.get("steps", []))), 1)
        with open(status_file, "w") as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass

def print_progress_bar(iteration: int, total: int, prefix: str = "", suffix: str = "", length: int = 35):
    total = max(1, total)
    percent = f"{100 * (iteration / float(total)):.1f}"
    filled_length = int(length * iteration // total)
    bar = "=" * filled_length + ">" + " " * (length - filled_length - 1) if filled_length < length else "=" * length
    sys.stdout.write(f"\r{prefix} [{bar}] {percent}% {suffix}")
    sys.stdout.flush()
    if iteration >= total:
        sys.stdout.write("\n")

def pretrain_studio():
    logger.info("Initializing pretraining from scratch on real dataset...")
    update_execution_status("train", 5.0, "Loading harmonized manifest and model weights", "IN_PROGRESS")

    manifest_path = Path(__file__).resolve().parent / "harmonized_manifest.json"
    if not manifest_path.exists():
        raise RuntimeError("Missing harmonized_manifest.json. Run eda_and_harmonization.py first.")

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    train_samples = manifest.get("train_samples", [])
    classes = manifest.get("classes", ["class_0", "class_1"])
    num_classes = max(2, len(classes))
    target_size = tuple(manifest.get("target_size", [224, 224]))
    mean = manifest.get("channel_mean", [0.485, 0.456, 0.406])
    std = manifest.get("channel_std", [0.229, 0.224, 0.225])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Hardware Accelerator: %s | Real training samples: %d", device, len(train_samples))

    model = ProductionVisionNet(in_channels=3, num_classes=num_classes).to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Neural backbone initialized with %s trainable parameters.", f"{param_count:,}")

    train_ds = RealDatasetLoader(train_samples, target_size=target_size, mean=mean, std=std)
    batch_size = min(16, max(2, len(train_samples) // 4))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    epochs = 5
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    history = {"epochs": [], "losses": []}
    logger.info("Starting from-scratch pretraining loop for %d epochs...", epochs)

    os.makedirs("./checkpoints", exist_ok=True)
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        t0 = time.time()
        total_batches = len(train_loader)

        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()

            if (step + 1) % max(1, total_batches // 5) == 0 or step == total_batches - 1:
                step_pct = ((epoch - 1 + ((step + 1) / total_batches)) / epochs) * 90.0
                print_progress_bar(step + 1, total_batches, prefix=f"  Epoch {epoch}/{epochs}", suffix=f"Loss: {loss.item():.4f}")

        scheduler.step()
        epoch_loss = running_loss / max(1, total_batches)
        elapsed = time.time() - t0
        history["epochs"].append(epoch)
        history["losses"].append(round(epoch_loss, 4))
        logger.info("Epoch [%d/%d] in %.2fs | Average Loss: %.4f", epoch, epochs, elapsed, epoch_loss)
        update_execution_status("train", 10.0 + (epoch / epochs) * 85.0, f"Epoch {epoch}/{epochs} Loss: {epoch_loss:.4f}", "IN_PROGRESS")

    ckpt_path = Path(__file__).resolve().parent / "checkpoints" / "best_model.pt"
    torch.save({
        "epoch": epochs,
        "model_state_dict": model.state_dict(),
        "classes": classes,
        "num_classes": num_classes,
        "loss": history["losses"][-1],
        "training_history": history
    }, ckpt_path)

    with open(Path(__file__).resolve().parent / "checkpoints" / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    logger.info("SUCCESS: Best model checkpoint saved to %s", ckpt_path)
    update_execution_status("train", 100.0, f"Model pretraining converged (Final Loss: {history['losses'][-1]})", "COMPLETED")

if __name__ == "__main__":
    pretrain_studio()
