"""
MODEL PRETRAINING PIPELINE FROM SCRATCH
Project: Power Grid Energy Demand Forecasting
"""

import os
import time
import logging
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("pretrain_studio")

class UnifiedDataset(Dataset):
    def __init__(self, size=50):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return torch.randn(16, 128), torch.randn(16, 128)

def pretrain_studio():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Pretraining from scratch on %s", device)
    model = nn.Sequential(
        nn.Linear(128, 256),
        nn.ReLU(),
        nn.Linear(256, 128)
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    loader = DataLoader(UnifiedDataset(), batch_size=4)

    os.makedirs("./checkpoints", exist_ok=True)
    logger.info("Executing verification pretraining...")
    for epoch in range(1, 3):
        t0 = time.time()
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
        logger.info("Epoch %d finished in %.2fs | Loss: %.4f", epoch, time.time() - t0, loss.item())

    torch.save(model.state_dict(), "./checkpoints/best_model.pt")
    logger.info("Checkpoint saved to ./checkpoints/best_model.pt")

if __name__ == "__main__":
    pretrain_studio()
