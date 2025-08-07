import os
import sys
import time
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from sklearn.metrics import jaccard_score

from src.datasets.vqgan_token_dataset import VQGANWeakSegDataset
from src.models.weakly_seg.silly_model import TokenCAMCNN

# Add VQGAN model path
sys.path.append("src/externals/taming-transformers")
from taming.models.vqgan import GumbelVQ

# Configuration
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 4
BATCH_SIZE = 128
EPOCHS = 10
LR = 1e-3
IMAGE_SIZE = 320
SAVE_PATH = "models/weakly_seg/token_cam_cnn.pth"

TRAIN_DIR = "/project/hnguyen2/mvu9/datasets/weakly_sup_segmentation_datasets/LUAD-HistoSeg/training"
VAL_IMG_DIR = "/project/hnguyen2/mvu9/datasets/weakly_sup_segmentation_datasets/LUAD-HistoSeg/val/img"
VAL_MASK_DIR = "/project/hnguyen2/mvu9/datasets/weakly_sup_segmentation_datasets/LUAD-HistoSeg/val/mask"

# Load VQGAN
def load_vqgan(config_path, ckpt_path):
    config = OmegaConf.load(config_path)
    model = GumbelVQ(**config.model.params)
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)["state_dict"]
    model.load_state_dict(sd, strict=False)
    return model.eval().to(DEVICE)

vqgan_path = "src/externals/taming-transformers/logs/vqgan_gumbel_f8"
vqgan = load_vqgan(
    config_path=os.path.join(vqgan_path, "configs/model.yaml"),
    ckpt_path=os.path.join(vqgan_path, "checkpoints/last.ckpt")
)

# Datasets and Dataloaders
train_dataset = VQGANWeakSegDataset(
    image_dir=TRAIN_DIR,
    config_path=os.path.join(vqgan_path, "configs/model.yaml"),
    ckpt_path=os.path.join(vqgan_path, "checkpoints/last.ckpt"),
    image_size=IMAGE_SIZE,
    device=DEVICE,
    return_mask=False
)

val_dataset = VQGANWeakSegDataset(
    image_dir=VAL_IMG_DIR,
    config_path=os.path.join(vqgan_path, "configs/model.yaml"),
    ckpt_path=os.path.join(vqgan_path, "checkpoints/last.ckpt"),
    image_size=IMAGE_SIZE,
    device=DEVICE,
    return_mask=True,
    mask_dir=VAL_MASK_DIR
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

# Model, Loss, Optimizer
model = TokenCAMCNN(num_classes=NUM_CLASSES).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.BCEWithLogitsLoss()

# Training Loop
for epoch in range(EPOCHS):
    start_time = time.time()
    model.train()
    total_loss = 0.0

    print(f"\nEpoch {epoch + 1}/{EPOCHS} ----------")
    log_interval = 10 #max(1, len(train_loader) // 5)  # Log every 5%

    for batch_idx, (indices, labels) in enumerate(train_loader):
        x = indices.unsqueeze(1).float().to(DEVICE)
        y = labels.to(DEVICE)

        optimizer.zero_grad()
        logits, _ = model(x)
        pooled_logits = logits.mean(dim=[2, 3])
        loss = criterion(pooled_logits, y)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if (batch_idx + 1) % log_interval == 0 or (batch_idx + 1) == len(train_loader):
            progress = 100.0 * (batch_idx + 1) / len(train_loader)
            print(f"  Batch {batch_idx + 1}/{len(train_loader)} [{progress:.1f}%] - Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(train_loader)
    print(f"Average training loss: {avg_loss:.4f}")

    # Validation (per-class IoU)
    model.eval()
    ious = []
    valid_samples = 0

    with torch.no_grad():
        for indices, _, mask in val_loader:
            x = indices.unsqueeze(1).float().to(DEVICE)
            mask = mask.squeeze(1).long().to(DEVICE)

            logits, _ = model(x)
            preds = torch.argmax(logits.squeeze(0), dim=0)

            pred_np = preds.cpu().numpy().flatten()
            mask_np = mask.cpu().numpy().flatten()

            if np.sum(mask_np) == 0:
                continue

            iou = jaccard_score(mask_np, pred_np, average="macro", labels=list(range(NUM_CLASSES)))
            ious.append(iou)
            valid_samples += 1

    avg_iou = np.mean(ious) if valid_samples > 0 else 0.0
    elapsed = time.time() - start_time

    print(f"Epoch {epoch + 1} finished in {elapsed:.2f} seconds")
    print(f"Validation mIoU: {avg_iou:.4f} from {valid_samples} samples")

# Save Model
os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
torch.save(model.state_dict(), SAVE_PATH)
print(f"Model saved to {SAVE_PATH}")


if __name__ == "__main__":
    # to be updated