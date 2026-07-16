"""
Training loop -- XRV Experiment (chest-specific pretrained backbone +
center-crop). Separate from train.py so we can directly compare this
experiment against the baseline model.

Run:
  python src/radiology_agent/train_xrv.py --root-dir data/pneumonia/chest_xray --epochs 5
"""

import argparse
import os
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from sklearn.metrics import roc_auc_score
import numpy as np

from dataset import get_datasets_xrv, CLASS_NAMES
from model_xrv import RadiologyClassifierXRV


class FocalLoss(nn.Module):
    """
    Focal Loss -- defined locally here (not imported from train.py) to
    avoid a module-name collision with torchxrayvision's internal
    'model' package.
    """
    def __init__(self, weight=None, gamma=2.0):
        super().__init__()
        self.weight = weight
        self.gamma = gamma

    def forward(self, logits, targets):
        ce_loss = nn.functional.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    train_ds, val_ds, test_ds = get_datasets_xrv(args.root_dir)
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = RadiologyClassifierXRV(num_classes=len(CLASS_NAMES), freeze_backbone=True).to(device)

    class_weights = torch.tensor([2.5, 0.67], dtype=torch.float32).to(device)
    criterion = FocalLoss(weight=class_weights, gamma=2.0)
    # Only the new classifier head has trainable params (backbone frozen)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    os.makedirs("checkpoints_xrv", exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        train_loss = running_loss / len(train_ds)
        val_auc, val_acc = evaluate(model, val_loader, device)
        print(f"Epoch {epoch}/{args.epochs} | train_loss={train_loss:.4f} | val_auc={val_auc:.4f} | val_acc={val_acc:.4f}")
        torch.save(model.state_dict(), f"checkpoints_xrv/radiology_xrv_epoch{epoch}.pt")


def evaluate(model, loader, device):
    model.eval()
    all_probs, all_labels, all_preds = [], [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)
            all_preds.append(preds)
            all_labels.append(labels.numpy())

    all_probs = np.concatenate(all_probs)
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    auc = roc_auc_score(all_labels, all_probs)
    acc = (all_preds == all_labels).mean()
    return auc, acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", default="data/pneumonia/chest_xray")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)  # higher LR ok since only head trains
    args = parser.parse_args()
    train(args)
