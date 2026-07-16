"""
Training loop for the Radiology Agent — Phase 1 v1 (binary classification).

Run in Colab:
  %cd src/radiology_agent
  !python train.py --epochs 3 --batch-size 32

Checkpoints save to checkpoints/radiology_epoch{N}.pt
"""

import argparse
import os
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from sklearn.metrics import roc_auc_score
import numpy as np

from dataset import get_datasets, CLASS_NAMES
from model import RadiologyClassifier


class FocalLoss(nn.Module):
    """
    Focal Loss down-weights 'easy' examples (where the model is already
    confident and correct) and focuses training on 'hard' examples
    (where the model is confused) -- on top of the class weights, this
    directly targets the confusing cases we saw in Grad-CAM (e.g.
    NORMAL images the model mistakes for PNEUMONIA).

    gamma controls how much focus shifts to hard examples (2.0 is a
    common default from the original Focal Loss paper).
    """
    def __init__(self, weight=None, gamma=2.0):
        super().__init__()
        self.weight = weight
        self.gamma = gamma

    def forward(self, logits, targets):
        ce_loss = nn.functional.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce_loss)  # pt = model's predicted probability for the TRUE class
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    train_ds, val_ds, test_ds = get_datasets(args.root_dir)
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = RadiologyClassifier(num_classes=len(CLASS_NAMES)).to(device)

    # Class weights to counter imbalance (NORMAL=1341, PNEUMONIA=3875 in train set).
    # Weight_i = total / (num_classes * count_i) -- standard inverse-frequency balancing.
    class_weights = torch.tensor([2.5, 0.67], dtype=torch.float32).to(device)
    criterion = FocalLoss(weight=class_weights, gamma=2.0)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    os.makedirs("checkpoints", exist_ok=True)

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

        torch.save(model.state_dict(), f"checkpoints/radiology_epoch{epoch}.pt")


def evaluate(model, loader, device):
    model.eval()
    all_probs, all_labels, all_preds = [], [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()  # P(PNEUMONIA)
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
    parser.add_argument("--root-dir", default="../../data/pneumonia/chest_xray")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()
    train(args)
