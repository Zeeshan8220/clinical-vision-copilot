"""
Training loop for the Radiology Agent.

Run on Colab (free GPU) or Kaggle notebooks:
  python train.py --epochs 5 --batch-size 32

Checkpoints save to checkpoints/radiology_epoch{N}.pt
"""

import argparse
import os
import torch
from torch.utils.data import DataLoader, random_split
from torch import nn, optim
from sklearn.metrics import roc_auc_score
import numpy as np

from dataset import ChestXrayDataset, LABELS
from model import RadiologyClassifier


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    full_ds = ChestXrayDataset(
        csv_path=args.csv_path,
        images_dir=args.images_dir,
    )
    val_size = int(len(full_ds) * 0.15)
    train_size = len(full_ds) - val_size
    train_ds, val_ds = random_split(full_ds, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = RadiologyClassifier(num_labels=len(LABELS)).to(device)
    criterion = nn.BCEWithLogitsLoss()
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

        train_loss = running_loss / train_size
        val_auc = evaluate(model, val_loader, device)
        print(f"Epoch {epoch}/{args.epochs} | train_loss={train_loss:.4f} | val_auc={val_auc:.4f}")

        torch.save(model.state_dict(), f"checkpoints/radiology_epoch{epoch}.pt")


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = torch.sigmoid(model(images)).cpu().numpy()
            all_preds.append(outputs)
            all_labels.append(labels.numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    aucs = []
    for i in range(all_labels.shape[1]):
        if len(np.unique(all_labels[:, i])) > 1:  # skip labels with no positive examples in val split
            aucs.append(roc_auc_score(all_labels[:, i], all_preds[:, i]))
    return np.mean(aucs) if aucs else 0.0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-path", default="../../data/chestxray14/Data_Entry_2017.csv")
    parser.add_argument("--images-dir", default="../../data/chestxray14/images")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()
    train(args)
