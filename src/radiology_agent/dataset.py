"""
Dataset loader for the Radiology Agent — Phase 1 v1.
Uses the Kaggle "Chest X-ray Pneumonia" dataset (Normal vs Pneumonia).
"""

import os
import torchvision.transforms as T
from torchvision.datasets import ImageFolder

CLASS_NAMES = ["NORMAL", "PNEUMONIA"]

TRAIN_TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.RandomHorizontalFlip(),
    T.Grayscale(num_output_channels=3),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

EVAL_TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.Grayscale(num_output_channels=3),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def get_datasets(root_dir):
    train_ds = ImageFolder(os.path.join(root_dir, "train"), transform=TRAIN_TRANSFORM)
    val_ds = ImageFolder(os.path.join(root_dir, "val"), transform=EVAL_TRANSFORM)
    test_ds = ImageFolder(os.path.join(root_dir, "test"), transform=EVAL_TRANSFORM)
    return train_ds, val_ds, test_ds


if __name__ == "__main__":
    train_ds, val_ds, test_ds = get_datasets("data/pneumonia/chest_xray")
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    print(f"Classes: {train_ds.classes}")
    img, label = train_ds[0]
    print(f"Image shape: {img.shape}, label: {label}")
