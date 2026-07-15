"""
Evaluate a saved checkpoint on the test set.
"""

import argparse
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, classification_report
import numpy as np

from dataset import get_datasets, CLASS_NAMES
from model import RadiologyClassifier


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, _, test_ds = get_datasets(args.root_dir)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=2)

    model = RadiologyClassifier(num_classes=len(CLASS_NAMES))
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    all_probs, all_preds, all_labels = [], [], []
    with torch.no_grad():
        for images, labels in test_loader:
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

    print(f"Test AUC: {roc_auc_score(all_labels, all_probs):.4f}")
    print(f"Test Accuracy: {(all_preds == all_labels).mean():.4f}")
    print()
    print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", default="data/pneumonia/chest_xray")
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    main(args)
