"""
Instead of retraining, find a better decision threshold on the existing
model. Default classification uses threshold=0.5 (if P(PNEUMONIA) > 50%,
predict PNEUMONIA). Raising the threshold makes the model more
"reluctant" to say PNEUMONIA, which should reduce false alarms on NORMAL
patients -- at some cost to catching every PNEUMONIA case.

Run:
  python src/radiology_agent/find_threshold.py --checkpoint checkpoints/radiology_epoch3.pt
"""

import argparse
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
import numpy as np

from dataset import get_datasets, CLASS_NAMES
from radiology_model import RadiologyClassifier


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, _, test_ds = get_datasets(args.root_dir)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=2)

    model = RadiologyClassifier(num_classes=len(CLASS_NAMES))
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    all_probs, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.numpy())

    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)

    for threshold in [0.9, 0.93, 0.95, 0.97, 0.98, 0.99]:
        preds = (all_probs >= threshold).astype(int)
        print(f"
=== Threshold = {threshold} ===")
        print(classification_report(all_labels, preds, target_names=CLASS_NAMES, zero_division=0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", default="data/pneumonia/chest_xray")
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    main(args)
