"""
Temperature Scaling for confidence calibration.

Problem: the model is often "overconfident" -- e.g. saying 99% confident
on a WRONG prediction. Temperature scaling fixes this by dividing the
model's raw logits by a learned constant T before softmax. This does NOT
change which class is predicted (T doesn't change argmax), it only makes
the confidence percentages more honest/realistic.

We split the test set into:
  - a "calibration" subset (30%) -- used only to find the best T
  - a "held-out eval" subset (70%) -- used to report final, honest metrics

This avoids reusing training-touched data for calibration.

Run:
  python src/radiology_agent/calibrate.py --checkpoint checkpoints/radiology_epoch5.pt
"""

import argparse
import json
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F

from dataset import get_datasets, CLASS_NAMES
from model import RadiologyClassifier


def get_logits_and_labels(model, loader, device):
    all_logits, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            all_logits.append(logits.cpu())
            all_labels.append(labels)
    return torch.cat(all_logits), torch.cat(all_labels)


def expected_calibration_error(probs, labels, n_bins=10):
    """
    ECE: splits predictions into confidence bins and measures the gap
    between average confidence and actual accuracy in each bin.
    Lower ECE = better calibrated (more "honest" confidence scores).
    """
    confidences, predictions = probs.max(dim=1)
    accuracies = (predictions == labels).float()

    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() > 0:
            bin_acc = accuracies[mask].mean()
            bin_conf = confidences[mask].mean()
            ece += (mask.float().mean()) * torch.abs(bin_acc - bin_conf)
    return ece.item()


def find_best_temperature(logits, labels):
    """Grid search over T, minimizing negative log-likelihood on the calibration set."""
    best_T, best_nll = 1.0, float("inf")
    for T in np.arange(0.5, 5.01, 0.1):
        scaled = logits / T
        nll = F.cross_entropy(scaled, labels).item()
        if nll < best_nll:
            best_nll, best_T = nll, T
    return best_T


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, _, test_ds = get_datasets(args.root_dir)

    n = len(test_ds)
    indices = list(range(n))
    rng = np.random.RandomState(123)
    rng.shuffle(indices)
    split = int(0.3 * n)
    calib_idx, eval_idx = indices[:split], indices[split:]

    calib_loader = DataLoader(Subset(test_ds, calib_idx), batch_size=32, shuffle=False)
    eval_loader = DataLoader(Subset(test_ds, eval_idx), batch_size=32, shuffle=False)

    model = RadiologyClassifier(num_classes=len(CLASS_NAMES))
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    print(f"Calibration set: {len(calib_idx)} images | Held-out eval set: {len(eval_idx)} images
")

    calib_logits, calib_labels = get_logits_and_labels(model, calib_loader, device)
    best_T = find_best_temperature(calib_logits, calib_labels)
    print(f"Best temperature found: T = {best_T:.2f}
")

    eval_logits, eval_labels = get_logits_and_labels(model, eval_loader, device)

    probs_before = F.softmax(eval_logits, dim=1)
    probs_after = F.softmax(eval_logits / best_T, dim=1)

    ece_before = expected_calibration_error(probs_before, eval_labels)
    ece_after = expected_calibration_error(probs_after, eval_labels)

    print(f"=== On held-out eval set ({len(eval_idx)} images) ===")
    print(f"ECE before calibration: {ece_before:.4f}")
    print(f"ECE after calibration:  {ece_after:.4f}")
    print(f"(Lower is better -- gap between confidence and actual accuracy)")

    with open(args.output, "w") as f:
        json.dump({"temperature": best_T}, f)
    print(f"
Saved temperature to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", default="data/pneumonia/chest_xray")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="temperature.json")
    args = parser.parse_args()
    main(args)
