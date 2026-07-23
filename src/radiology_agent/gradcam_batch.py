"""
Run Grad-CAM on multiple images at once (mix of NORMAL and PNEUMONIA)
to sanity-check that the model is looking at plausible, consistent
regions rather than random/spurious patterns.

Run:
  python src/radiology_agent/gradcam_batch.py --checkpoint checkpoints/radiology_epoch5.pt
"""

import argparse
import os
import random
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from dataset import EVAL_TRANSFORM, CLASS_NAMES
from radiology_model import RadiologyClassifier


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = RadiologyClassifier(num_classes=len(CLASS_NAMES))
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    # Try an earlier block for potentially sharper localization
    # (features[-1] is the final 1x1 projection -- very coarse 7x7 grid;
    # features[-3] is an earlier MBConv block with more spatial detail)
    target_layer = model.backbone.features[-1]
    cam = GradCAM(model=model, target_layers=[target_layer])

    # Pick a random sample of images from each class
    # No fixed seed -- different random images picked each run
    samples = []
    for cls in CLASS_NAMES:
        cls_dir = os.path.join(args.root_dir, "test", cls)
        files = os.listdir(cls_dir)
        chosen = random.sample(files, min(args.per_class, len(files)))
        samples.extend([(os.path.join(cls_dir, f), cls) for f in chosen])

    n = len(samples)
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 8))

    for i, (path, true_label) in enumerate(samples):
        original = Image.open(path).convert("L")
        input_tensor = EVAL_TRANSFORM(original).unsqueeze(0).to(device)
        rgb_img = np.array(original.convert("RGB").resize((224, 224))).astype(np.float32) / 255.0

        with torch.no_grad():
            probs = torch.softmax(model(input_tensor), dim=1)[0]
            pneumonia_prob = float(probs[1])
            pred_class = 1 if pneumonia_prob >= args.threshold else 0
            confidence = pneumonia_prob if pred_class == 1 else (1 - pneumonia_prob)

        grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(pred_class)])[0]
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        correct = "CORRECT" if CLASS_NAMES[pred_class] == true_label else "WRONG"

        axes[0, i].imshow(rgb_img)
        axes[0, i].set_title(f"True: {true_label}", fontsize=10)
        axes[0, i].axis("off")

        axes[1, i].imshow(visualization)
        axes[1, i].set_title(f"Pred: {CLASS_NAMES[pred_class]} ({confidence:.0%}) [{correct}]", fontsize=9)
        axes[1, i].axis("off")

    plt.tight_layout()
    plt.savefig(args.output, dpi=130, bbox_inches="tight")
    print(f"Saved grid to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", default="data/pneumonia/chest_xray")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--per-class", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--output", default="gradcam_grid.png")
    args = parser.parse_args()
    main(args)
