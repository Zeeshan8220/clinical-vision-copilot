"""
Grad-CAM explainability for the Radiology Agent.
"""

import argparse
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from dataset import EVAL_TRANSFORM, CLASS_NAMES
from model import RadiologyClassifier


def generate_gradcam(checkpoint_path, image_path, output_path, device="cpu"):
    model = RadiologyClassifier(num_classes=len(CLASS_NAMES))
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    target_layer = model.backbone.features[-1]

    original = Image.open(image_path).convert("L")
    input_tensor = EVAL_TRANSFORM(original).unsqueeze(0).to(device)

    rgb_img = original.convert("RGB").resize((224, 224))
    rgb_img = np.array(rgb_img).astype(np.float32) / 255.0

    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)[0]
        pred_class = int(torch.argmax(probs))
        confidence = float(probs[pred_class])

    cam = GradCAM(model=model, target_layers=[target_layer])
    targets = [ClassifierOutputTarget(pred_class)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]

    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(rgb_img)
    axes[0].set_title("Original X-ray")
    axes[0].axis("off")

    axes[1].imshow(visualization)
    axes[1].set_title(f"Grad-CAM: {CLASS_NAMES[pred_class]} ({confidence:.1%})")
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Prediction: {CLASS_NAMES[pred_class]} ({confidence:.1%} confidence)")
    print(f"Saved visualization to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", default="gradcam_result.png")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    generate_gradcam(args.checkpoint, args.image, args.output, device=device)
