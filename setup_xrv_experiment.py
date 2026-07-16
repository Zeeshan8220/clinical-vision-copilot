import os
files = {}

files['src/radiology_agent/dataset.py'] = '''"""
Dataset loader for the Radiology Agent — Phase 1 v1.

Uses the Kaggle "Chest X-ray Pneumonia" dataset (Normal vs Pneumonia,
binary classification):
  kaggle datasets download -d paultimothymooney/chest-xray-pneumonia \
      -p data/pneumonia --unzip

Layout (already train/test/val split by folder, so we use torchvision's
built-in ImageFolder instead of hand-parsing a CSV):
  data/pneumonia/chest_xray/train/NORMAL/*.jpeg
  data/pneumonia/chest_xray/train/PNEUMONIA/*.jpeg
  data/pneumonia/chest_xray/test/NORMAL/*.jpeg
  data/pneumonia/chest_xray/test/PNEUMONIA/*.jpeg
  data/pneumonia/chest_xray/val/NORMAL/*.jpeg
  data/pneumonia/chest_xray/val/PNEUMONIA/*.jpeg

Class index mapping (alphabetical, ImageFolder's default):
  0 = NORMAL, 1 = PNEUMONIA
"""

import os
import torchvision.transforms as T
from torchvision.datasets import ImageFolder

CLASS_NAMES = ["NORMAL", "PNEUMONIA"]


class CropTop:
    """
    Crops off the top portion of the image (shoulders/collarbone/neck
    area) before resizing. This removes a region the model was using as
    a 'shortcut' cue instead of looking at actual lung tissue.
    """
    def __init__(self, fraction=0.18):
        self.fraction = fraction

    def __call__(self, img):
        w, h = img.size
        top = int(h * self.fraction)
        return img.crop((0, top, w, h))


TRAIN_TRANSFORM = T.Compose([
    CropTop(0.18),
    T.Resize((224, 224)),
    T.RandomHorizontalFlip(),
    T.RandomRotation(10),
    T.ColorJitter(brightness=0.2, contrast=0.2),
    T.Grayscale(num_output_channels=3),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

EVAL_TRANSFORM = T.Compose([
    CropTop(0.18),
    T.Resize((224, 224)),
    T.Grayscale(num_output_channels=3),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def get_datasets(root_dir):
    """
    root_dir should point to .../chest_xray (the folder containing
    train/, test/, val/ subfolders).
    """
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


# === XRV Experiment: chest-specific pretrained backbone pipeline ===
# Separate from the main pipeline above so we can compare results
# side-by-side instead of overwriting the working baseline.

import numpy as np
try:
    import torchxrayvision as xrv
except ImportError:
    xrv = None


class CropCenter:
    """
    Cuts out the central vertical strip (heart/spine/mediastinum area)
    and stitches the left and right lung fields back together. This is
    a cheap heuristic (no segmentation model needed) to stop the model
    from using central chest anatomy as a shortcut instead of lung
    tissue.
    """
    def __init__(self, fraction=0.15):
        self.fraction = fraction

    def __call__(self, img):
        w, h = img.size
        strip = int(w * self.fraction)
        cx = w // 2
        left_img = img.crop((0, 0, max(cx - strip // 2, 1), h))
        right_img = img.crop((min(cx + strip // 2, w - 1), 0, w, h))
        new_w = left_img.width + right_img.width
        new_img = Image.new("L", (new_w, h))
        new_img.paste(left_img, (0, 0))
        new_img.paste(right_img, (left_img.width, 0))
        return new_img


class XRVPreprocess:
    """
    torchxrayvision models expect single-channel images normalized with
    their own scheme (not ImageNet mean/std) -- roughly maps pixel
    values into a [-1024, 1024] range that matches what the pretrained
    model saw during its own training.
    """
    def __call__(self, img):
        img = img.resize((224, 224))
        arr = np.array(img).astype(np.float32)
        arr = xrv.datasets.normalize(arr, maxval=255)
        arr = arr[None, :, :]  # add channel dim -> [1, 224, 224]
        return torch.from_numpy(arr).float()


import torch  # noqa: E402 (needed for XRVPreprocess above)
from PIL import Image  # noqa: E402

XRV_TRAIN_TRANSFORM = T.Compose([
    CropTop(0.18),
    CropCenter(0.15),
    T.RandomHorizontalFlip(),
    XRVPreprocess(),
])

XRV_EVAL_TRANSFORM = T.Compose([
    CropTop(0.18),
    CropCenter(0.15),
    XRVPreprocess(),
])


def get_datasets_xrv(root_dir):
    train_ds = ImageFolder(os.path.join(root_dir, "train"), transform=XRV_TRAIN_TRANSFORM)
    val_ds = ImageFolder(os.path.join(root_dir, "val"), transform=XRV_EVAL_TRANSFORM)
    test_ds = ImageFolder(os.path.join(root_dir, "test"), transform=XRV_EVAL_TRANSFORM)
    return train_ds, val_ds, test_ds
'''

files['src/radiology_agent/model_xrv.py'] = '''"""
Radiology Agent model -- XRV Experiment.

Uses torchxrayvision's DenseNet121, pretrained on ~200,000 real chest
X-rays across multiple public datasets (already knows real lung/disease
visual patterns, unlike our EfficientNet which started from generic
ImageNet photos). We freeze the pretrained backbone and only train a
new classifier head on top for our NORMAL vs PNEUMONIA task.
"""

import torch
import torch.nn as nn
import torchxrayvision as xrv


class RadiologyClassifierXRV(nn.Module):
    def __init__(self, num_classes=2, freeze_backbone=True):
        super().__init__()
        base = xrv.models.DenseNet(weights="densenet121-res224-all")
        self.features = base.features  # DenseNet121 conv feature extractor

        if freeze_backbone:
            for p in self.features.parameters():
                p.requires_grad = False

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(1024, num_classes)

    def forward(self, x):
        feats = self.features(x)
        feats = torch.relu(feats)  # DenseNet convention before pooling
        pooled = self.pool(feats).flatten(1)
        return self.classifier(pooled)


if __name__ == "__main__":
    model = RadiologyClassifierXRV()
    dummy = torch.randn(2, 1, 224, 224)  # note: 1 channel, not 3
    out = model(dummy)
    print(f"Output shape: {out.shape}")  # expect [2, 2]
'''

files['src/radiology_agent/train_xrv.py'] = '''"""
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
from train import FocalLoss  # reuse the same FocalLoss we already built


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
'''

files['src/radiology_agent/gradcam_batch_xrv.py'] = '''"""
Grad-CAM batch test -- XRV Experiment version.

Run:
  python src/radiology_agent/gradcam_batch_xrv.py --checkpoint checkpoints_xrv/radiology_xrv_epoch5.pt
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

from dataset import CropTop, CropCenter, XRVPreprocess, CLASS_NAMES
import torchvision.transforms as T
from model_xrv import RadiologyClassifierXRV


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = RadiologyClassifierXRV(num_classes=len(CLASS_NAMES))
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    # Last dense block before the final batch-norm -- a reasonable
    # Grad-CAM target for DenseNet architectures
    target_layer = model.features.denseblock4
    cam = GradCAM(model=model, target_layers=[target_layer])

    preprocess = T.Compose([CropTop(0.18), CropCenter(0.15), XRVPreprocess()])

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
        cropped = CropCenter(0.15)(CropTop(0.18)(original))  # for display
        input_tensor = preprocess(original).unsqueeze(0).to(device)

        display_img = np.array(cropped.resize((224, 224)).convert("RGB")).astype(np.float32) / 255.0

        with torch.no_grad():
            probs = torch.softmax(model(input_tensor), dim=1)[0]
            pneumonia_prob = float(probs[1])
            pred_class = 1 if pneumonia_prob >= args.threshold else 0
            confidence = pneumonia_prob if pred_class == 1 else (1 - pneumonia_prob)

        grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(pred_class)])[0]
        visualization = show_cam_on_image(display_img, grayscale_cam, use_rgb=True)

        correct = "CORRECT" if CLASS_NAMES[pred_class] == true_label else "WRONG"

        axes[0, i].imshow(display_img)
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
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output", default="gradcam_grid_xrv.png")
    args = parser.parse_args()
    main(args)
'''


for path, content in files.items():
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.write(content)
    print(f"Written: {path}")

import subprocess
from google.colab import userdata

subprocess.run(["git", "add", "."])
commit = subprocess.run(["git", "commit", "-m", "Add TorchXRayVision experiment: center-crop + chest-specific pretrained backbone"], capture_output=True, text=True)
print(commit.stdout, commit.stderr)

token = userdata.get('GITHUB_TOKEN')
remote_url = f"https://{token}@github.com/Zeeshan8220/clinical-vision-copilot.git"
push = subprocess.run(["git", "push", remote_url, "main"], capture_output=True, text=True)
print(push.stdout, push.stderr)
