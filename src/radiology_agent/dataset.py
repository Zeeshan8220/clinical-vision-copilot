"""
Dataset loader for the Radiology Agent — Phase 1 v1.

Uses the Kaggle "Chest X-ray Pneumonia" dataset (Normal vs Pneumonia,
binary classification):
  kaggle datasets download -d paultimothymooney/chest-xray-pneumonia       -p data/pneumonia --unzip

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
