"""
Dataset loader for the Radiology Agent.

Phase 1 uses a subset of ChestX-ray14 (NIH) — 14 disease labels on chest
X-rays. It's smaller to download and iterate on than full CT volumes
(LIDC-IDRI), so it's the faster path to a working v1.

Manual download (no API key needed):
  https://nihcc.app.box.com/v/ChestXray-NIHCC
  -> download `images_001.zip` ... `images_012.zip` (start with just 001
     for a fast first pass, ~4GB) and `Data_Entry_2017.csv` (the labels)

  Put them here:
    data/chestxray14/images/*.png
    data/chestxray14/Data_Entry_2017.csv

Kaggle mirror (alternative, needs kaggle.json API key):
  kaggle datasets download -d nih-chest-xrays/data -p data/chestxray14 --unzip
"""

import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

LABELS = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
    "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema",
    "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia",
]

DEFAULT_TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.Grayscale(num_output_channels=3),  # chest x-rays are grayscale
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class ChestXrayDataset(Dataset):
    """
    Multi-label dataset: each image can have 0+ of the 14 findings.
    Reads NIH's Data_Entry_2017.csv where the 'Finding Labels' column
    is a pipe-separated string, e.g. 'Cardiomegaly|Effusion'.
    """

    def __init__(self, csv_path, images_dir, transform=None):
        self.df = pd.read_csv(csv_path)
        self.images_dir = images_dir
        self.transform = transform or DEFAULT_TRANSFORM

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.images_dir, row["Image Index"])
        image = Image.open(img_path).convert("L")
        image = self.transform(image)

        findings = row["Finding Labels"].split("|")
        label = [1.0 if lbl in findings else 0.0 for lbl in LABELS]

        import torch
        return image, torch.tensor(label, dtype=torch.float32)


if __name__ == "__main__":
    # quick sanity check once you've downloaded data
    ds = ChestXrayDataset(
        csv_path="data/chestxray14/Data_Entry_2017.csv",
        images_dir="data/chestxray14/images",
    )
    print(f"Dataset size: {len(ds)}")
    img, label = ds[0]
    print(f"Image shape: {img.shape}, label: {label}")
