"""
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
