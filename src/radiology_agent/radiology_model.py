"""
Radiology Agent model — Phase 1 v1.
EfficientNet-B0 backbone, fine-tuned for binary classification
(NORMAL vs PNEUMONIA).
"""

import torch
import torch.nn as nn
import torchvision.models as models


class RadiologyClassifier(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        backbone = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        )
        in_features = backbone.classifier[1].in_features
        backbone.classifier[1] = nn.Linear(in_features, num_classes)
        self.backbone = backbone

    def forward(self, x):
        return self.backbone(x)  # raw logits — use with CrossEntropyLoss


def load_model(checkpoint_path=None, num_classes=2, device="cpu"):
    model = RadiologyClassifier(num_classes=num_classes, pretrained=checkpoint_path is None)
    if checkpoint_path:
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


if __name__ == "__main__":
    model = RadiologyClassifier()
    dummy = torch.randn(2, 3, 224, 224)
    out = model(dummy)
    print(f"Output shape: {out.shape}")  # expect [2, 2]
