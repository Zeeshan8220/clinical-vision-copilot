"""
Radiology Agent model: EfficientNet-B0 backbone, fine-tuned for multi-label
chest X-ray classification (14 findings, sigmoid output — an image can
have more than one finding at once, so this is NOT softmax/single-class).
"""

import torch
import torch.nn as nn
import torchvision.models as models


class RadiologyClassifier(nn.Module):
    def __init__(self, num_labels=14, pretrained=True):
        super().__init__()
        backbone = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        )
        in_features = backbone.classifier[1].in_features
        backbone.classifier[1] = nn.Linear(in_features, num_labels)
        self.backbone = backbone

    def forward(self, x):
        return self.backbone(x)  # raw logits — apply sigmoid at inference time


def load_model(checkpoint_path=None, num_labels=14, device="cpu"):
    model = RadiologyClassifier(num_labels=num_labels, pretrained=checkpoint_path is None)
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
    print(f"Output shape: {out.shape}")  # expect [2, 14]
