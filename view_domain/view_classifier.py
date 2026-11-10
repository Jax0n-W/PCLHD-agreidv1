from __future__ import annotations

from typing import Any, Dict

import torch
from torch import nn
from torchvision import models


class ViewClassifier(nn.Module):
    """ResNet-18 -> global average pooling -> one ground-domain logit."""

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        try:
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            backbone = models.resnet18(weights=weights)
        except (AttributeError, TypeError):
            # torchvision 0.9 used by the original PCLHD environment.
            backbone = models.resnet18(pretrained=pretrained)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Linear(in_features, 1)
        self.backbone = backbone

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.backbone(images).flatten(1).squeeze(1)


def load_view_checkpoint(path, device: torch.device):
    checkpoint = torch.load(str(path), map_location=device)
    model = ViewClassifier(pretrained=False)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    return model, checkpoint

