"""Pretrained backbones with a roster-sized classification head."""
from __future__ import annotations

import torch.nn as nn
import torchvision.models as models

BACKBONES = {
    "mobilenet_v3_small",
    "mobilenet_v3_large",
    "efficientnet_b0",
    "resnet18",
    "resnet34",
    "resnet50",
}


def build_model(backbone: str, num_classes: int, freeze: bool = True, pretrained: bool = True) -> nn.Module:
    if backbone not in BACKBONES:
        raise ValueError(f"Unknown backbone {backbone!r}. Choose from {sorted(BACKBONES)}.")
    weights = "DEFAULT" if pretrained else None

    if backbone == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=weights)
    elif backbone == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(weights=weights)
    elif backbone == "efficientnet_b0":
        model = models.efficientnet_b0(weights=weights)
    else:  # resnet family
        model = getattr(models, backbone)(weights=weights)

    # Replace the classification head, reading in_features from the pretrained
    # model so the size is always correct (no hard-coded channel counts).
    if hasattr(model, "classifier"):
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
    else:
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    if freeze:
        if hasattr(model, "features"):
            for p in model.features.parameters():
                p.requires_grad = False
        else:  # resnet: freeze everything, then unfreeze the new head
            for p in model.parameters():
                p.requires_grad = False
            for p in model.fc.parameters():
                p.requires_grad = True

    return model
