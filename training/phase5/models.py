"""Phase 5 architectures (torchvision only, no new dependencies).

build_model never downloads weights unless pretrained=True is passed explicitly (real-data stage, after approval).
ResNet-50 reuses the exact Phase 1-3 head (Dropout + Linear), so ResNet-50 checkpoints stay interchangeable.
"""

from __future__ import annotations

import torch
from torch import nn
from torchvision import models

ARCHITECTURES = ("resnet50", "efficientnet_b0", "mobilenet_v3_large")


def build_model(arch: str, num_classes: int, pretrained: bool = False, dropout: float = 0.3) -> nn.Module:
    if arch == "resnet50":
        from training.composition.model import build_model as build_resnet50

        return build_resnet50(num_classes, pretrained=pretrained, dropout=dropout)
    if arch == "efficientnet_b0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
        m.classifier = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(m.classifier[1].in_features, num_classes))
        return m
    if arch == "mobilenet_v3_large":
        m = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None)
        m.classifier[2] = nn.Dropout(p=dropout)
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, num_classes)
        return m
    raise ValueError(f"unknown architecture {arch!r}; expected one of {ARCHITECTURES}")


def head_parameters(model: nn.Module, arch: str) -> list:
    head = model.fc if arch == "resnet50" else model.classifier
    return list(head.parameters())


def parameter_groups(model: nn.Module, arch: str, lr_backbone: float, lr_head: float) -> list[dict]:
    head = head_parameters(model, arch)
    ids = {id(p) for p in head}
    return [{"params": [p for p in model.parameters() if id(p) not in ids], "lr": lr_backbone},
            {"params": head, "lr": lr_head}]


@torch.inference_mode()
def penultimate_features(model: nn.Module, arch: str, x: torch.Tensor) -> torch.Tensor:
    """Backbone features for the E2 linear probe (frozen backbone)."""
    model.eval()
    if arch == "resnet50":
        body = nn.Sequential(*list(model.children())[:-1])
        return torch.flatten(body(x), 1)
    feats = model.avgpool(model.features(x))
    feats = torch.flatten(feats, 1)
    if arch == "mobilenet_v3_large":
        feats = model.classifier[1](model.classifier[0](feats))   # 960 -> 1280 + Hardswish
    return feats
