"""ResNet-50 transfer-learning classifier for cotton-percentage categories."""

from __future__ import annotations

import torch
from torch import nn
from torchvision import models


def build_model(num_classes: int, pretrained: bool = True, dropout: float = 0.3) -> nn.Module:
    """ResNet-50 with its ImageNet head replaced by ``num_classes`` outputs.

    ``pretrained=True`` downloads ImageNet weights (IMAGENET1K_V2) on first use.
    Inference passes ``pretrained=False`` because all weights come from the
    trained checkpoint.
    """
    if num_classes < 2:
        raise ValueError(f"num_classes must be >= 2, got {num_classes}")
    weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    model = models.resnet50(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(in_features, num_classes))
    return model


def parameter_groups(model: nn.Module, lr_backbone: float, lr_head: float) -> list[dict]:
    """Lower learning rate for the pretrained backbone, higher for the new head."""
    head = list(model.fc.parameters())
    head_ids = {id(p) for p in head}
    backbone = [p for p in model.parameters() if id(p) not in head_ids]
    return [
        {"params": backbone, "lr": lr_backbone},
        {"params": head, "lr": lr_head},
    ]


def load_model_from_checkpoint(checkpoint: dict, device: torch.device) -> nn.Module:
    cfg = checkpoint.get("config", {})
    model = build_model(
        num_classes=len(checkpoint["class_to_idx"]),
        pretrained=False,
        dropout=float(cfg.get("dropout", 0.3)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device).eval()
