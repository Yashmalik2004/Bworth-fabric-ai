"""ImageNet ResNet-50 embeddings for the R8 look-alike review, from LOCALLY CACHED weights only.

Never downloads: the weights file is looked up in the torch hub cache and loaded with torch.load. If it is missing,
`cached_weights_path()` returns None and callers fall back to the colour/structure similarity.
Embeddings only flag pairs for human review; they never decide that two fabrics are the same.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

WEIGHTS_NAME = "IMAGENET1K_V2"


def cached_weights_path() -> Path | None:
    import torch
    from torchvision import models

    w = getattr(models.ResNet50_Weights, WEIGHTS_NAME)
    f = Path(torch.hub.get_dir()) / "checkpoints" / os.path.basename(w.url)
    return f if f.is_file() else None


def describe() -> dict:
    f = cached_weights_path()
    return {"available": f is not None, "model": f"torchvision ResNet-50 {WEIGHTS_NAME}",
            "weights_file": str(f) if f else None,
            "note": "loaded from local cache only; never downloaded" if f else
                    "cached weights not found; embedding check unavailable, colour/structure check used alone"}


def embed(paths: list[Path], batch_size: int = 16, device: str = "cpu") -> np.ndarray | None:
    """L2-normalised 2048-d penultimate features, or None if the cached weights are unavailable."""
    f = cached_weights_path()
    if f is None:
        return None
    import torch
    from PIL import Image, ImageOps
    from torchvision import models

    net = models.resnet50(weights=None)
    net.load_state_dict(torch.load(f, map_location="cpu", weights_only=True))
    net.fc = torch.nn.Identity()
    net = net.to(device).eval()
    tf = getattr(models.ResNet50_Weights, WEIGHTS_NAME).transforms()   # preprocessing only, no download
    out = []
    with torch.inference_mode():
        for i in range(0, len(paths), batch_size):
            batch = []
            for p in paths[i:i + batch_size]:
                with Image.open(p) as im:
                    batch.append(tf(ImageOps.exif_transpose(im).convert("RGB")))
            feats = net(torch.stack(batch).to(device)).float()
            out.append(torch.nn.functional.normalize(feats, dim=1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 2048), dtype=np.float32)
