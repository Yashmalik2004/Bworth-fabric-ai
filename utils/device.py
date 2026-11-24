"""Device selection helpers (CUDA when available, CPU otherwise)."""

from __future__ import annotations

import torch


def get_device(prefer: str | None = None) -> torch.device:
    """Return the requested device, falling back to CPU if CUDA is unavailable."""
    if prefer:
        if prefer.startswith("cuda") and not torch.cuda.is_available():
            return torch.device("cpu")
        return torch.device(prefer)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def amp_enabled(device: torch.device, requested: bool = True) -> bool:
    """Mixed precision is only used on CUDA, where it saves memory and time."""
    return bool(requested and device.type == "cuda")


def describe_device(device: torch.device) -> str:
    if device.type == "cuda":
        idx = device.index or 0
        props = torch.cuda.get_device_properties(idx)
        return f"cuda:{idx} ({props.name}, {props.total_memory / 1024**3:.1f} GB)"
    return "cpu"
