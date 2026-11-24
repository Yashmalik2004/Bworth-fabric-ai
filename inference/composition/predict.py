"""Cotton-percentage category prediction for a single fabric image.

Python:
    from inference.composition.predict import CompositionPredictor
    predictor = CompositionPredictor()            # loads models/composition/best_model.pth
    result = predictor.predict("fabric.jpg")
    result.predicted_label, result.score, result.probabilities

CLI:
    python -m inference.composition.predict path/to/image.jpg [--checkpoint ...] [--json]

The score is the model's highest softmax probability. It is NOT a calibrated
confidence and NOT a chemically verified composition measurement.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import DEFAULT_CHECKPOINT, cotton_label  # noqa: E402
from training.composition.model import load_model_from_checkpoint  # noqa: E402
from training.composition.transforms import build_eval_transform  # noqa: E402
from utils.device import get_device  # noqa: E402
from utils.image_utils import ImageSource, InvalidImageError, load_rgb_image  # noqa: E402

__all__ = [
    "CompositionPredictor",
    "PredictionResult",
    "CheckpointNotFoundError",
    "InvalidCheckpointError",
    "InvalidImageError",
]

REQUIRED_KEYS = ("model_state_dict", "class_to_idx", "idx_to_class", "preprocessing", "config")


class CheckpointNotFoundError(FileNotFoundError):
    """No trained checkpoint exists at the given path (missing or empty file)."""


class InvalidCheckpointError(RuntimeError):
    """The file exists but is not a usable trained checkpoint."""


@dataclass
class PredictionResult:
    predicted_class: str          # folder name, e.g. "65cotton"
    predicted_label: str          # display name, e.g. "65% cotton"
    score: float                  # highest class probability (uncalibrated)
    probabilities: dict[str, float]  # class name -> probability, in class order

    def top_k(self, k: int = 3) -> list[tuple[str, float]]:
        return sorted(self.probabilities.items(), key=lambda kv: -kv[1])[:k]

    def to_dict(self) -> dict:
        return asdict(self)


def load_checkpoint(path: str | Path) -> dict:
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.is_file() or path.stat().st_size == 0:
        raise CheckpointNotFoundError(
            f"No trained model found at {path}. Train one with: python -m training.composition.train"
        )
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:  # noqa: BLE001
        raise InvalidCheckpointError(f"Could not load checkpoint {path}: {exc}") from exc
    if not isinstance(ckpt, dict) or any(k not in ckpt for k in REQUIRED_KEYS):
        raise InvalidCheckpointError(f"{path} is not a composition checkpoint produced by train.py")
    if not ckpt.get("trained", False):
        raise InvalidCheckpointError(f"{path} is not marked as a trained model")
    return ckpt


class CompositionPredictor:
    def __init__(self, checkpoint_path: str | Path = DEFAULT_CHECKPOINT, device: str | torch.device | None = None):
        self.checkpoint_path = Path(checkpoint_path)
        self.device = device if isinstance(device, torch.device) else get_device(device)
        self.checkpoint = load_checkpoint(self.checkpoint_path)
        idx_to_class = {int(k): v for k, v in self.checkpoint["idx_to_class"].items()}
        self.classes = [idx_to_class[i] for i in range(len(idx_to_class))]
        self.preprocessing = self.checkpoint["preprocessing"]
        self.transform = build_eval_transform(self.preprocessing)
        try:
            self.model = load_model_from_checkpoint(self.checkpoint, self.device)
        except Exception as exc:  # noqa: BLE001
            raise InvalidCheckpointError(f"Checkpoint weights do not match the model: {exc}") from exc

    @property
    def info(self) -> dict:
        c = self.checkpoint
        cfg = c.get("config", {})
        return {
            "architecture": c.get("architecture", "resnet50"),
            "num_classes": len(self.classes),
            "best_epoch": c.get("best_epoch"),
            "val_acc": c.get("val_acc"),
            "split_strategy": cfg.get("split_strategy"),
            "image_size": self.preprocessing.get("image_size"),
            "created_utc": c.get("created_utc"),
            "device": str(self.device),
        }

    @torch.inference_mode()
    def predict(self, image: ImageSource) -> PredictionResult:
        """Predict the cotton-percentage category of one image.

        Raises ``InvalidImageError`` if the input cannot be decoded.
        """
        img = load_rgb_image(image)
        x = self.transform(img).unsqueeze(0).to(self.device)
        probs = torch.softmax(self.model(x).float(), dim=1)[0].cpu().tolist()
        best = max(range(len(probs)), key=probs.__getitem__)
        return PredictionResult(
            predicted_class=self.classes[best],
            predicted_label=cotton_label(self.classes[best]),
            score=float(probs[best]),
            probabilities={c: float(p) for c, p in zip(self.classes, probs)},
        )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Predict the cotton-percentage category of a fabric image.")
    ap.add_argument("image", help="path to an image file")
    ap.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    ap.add_argument("--device", default=None)
    ap.add_argument("--json", action="store_true", help="print the full result as JSON")
    args = ap.parse_args(argv)
    try:
        predictor = CompositionPredictor(args.checkpoint, args.device)
        result = predictor.predict(args.image)
    except (CheckpointNotFoundError, InvalidCheckpointError, InvalidImageError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(f"Predicted category: {result.predicted_label}  (score {result.score:.3f}, uncalibrated)")
        for cls, p in result.top_k(5):
            print(f"  {cotton_label(cls):>12}: {p:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
