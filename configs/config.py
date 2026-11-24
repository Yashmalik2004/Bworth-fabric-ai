"""Central configuration for the BWorth fabric-composition pipeline.

All paths are resolved relative to the project root so scripts work no matter
which directory they are launched from. Hyperparameters live in
``CompositionConfig`` and can be overridden from the command line by
``training/composition/train.py``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_ROOT = PROJECT_ROOT / "datasets" / "compositions" / "CottonFabricImageBD"
ORIGINAL_DIR = DATASET_ROOT / "Original"
AUGMENTED_DIR = DATASET_ROOT / "Augmented"

PROCESSED_DIR = PROJECT_ROOT / "processed" / "composition"
MODELS_DIR = PROJECT_ROOT / "models" / "composition"
EVALUATION_DIR = PROJECT_ROOT / "evaluation" / "composition"
IMAGES_DIR = PROJECT_ROOT / "images"

DEFAULT_CHECKPOINT = MODELS_DIR / "best_model.pth"

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# ImageNet statistics: required because the backbone is ImageNet-pretrained.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def class_sort_key(name: str) -> int:
    """Sort '30cotton' < '53cotton' < '99cotton' numerically, not lexically."""
    digits = "".join(ch for ch in name if ch.isdigit())
    return int(digits) if digits else 10**9


def cotton_label(class_name: str) -> str:
    """Human-readable label, e.g. '65cotton' -> '65% cotton'."""
    digits = "".join(ch for ch in class_name if ch.isdigit())
    return f"{digits}% cotton" if digits else class_name


@dataclass
class CompositionConfig:
    # Data
    data_dir: str = str(ORIGINAL_DIR)
    split_strategy: str = "group"  # "group" (swatch-aware) or "random" (per image)
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    seed: int = 42
    # Swatch grouping (only used when split_strategy == "group")
    group_distance_threshold: float = 0.35
    group_min_groups_per_class: int = 3

    # Preprocessing (saved inside the checkpoint and reused at inference)
    image_size: int = 320
    resize_size: int = 366
    mean: tuple = IMAGENET_MEAN
    std: tuple = IMAGENET_STD

    # Model
    architecture: str = "resnet50"
    pretrained: bool = True
    dropout: float = 0.3

    # Optimisation (defaults sized for a ~4 GB laptop GPU)
    epochs: int = 30
    batch_size: int = 16
    lr_head: float = 1e-3
    lr_backbone: float = 1e-4
    weight_decay: float = 1e-4
    label_smoothing: float = 0.1
    class_weighted_loss: bool = True
    amp: bool = True
    num_workers: int = 2  # each Windows worker re-imports torch; keep low on 16 GB RAM
    early_stopping_patience: int = 8

    # Outputs
    run_name: str = "resnet50_group_split"
    checkpoint_path: str = str(DEFAULT_CHECKPOINT)
    output_dir: str = str(EVALUATION_DIR)
    splits_path: str = field(default="")

    def __post_init__(self) -> None:
        if not self.splits_path:
            self.splits_path = str(PROCESSED_DIR / f"splits_{self.split_strategy}.csv")

    def preprocessing(self) -> dict:
        """The exact evaluation-time preprocessing, stored with the checkpoint."""
        return {
            "resize_size": self.resize_size,
            "image_size": self.image_size,
            "mean": list(self.mean),
            "std": list(self.std),
        }

    def to_dict(self) -> dict:
        d = asdict(self)
        d["mean"] = list(self.mean)
        d["std"] = list(self.std)
        return d
