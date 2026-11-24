"""Image transforms for training and evaluation.

The evaluation transform is fully determined by the ``preprocessing`` dict that
is saved inside the checkpoint, so inference reproduces it exactly.
"""

from __future__ import annotations

from torchvision import transforms as T


def build_eval_transform(preprocessing: dict) -> T.Compose:
    """Deterministic: resize shorter side -> centre crop -> tensor -> normalise."""
    return T.Compose(
        [
            T.Resize(int(preprocessing["resize_size"]), interpolation=T.InterpolationMode.BILINEAR, antialias=True),
            T.CenterCrop(int(preprocessing["image_size"])),
            T.ToTensor(),
            T.Normalize(mean=preprocessing["mean"], std=preprocessing["std"]),
        ]
    )


COLOR_AUG_MODES = ("standard", "strong")


def color_transforms(color_aug: str = "standard") -> list:
    """Colour augmentation.

    standard: mild jitter (small hue shift), the 13-class baseline setting.
    strong:   any hue, large saturation/brightness changes and 30 % grayscale, so
              colour alone cannot identify a swatch and the model must rely on texture.
    """
    if color_aug == "standard":
        return [T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05)]
    if color_aug == "strong":
        return [T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.8, hue=0.5), T.RandomGrayscale(p=0.3)]
    raise ValueError(f"Unknown color_aug {color_aug!r}; expected one of {COLOR_AUG_MODES}")


def build_train_transform(preprocessing: dict, color_aug: str = "standard") -> T.Compose:
    """On-the-fly augmentation applied ONLY to training images.

    Fabric texture has no canonical orientation, so flips and 90-degree style
    rotations are label-preserving. Colour jitter (see ``color_transforms``)
    discourages memorising the colour of individual swatches.
    """
    size = int(preprocessing["image_size"])
    resize = int(preprocessing["resize_size"])
    return T.Compose(
        [
            T.Resize(resize, interpolation=T.InterpolationMode.BILINEAR, antialias=True),
            T.RandomResizedCrop(size, scale=(0.4, 1.0), ratio=(0.75, 1.333), antialias=True),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomApply([T.RandomRotation(degrees=(90, 90))], p=0.5),
            *color_transforms(color_aug),
            T.RandomApply([T.GaussianBlur(kernel_size=5, sigma=(0.1, 1.5))], p=0.2),
            T.ToTensor(),
            T.Normalize(mean=preprocessing["mean"], std=preprocessing["std"]),
        ]
    )
