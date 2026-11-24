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


def build_train_transform(preprocessing: dict) -> T.Compose:
    """On-the-fly augmentation applied ONLY to training images.

    Fabric texture has no canonical orientation, so flips and 90-degree style
    rotations are label-preserving. Colour jitter (including a little hue) is
    used to discourage the model from memorising the colour of individual
    swatches instead of their texture.
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
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
            T.RandomApply([T.GaussianBlur(kernel_size=5, sigma=(0.1, 1.5))], p=0.2),
            T.ToTensor(),
            T.Normalize(mean=preprocessing["mean"], std=preprocessing["std"]),
        ]
    )
