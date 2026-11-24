"""Image loading helpers shared by training, inference and the app."""

from __future__ import annotations

import io
from pathlib import Path
from typing import BinaryIO, Union

from PIL import Image, ImageOps, UnidentifiedImageError

ImageSource = Union[str, Path, bytes, BinaryIO, Image.Image]


class InvalidImageError(ValueError):
    """Raised when an input cannot be decoded as an image."""


def load_rgb_image(source: ImageSource) -> Image.Image:
    """Load an image from a path, raw bytes, a file-like object or a PIL image.

    EXIF orientation is applied (phone photos are often stored rotated) and the
    result is always a fully decoded RGB image.
    """
    try:
        if isinstance(source, Image.Image):
            img = source
        elif isinstance(source, (bytes, bytearray)):
            img = Image.open(io.BytesIO(source))
        elif isinstance(source, (str, Path)):
            path = Path(source)
            if not path.is_file():
                raise InvalidImageError(f"Image file not found: {path}")
            img = Image.open(path)
        else:  # file-like (e.g. Streamlit UploadedFile)
            if hasattr(source, "seek"):
                source.seek(0)
            img = Image.open(source)
        img = ImageOps.exif_transpose(img)
        img = img.convert("RGB")
        img.load()
    except InvalidImageError:
        raise
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
        raise InvalidImageError(f"Could not read image: {exc}") from exc
    if min(img.size) < 32:
        raise InvalidImageError(f"Image is too small ({img.size[0]}x{img.size[1]} px).")
    return img
