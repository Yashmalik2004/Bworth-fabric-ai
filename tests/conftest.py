import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _texture(seed: int, size=(96, 96)) -> Image.Image:
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 255, (*size, 3), dtype=np.uint8))


@pytest.fixture
def tiny_dataset(tmp_path):
    """3 classes x 6 unique images, plus one in-class duplicate and one cross-class conflict."""
    root = tmp_path / "Original"
    seed = 0
    for cls in ("30cotton", "65cotton", "99cotton"):
        (root / cls).mkdir(parents=True)
        for i in range(6):
            _texture(seed).save(root / cls / f"original_{1001 + i}.jpg", quality=95)
            seed += 1
    # duplicate within a class
    _texture(0).save(root / "30cotton" / "original_1099.jpg", quality=95)
    # identical image under two different labels -> label conflict
    _texture(999).save(root / "30cotton" / "original_1098.jpg", quality=95)
    _texture(999).save(root / "65cotton" / "original_1098.jpg", quality=95)
    return root


@pytest.fixture
def texture():
    return _texture
