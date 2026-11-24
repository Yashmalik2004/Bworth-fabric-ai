import io

import pytest
import torch

from configs.config import DEFAULT_CHECKPOINT, ORIGINAL_DIR, CompositionConfig
from inference.composition.predict import (
    CheckpointNotFoundError,
    CompositionPredictor,
    InvalidCheckpointError,
    InvalidImageError,
)
from training.composition.model import build_model

CLASSES = ["30cotton", "65cotton", "99cotton"]


def _write_checkpoint(path, trained=True):
    cfg = CompositionConfig(image_size=64, resize_size=72)
    model = build_model(len(CLASSES), pretrained=False)
    torch.save({
        "model_state_dict": model.state_dict(),
        "class_to_idx": {c: i for i, c in enumerate(CLASSES)},
        "idx_to_class": {i: c for i, c in enumerate(CLASSES)},
        "preprocessing": cfg.preprocessing(),
        "config": cfg.to_dict(),
        "best_epoch": 1,
        "trained": trained,
    }, path)
    return path


def test_missing_checkpoint(tmp_path):
    with pytest.raises(CheckpointNotFoundError):
        CompositionPredictor(tmp_path / "missing.pth", device="cpu")


def test_empty_checkpoint_is_treated_as_missing(tmp_path):
    p = tmp_path / "empty.pth"
    p.write_bytes(b"")
    with pytest.raises(CheckpointNotFoundError):
        CompositionPredictor(p, device="cpu")


def test_garbage_checkpoint(tmp_path):
    p = tmp_path / "bad.pth"
    p.write_bytes(b"not a checkpoint")
    with pytest.raises(InvalidCheckpointError):
        CompositionPredictor(p, device="cpu")


def test_untrained_checkpoint_rejected(tmp_path):
    p = _write_checkpoint(tmp_path / "untrained.pth", trained=False)
    with pytest.raises(InvalidCheckpointError):
        CompositionPredictor(p, device="cpu")


def test_prediction_mechanics(tmp_path, texture):
    """Mechanics only (weights are random): output structure and probability sum."""
    predictor = CompositionPredictor(_write_checkpoint(tmp_path / "m.pth"), device="cpu")
    img = texture(3, size=(120, 90))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    for source in (img, buf.getvalue()):
        r = predictor.predict(source)
        assert list(r.probabilities) == CLASSES
        assert abs(sum(r.probabilities.values()) - 1) < 1e-4
        assert r.score == max(r.probabilities.values())
        assert r.predicted_class == max(r.probabilities, key=r.probabilities.get)
        assert r.predicted_label.endswith("% cotton")


def test_invalid_image(tmp_path):
    predictor = CompositionPredictor(_write_checkpoint(tmp_path / "m.pth"), device="cpu")
    with pytest.raises(InvalidImageError):
        predictor.predict(b"definitely not an image")
    with pytest.raises(InvalidImageError):
        predictor.predict(tmp_path / "does_not_exist.jpg")


_trained = DEFAULT_CHECKPOINT.is_file() and DEFAULT_CHECKPOINT.stat().st_size > 0


@pytest.mark.skipif(not (_trained and ORIGINAL_DIR.is_dir()), reason="trained checkpoint or dataset missing")
def test_trained_checkpoint_single_image():
    predictor = CompositionPredictor(DEFAULT_CHECKPOINT, device="cpu")
    assert len(predictor.classes) == 13
    sample = next((ORIGINAL_DIR / "30cotton").glob("*.jpg"))
    r = predictor.predict(sample)
    assert r.predicted_class in predictor.classes
    assert 0 < r.score <= 1
    assert abs(sum(r.probabilities.values()) - 1) < 1e-4
