"""Phase 5 app scaffolding: dormant by default, safe when a (synthetic) model is selected."""

from pathlib import Path

import pytest
import torch

from app import phase5_model as p5

APP = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")


def _checkpoint(tmp_path, synthetic=True, scheme="phase5_band4"):
    from training.phase5.models import build_model

    m = build_model("mobilenet_v3_large", 4, pretrained=False)
    f = tmp_path / "p5.pth"
    torch.save({"model_state_dict": m.state_dict(), "architecture": "mobilenet_v3_large", "label_scheme": scheme,
                "class_to_idx": {c: i for i, c in enumerate(p5.EXPECTED_CLASSES)},
                "idx_to_class": dict(enumerate(p5.EXPECTED_CLASSES)),
                "preprocessing": {"resize_size": 72, "image_size": 64, "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
                "config": {"dropout": 0.3, "color_aug": "strong"}, "best_epoch": 1, "trained": True,
                "synthetic": synthetic, "snapshot_version": "SYNTH-T"}, f)
    return f


def test_labels_and_disclaimer():
    assert p5.CLASS_LABELS == {"low": "Low cotton: 30–40%", "medium": "Medium cotton: 50–66%",
                               "high": "High cotton: 80–95%", "very_high": "Very high cotton: 98–100%"}
    from app.phase3_model import DISCLAIMER

    assert p5.DISCLAIMER == DISCLAIMER


def test_selection_guards(tmp_path, monkeypatch):
    monkeypatch.setenv("BWORTH_PHASE5_SELECTION", str(tmp_path / "sel.json"))
    ck = _checkpoint(tmp_path)
    with pytest.raises(ValueError, match="synthetic"):
        p5.write_selection(ck)
    bad = _checkpoint(tmp_path / "x" if (tmp_path / "x").mkdir() is None else tmp_path, scheme="band4")
    with pytest.raises(ValueError, match="not a Phase 5"):
        p5.write_selection(bad, allow_synthetic=True)


def test_app_without_phase5_model_is_unchanged(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("BWORTH_PHASE5_SELECTION", str(tmp_path / "absent.json"))
    at = AppTest.from_file(APP, default_timeout=300).run()
    assert not at.exception, at.exception
    assert at.sidebar.radio[0].value == "baseline13"
    assert not any("Phase 5" in o for o in at.sidebar.radio[0].options)


def test_app_offers_selected_synthetic_model_with_warnings(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("BWORTH_PHASE5_SELECTION", str(tmp_path / "sel.json"))
    p5.write_selection(_checkpoint(tmp_path), summary=None, allow_synthetic=True)
    at = AppTest.from_file(APP, default_timeout=300).run()
    assert not at.exception, at.exception
    assert at.sidebar.radio[0].value == "baseline13"                  # default unchanged
    at.sidebar.radio[0].set_value("phase5_band4").run()
    assert not at.exception, at.exception
    errors = [e.value for e in at.error]
    assert p5.SYNTHETIC_WARNING in errors and p5.DISCLAIMER in errors
    assert any("PIPELINE TEST" in m.value for m in at.sidebar.markdown)
    caps = " ".join(c.value for c in at.sidebar.caption)
    assert "Very high cotton: 98–100%" in caps and "Low cotton: 30–40%" in caps


def test_tampered_selection_is_not_offered(tmp_path, monkeypatch):
    monkeypatch.setenv("BWORTH_PHASE5_SELECTION", str(tmp_path / "sel.json"))
    ck = _checkpoint(tmp_path)
    p5.write_selection(ck, allow_synthetic=True)
    assert p5.load_selection() is not None
    ck.write_bytes(ck.read_bytes() + b"x")
    assert p5.load_selection() is None


def test_fabric_level_aggregation(tmp_path, monkeypatch):
    import numpy as np
    from PIL import Image

    from inference.composition.predict import CompositionPredictor

    pr = CompositionPredictor(_checkpoint(tmp_path), device="cpu")
    imgs = [Image.fromarray(np.random.default_rng(i).integers(0, 255, (64, 64, 3), dtype=np.uint8)) for i in range(3)]
    agg = p5.aggregate_fabric_prediction(pr, imgs)
    singles = [pr.predict(im).probabilities for im in imgs]
    for c in p5.EXPECTED_CLASSES:
        assert agg["probabilities"][c] == pytest.approx(sum(s[c] for s in singles) / 3)
    assert agg["n_images"] == 3 and agg["predicted_class"] in p5.EXPECTED_CLASSES
