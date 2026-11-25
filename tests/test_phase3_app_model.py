"""Phase 3 three-class model in the app: selection, preprocessing, inference, labels, UI."""

import json

import pandas as pd
import pytest

from app import phase3_model as p3

HAVE_P3 = p3.SELECTION_FILE.is_file() and (p3.PROJECT_ROOT / json.loads(
    p3.SELECTION_FILE.read_text(encoding="utf-8"))["checkpoint"]).is_file() if p3.SELECTION_FILE.is_file() else False
needs_p3 = pytest.mark.skipif(not HAVE_P3, reason="Phase 3 checkpoint/selection not available")


def test_display_labels_and_texts_are_exact():
    assert p3.EXPECTED_CLASSES == ["medium_cotton", "high_cotton", "very_high_cotton"]
    assert p3.CLASS_LABELS == {"medium_cotton": "Medium cotton: 50–66%", "high_cotton": "High cotton: 80–95%",
                               "very_high_cotton": "Very high cotton: 98–99%"}
    assert "low_cotton" not in p3.CLASS_LABELS
    assert p3.DISCLAIMER == (
        "Experimental visual estimate. This model has limited validation on independent physical fabrics. "
        "It does not chemically verify fibre composition or determine exact cotton percentage. "
        "Do not use this result for automatic garment valuation or token allocation.")
    assert "not evaluable" in p3.LOW_NOTE


@needs_p3
def test_selection_record_matches_rule_and_checksum():
    sel = p3.load_selection(verify_hash=True)
    assert sel is not None, "checkpoint missing or SHA-256 mismatch"
    cands = sel["candidates"]
    assert len(cands) == 10 and all(c["valid"] and c["variant"] == "baseline" for c in cands)
    best = sorted(cands, key=lambda c: (-c["val_acc"], c["val_loss"]))[0]
    assert sel["checkpoint"] == best["checkpoint"]           # highest validation accuracy, test not used
    assert sel["held_out_samples"].endswith(f"baseline_fold{best['fold']}.csv")


@needs_p3
def test_checkpoint_preprocessing_is_training_preprocessing():
    from inference.composition.predict import CompositionPredictor
    from training.composition.transforms import build_eval_transform

    sel = p3.load_selection(verify_hash=False)
    pr = CompositionPredictor(p3.PROJECT_ROOT / sel["checkpoint"], device="cpu")
    assert pr.preprocessing == p3.EXPECTED_PREPROCESSING
    assert pr.checkpoint["config"]["color_aug"] == "strong"
    assert pr.classes == p3.EXPECTED_CLASSES
    assert repr(pr.transform) == repr(build_eval_transform(p3.EXPECTED_PREPROCESSING))


@needs_p3
def test_three_class_inference_reproduces_recorded_held_out_predictions():
    from inference.composition.predict import CompositionPredictor

    sel = p3.load_selection(verify_hash=False)
    pr = CompositionPredictor(p3.PROJECT_ROOT / sel["checkpoint"], device="cpu")
    c = sel["chosen"]
    rec = pd.read_csv(p3.STAGE_B / "evaluation" / "baseline" / f"fold{c['fold']}_seed{c['seed']}" / "predictions_test.csv")
    for r in rec.head(4).itertuples():
        out = pr.predict(p3.PROJECT_ROOT / r.path)
        assert set(out.probabilities) == set(p3.EXPECTED_CLASSES)          # no Low class
        assert abs(sum(out.probabilities.values()) - 1) < 1e-4
        assert out.predicted_class == r.predicted
        for cls in p3.EXPECTED_CLASSES:                                      # recorded run used GPU fp16
            assert abs(out.probabilities[cls] - getattr(r, f"p_{cls}")) < 0.01


@needs_p3
def test_app_spec_fields():
    spec = p3.app_spec(p3.load_selection(verify_hash=False))
    assert spec["kind"] == "phase3_band3" and spec["badge"] == "EXPERIMENTAL"
    assert spec["labels"] == p3.CLASS_LABELS
    assert spec["sample_splits"].is_file()
    test_rows = pd.read_csv(spec["sample_splits"]).query("split == 'test'")
    assert set(test_rows.label) <= set(p3.EXPECTED_CLASSES)


APP = str(p3.PROJECT_ROOT / "app" / "streamlit_app.py")


def _run_model(model_key, sample_index=0):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(APP, default_timeout=300).run()
    assert not at.exception, at.exception
    at.sidebar.radio[0].set_value(model_key).run()
    at.selectbox[0].set_value(sample_index).run()
    at.button[0].click().run()
    assert not at.exception, at.exception
    return at


@needs_p3
def test_app_phase3_model_end_to_end():
    at = _run_model("phase3_band3", 3)
    assert at.sidebar.radio[0].value == "phase3_band3"
    texts = " ".join(m.value for m in at.markdown)
    assert any(lbl in texts for lbl in p3.CLASS_LABELS.values())          # predicted band shown with exact label
    table = at.dataframe[0].value
    assert list(table["Category"]) == list(p3.CLASS_LABELS.values())
    assert abs(table["Probability (%)"].sum() - 100) < 0.1
    alerts = [x.value for x in at.error] + [x.value for x in at.warning]
    assert p3.DISCLAIMER in alerts
    captions = " ".join(c.value for c in at.caption) + " ".join(c.value for c in at.sidebar.caption)
    assert "not evaluable" in captions and "Low cotton" in captions
    assert "uncalibrated" in texts or "uncalibrated" in captions
    assert "EXPERIMENTAL" in " ".join(m.value for m in at.sidebar.markdown)


def test_app_default_and_existing_models_still_work():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(APP, default_timeout=300).run()
    assert not at.exception, at.exception
    assert at.sidebar.radio[0].value == "baseline13"                      # default unchanged
    tested = []
    for key, path in (("baseline13", "models/composition/best_model.pth"), ("band4", "models/composition/band4_strong.pth")):
        if not (p3.PROJECT_ROOT / path).is_file():
            continue
        tested.append(key)
        at2 = _run_model(key, 5)
        assert at2.sidebar.radio[0].value == key
        table = at2.dataframe[0].value
        assert len(table) == (13 if key == "baseline13" else 4)
        alerts = [x.value for x in at2.error] + [x.value for x in at2.warning]
        assert p3.DISCLAIMER not in alerts                                 # Phase 3 text only for Phase 3 model
    assert tested, "no existing model checkpoint found to test"
