"""Phase 2: four-band labels, colour augmentation, swatch-group overrides, CV folds."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from configs.config import BAND_DEFINITIONS, FINE_TO_BAND, PROCESSED_DIR, CompositionConfig, display_label
from training.composition.dataset import apply_label_scheme, check_no_leakage, scheme_classes
from training.composition.evaluate import fine_to_band_probs
from training.composition.experiment_splits import OVERRIDES, build_v2, merged_group_ids
from training.composition.transforms import build_train_transform

FINE = ["30cotton", "40cotton", "50cotton", "53cotton", "58cotton", "60cotton", "63cotton",
        "65cotton", "66cotton", "80cotton", "95cotton", "98cotton", "99cotton"]


def test_band_mapping_matches_specification():
    assert set(FINE_TO_BAND) == set(FINE)
    assert BAND_DEFINITIONS["low_cotton"][1] == ("30cotton", "40cotton")
    assert BAND_DEFINITIONS["medium_cotton"][1] == ("50cotton", "53cotton", "58cotton", "60cotton",
                                                    "63cotton", "65cotton", "66cotton")
    assert BAND_DEFINITIONS["high_cotton"][1] == ("80cotton", "95cotton")
    assert BAND_DEFINITIONS["very_high_cotton"][1] == ("98cotton", "99cotton")
    assert display_label("low_cotton") == "Low cotton (30-40%)"
    assert display_label("65cotton") == "65% cotton"


def test_scheme_classes_and_targets():
    assert scheme_classes("fine13", FINE) == FINE
    bands = scheme_classes("band4", FINE)
    assert bands == list(BAND_DEFINITIONS)
    df = pd.DataFrame({"label": ["30cotton", "66cotton", "95cotton", "99cotton"], "label_idx": [0, 8, 10, 12]})
    out = apply_label_scheme(df, "band4", bands)
    assert out["target"].tolist() == [0, 1, 2, 3]
    assert out["label_idx"].tolist() == [0, 8, 10, 12]  # fine labels untouched
    with pytest.raises(ValueError):
        scheme_classes("band7", FINE)


def test_fine_to_band_probabilities_sum_within_bands():
    p = np.full((2, 13), 1 / 13)
    b = fine_to_band_probs(p, FINE)
    assert b.shape == (2, 4)
    np.testing.assert_allclose(b.sum(1), 1)
    np.testing.assert_allclose(b[0], np.array([2, 7, 2, 2]) / 13)


def test_strong_colour_augmentation_runs(texture):
    pre = CompositionConfig().preprocessing()
    tf = build_train_transform(pre, "strong")
    torch.manual_seed(0)
    assert tf(texture(5, size=(400, 300))).shape == (3, pre["image_size"], pre["image_size"])
    with pytest.raises(ValueError):
        build_train_transform(pre, "extreme")


def test_merged_group_ids_union():
    ids = merged_group_ids(["a", "b", "c", "d"], [["a", "b"], ["b", "c"]])
    assert ids["a"] == ids["b"] == ids["c"] == "a+b+c"
    assert ids["d"] == "d"
    with pytest.raises(KeyError):
        merged_group_ids(["a"], [["a", "zzz"]])


def test_build_v2_never_moves_train_images_into_test():
    v1 = pd.DataFrame({
        "path": [f"p{i}" for i in range(6)], "label": ["30cotton"] * 6, "label_idx": [0] * 6,
        "group": ["g0", "g0", "g1", "g2", "g2", "g3"], "split": ["train", "train", "test", "val", "val", "test"],
        "dup_cluster": range(6),
    })
    v2, _ = build_v2(v1, [["g0", "g1"], ["g2", "g3"]])
    assert v2.loc[v2.group_v1 == "g1", "split"].item() == "train"   # merged with a train group
    assert set(v2.loc[v2.group_v1.isin(["g2", "g3"]), "split"]) == {"val"}  # largest part wins
    check_no_leakage(v2)


def test_override_file_is_well_formed():
    data = json.loads(Path(OVERRIDES).read_text())
    for m in data["merges"]:
        assert len(m["groups"]) >= 2 and m["confidence"] in ("definite", "probable")


@pytest.mark.skipif(not (PROCESSED_DIR / "splits_group_v2.csv").is_file(), reason="v2 split not built")
def test_saved_v2_and_folds_are_leakage_free():
    v1 = pd.read_csv(PROCESSED_DIR / "splits_group.csv")
    v2 = pd.read_csv(PROCESSED_DIR / "splits_group_v2.csv")
    check_no_leakage(v2)
    assert set(v1.path) == set(v2.path)
    train_v1 = set(v1.loc[v1.split == "train", "path"])
    assert not train_v1 & set(v2.loc[v2.split == "test", "path"])
    folds = pd.read_csv(PROCESSED_DIR / "cv" / "folds.csv")
    assert folds.groupby("group").fold.nunique().max() == 1
    assert sorted(folds.fold.unique()) == [0, 1, 2, 3, 4]


def test_app_defaults_to_baseline_and_keeps_it_unchanged():
    from configs.config import APP_MODELS, DEFAULT_CHECKPOINT

    first = next(iter(APP_MODELS.values()))
    assert first["checkpoint"] == DEFAULT_CHECKPOINT            # 13-class baseline is the default
    assert all(m["checkpoint"] != DEFAULT_CHECKPOINT for m in list(APP_MODELS.values())[1:])
