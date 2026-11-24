import pandas as pd
import pytest

from configs.config import ORIGINAL_DIR, CompositionConfig, class_sort_key
from training.composition.dataset import (
    FabricDataset,
    check_no_leakage,
    clean_index,
    discover_classes,
    index_images,
    split_groups,
)


def test_class_sort_is_numeric():
    assert sorted(["99cotton", "30cotton", "5cotton"], key=class_sort_key) == ["5cotton", "30cotton", "99cotton"]


def test_discover_classes(tiny_dataset):
    assert discover_classes(tiny_dataset) == ["30cotton", "65cotton", "99cotton"]


def test_discover_classes_missing_dir(tmp_path):
    with pytest.raises(FileNotFoundError):
        discover_classes(tmp_path / "nope")


def test_clean_index_removes_duplicates_and_conflicts(tiny_dataset):
    raw = index_images(tiny_dataset)
    assert len(raw) == 18 + 3
    res = clean_index(raw, hamming_threshold=0)
    assert len(res.clean) == 18                 # one copy of every unique, unambiguous image
    assert len(res.removed_duplicates) == 1     # the in-class duplicate
    assert len(res.removed_conflicts) == 2      # both copies of the conflicting image
    assert set(res.removed_conflicts["label"]) == {"30cotton", "65cotton"}


def _grouped_frame(n_classes=3, groups_per_class=6, imgs_per_group=4):
    rows = []
    for c in range(n_classes):
        for g in range(groups_per_class):
            for i in range(imgs_per_group):
                rows.append({"path": f"c{c}/g{g}/{i}.jpg", "label": f"c{c}", "label_idx": c,
                             "group": f"c{c}_g{g}", "dup_cluster": len(rows)})
    return pd.DataFrame(rows)


def test_split_groups_no_leakage_and_stratified():
    df = split_groups(_grouped_frame(), val_fraction=0.15, test_fraction=0.15, seed=0)
    check_no_leakage(df)
    assert set(df.split) == {"train", "val", "test"}
    for _, cdf in df.groupby("label"):
        assert set(cdf.split) == {"train", "val", "test"}  # every class in every split
        assert (cdf.split == "train").mean() >= 0.5


def test_split_groups_is_deterministic():
    a = split_groups(_grouped_frame(), 0.15, 0.15, seed=7)
    b = split_groups(_grouped_frame(), 0.15, 0.15, seed=7)
    assert a.split.tolist() == b.split.tolist()


def test_check_no_leakage_detects_leak():
    df = _grouped_frame()
    df["split"] = "train"
    df.loc[0, "split"] = "test"  # group c0_g0 now spans train and test
    with pytest.raises(AssertionError):
        check_no_leakage(df)


def test_fabric_dataset_returns_image_and_label(tiny_dataset):
    df = clean_index(index_images(tiny_dataset), hamming_threshold=0).clean
    ds = FabricDataset(df)
    img, label = ds[0]
    assert img.mode == "RGB" and isinstance(label, int)


@pytest.mark.skipif(not ORIGINAL_DIR.is_dir(), reason="CottonFabricImageBD not present")
def test_real_dataset_structure():
    classes = discover_classes(ORIGINAL_DIR)
    assert len(classes) == 13
    assert classes[0] == "30cotton" and classes[-1] == "99cotton"
    counts = index_images(ORIGINAL_DIR).groupby("label").size()
    assert (counts == 100).all()


@pytest.mark.skipif(not ORIGINAL_DIR.is_dir(), reason="CottonFabricImageBD not present")
def test_saved_splits_have_no_leakage():
    for strategy in ("group", "random"):
        path = CompositionConfig(split_strategy=strategy).splits_path
        try:
            df = pd.read_csv(path)
        except FileNotFoundError:
            pytest.skip(f"{path} not built yet")
        check_no_leakage(df)
        assert df.groupby("dup_cluster").size().max() == 1  # duplicates were collapsed
