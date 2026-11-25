"""Phase 5 pipeline on SYNTHETIC data (E0). These are software tests, not model-performance results."""

import json
import os
import stat
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from data_collection.phase4 import registry as R
from training.phase5 import evaluate as E
from training.phase5 import models as M
from training.phase5 import readiness, snapshot, splits
from training.phase5.protocol import CLASSES, SYNTHETIC_BANNER, load_protocol
from training.phase5.provenance import build_groups
from training.phase5.synthetic import make_synthetic_dataset
from training.phase5.train import TrainingRefused, train_split


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    base = tmp_path_factory.mktemp("p5")
    make_synthetic_dataset(base / "p4", fabrics_per_band=12, photos_per_fabric=4, seed=0, image_size=64)
    snap = snapshot.create_snapshot(base / "p4", base / "snaps", version="SYNTH-T")
    return {"base": base, "snap": snap}


def writable(path):
    os.chmod(path, stat.S_IWRITE | stat.S_IREAD)


# ---- protocol ------------------------------------------------------------------ #
def test_protocol_bands_match_phase4_registry():
    p = load_protocol()
    assert {k: tuple(v) for k, v in p["task"]["bands"].items()} == R.BANDS
    assert p["readiness_gates"]["R1_min_verified_fabrics_per_band"] == 30
    assert p["success_criteria"]["S1_fabric_balanced_accuracy_mean_min"] == 0.55


def test_synthetic_generator_refuses_real_dataset_path():
    with pytest.raises(ValueError, match="real Phase 4 dataset"):
        make_synthetic_dataset(R.PROJECT_ROOT / "datasets" / "phase4_verified")


# ---- snapshot ------------------------------------------------------------------ #
def test_snapshot_is_frozen_hashed_and_marked_synthetic(synth):
    snap = synth["snap"]
    meta = snapshot.load_meta(snap)
    assert meta["synthetic"] is True and meta["verification_policy_version"] == 2
    assert meta["counts"]["manifest_fabrics"] == 48 and meta["counts"]["validator_errors"] == 0
    v = snapshot.verify_snapshot(snap)
    assert v["ok"] and v["checked"] == len(snapshot.FROZEN) + meta["counts"]["manifest_images"]
    assert not os.access(snap / "manifest.csv", os.W_OK)                      # frozen files read-only
    with pytest.raises(snapshot.SnapshotRefused, match="immutable"):
        snapshot.create_snapshot(synth["base"] / "p4", synth["base"] / "snaps", version="SYNTH-T")


def test_snapshot_detects_tampered_image_and_protocol_change(tmp_path, monkeypatch):
    make_synthetic_dataset(tmp_path / "p4", fabrics_per_band=2, photos_per_fabric=1, seed=3, image_size=32)
    snap = snapshot.create_snapshot(tmp_path / "p4", tmp_path / "s")
    man = snapshot.load_tables(snap)["manifest"]
    (tmp_path / "p4" / man.image_path.iloc[0]).write_bytes(b"tampered")
    assert any("hash mismatch" in x for x in snapshot.verify_snapshot(snap)["problems"])
    monkeypatch.setattr(snapshot, "protocol_sha256", lambda: "0" * 64)
    assert any("protocol" in x for x in snapshot.verify_snapshot(snap)["problems"])


def test_snapshot_refused_when_validator_has_errors(tmp_path, monkeypatch):
    monkeypatch.setenv("BWORTH_PHASE4_ROOT", str(tmp_path / "bad"))
    R.init_dataset(policy_version=2)
    R.add_fabric({"cotton_percentage": 150})
    with pytest.raises(snapshot.SnapshotRefused, match="ERROR"):
        snapshot.create_snapshot(tmp_path / "bad", tmp_path / "s")


# ---- provenance ------------------------------------------------------------------ #
def test_provenance_groups_are_transitive_and_normalised():
    fab = pd.DataFrame({"fabric_id": ["A", "B", "C", "D", "E"], "provenance_group": ["PG1", "pg1 ", "", "", ""],
                        "source_product_id": ["", "SKU 1", "sku  1", "", ""]})
    ver = pd.DataFrame({"fabric_id": ["D", "E"], "product_identifier": ["X9", ""]})
    g = build_groups(fab, ver, ["A", "B", "C", "D", "E"]).set_index("fabric_id")
    assert g.provenance_id["A"] == g.provenance_id["B"] == g.provenance_id["C"]   # PG1 links A-B, SKU 1 links B-C
    assert len({g.provenance_id["D"], g.provenance_id["E"], g.provenance_id["A"]}) == 3


# ---- readiness, lockbox, splits --------------------------------------------------- #
def test_readiness_lockbox_and_splits_end_to_end(synth):
    snap = synth["snap"]
    pre = readiness.evaluate(snap)
    g = pre["gates"]
    assert pre["overall"] == "NOT_READY" and pre["synthetic"]
    assert g["R1"]["status"] == "FAIL" and g["R1"]["verified_fabrics_per_band"] == {c: 12 for c in CLASSES}
    assert g["R2"]["status"] == "PASS" and g["R3"]["status"] == "PASS"
    assert g["R7"]["definition"].startswith("adjacent_band_distance")       # protocol v1.1
    assert g["R7"]["boundary_share_per_band"]["low"] == 0.0 and g["R7"]["boundary_share_per_band"]["medium"] == 0.0
    assert g["R9"]["status"] == "FAIL" and not g["R9"]["lockbox_sealed"]
    assert set(g["R6"]["cramers_v"]) == {"capture_device", "lighting", "capture_session"}
    assert (snap / "readiness" / "READINESS_REPORT.md").read_text().count("SYNTHETIC") >= 1

    lb = splits.seal_lockbox(snap)
    assert splits.verify_lockbox(snap)["ok"]
    with pytest.raises(splits.SplitError, match="already sealed"):
        splits.seal_lockbox(snap)
    assert all(n >= 1 for n in lb["fabrics_per_band"].values())
    post = readiness.evaluate(snap)
    assert post["gates"]["R9"]["status"] == "PASS"

    files = splits.make_cv_splits(snap, synth["base"] / "splits", repeats=2, folds=3)
    assert len(files) == 6
    fab = splits.fabric_table(snap)
    related = fab[fab.linked_by != ""].groupby("provenance_id").fabric_id.apply(set)
    assert len(related) >= 4                                            # synthetic related pairs were grouped
    for f in files:
        df = pd.read_csv(f, dtype=str, keep_default_na=False)
        assert df.groupby("provenance_id").split.nunique().max() == 1
        assert df.groupby("fabric_id").split.nunique().max() == 1
        assert not set(df.fabric_id) & set(lb["fabrics"])
        assert set(df.split) == {"train", "val", "test"}
        for members in related:                                          # related fabrics share a split
            assert df[df.fabric_id.isin(members)].split.nunique() <= 1
    synth["files"] = files


def test_lockbox_tamper_detected(tmp_path):
    make_synthetic_dataset(tmp_path / "p4", fabrics_per_band=6, photos_per_fabric=1, seed=5, image_size=32)
    snap = snapshot.create_snapshot(tmp_path / "p4", tmp_path / "s")
    splits.seal_lockbox(snap)
    f = snap / "lockbox.json"
    writable(f)
    d = json.loads(f.read_text())
    d["fabrics"] = d["fabrics"][:-1]
    f.write_text(json.dumps(d))
    assert not splits.verify_lockbox(snap)["ok"]


def test_splits_refused_without_lockbox_or_with_too_few_groups(tmp_path):
    make_synthetic_dataset(tmp_path / "p4", fabrics_per_band=4, photos_per_fabric=1, seed=6, image_size=32, related_pairs=0)
    snap = snapshot.create_snapshot(tmp_path / "p4", tmp_path / "s")
    with pytest.raises(splits.SplitError, match="lockbox not sealed"):
        splits.make_cv_splits(snap, tmp_path / "x", repeats=1, folds=3)
    splits.seal_lockbox(snap)
    with pytest.raises(splits.SplitError, match="too few provenance groups"):
        splits.make_cv_splits(snap, tmp_path / "x", repeats=1, folds=5)


def test_check_split_catches_leaks():
    df = pd.DataFrame({"image_id": ["a", "b", "c"], "fabric_id": ["F1", "F1", "F2"], "provenance_id": ["P1", "P1", "P1"],
                       "split": ["train", "test", "train"], "band": ["low"] * 3, "sha256": ["1", "2", "3"]})
    chk = splits.check_split(df, {"F2"})
    assert not chk["ok"] and len(chk["problems"]) == 3


# ---- R8 review mechanism ------------------------------------------------------------ #
def test_r8_review_file_controls_gate(tmp_path):
    make_synthetic_dataset(tmp_path / "p4", fabrics_per_band=3, photos_per_fabric=1, seed=7, image_size=32)
    snap = snapshot.create_snapshot(tmp_path / "p4", tmp_path / "s")
    r = readiness.evaluate(snap)
    cand = pd.read_csv(snap / "readiness" / "lookalike_candidates.csv")
    if cand.empty:
        assert r["gates"]["R8"]["status"] == "PASS"
        return
    rev = cand[["image_a", "image_b"]].assign(decision="distinct", reviewer="QA", date="2026-11-25")
    rev.to_csv(snap / "readiness" / "lookalike_review.csv", index=False)
    assert readiness.evaluate(snap)["gates"]["R8"]["status"] == "PASS"
    rev.loc[rev.index[0], "decision"] = "same"
    rev.to_csv(snap / "readiness" / "lookalike_review.csv", index=False)
    assert readiness.evaluate(snap)["gates"]["R8"]["status"] == "FAIL"


def test_cramers_v():
    assert readiness.cramers_v(pd.Series(list("aabb")), pd.Series(list("xxyy"))) == pytest.approx(1.0)
    assert readiness.cramers_v(pd.Series(list("abab")), pd.Series(list("xxyy"))) == pytest.approx(0.0)


# ---- models, guard, training, evaluation ---------------------------------------------- #
@pytest.mark.parametrize("arch", M.ARCHITECTURES)
def test_architectures_build_without_downloading(arch, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("weights download attempted")
    monkeypatch.setattr(torch.hub, "load_state_dict_from_url", boom)
    m = M.build_model(arch, 4, pretrained=False).eval()
    with torch.no_grad():
        assert m(torch.randn(2, 3, 64, 64)).shape == (2, 4)
        assert M.penultimate_features(m, arch, torch.randn(1, 3, 64, 64)).ndim == 2
    groups = M.parameter_groups(m, arch, 1e-4, 1e-3)
    assert sum(len(g["params"]) for g in groups) == len(list(m.parameters()))


def test_real_snapshot_training_is_refused(tmp_path, monkeypatch):
    monkeypatch.setenv("BWORTH_PHASE4_ROOT", str(tmp_path / "real"))
    R.init_dataset(policy_version=2)                                     # not marked synthetic
    snap = snapshot.create_snapshot(tmp_path / "real", tmp_path / "s")
    assert not snapshot.load_meta(snap)["synthetic"]
    with pytest.raises(TrainingRefused) as e:
        train_split(tmp_path / "none.csv", snap, tmp_path / "run", approved=True)
    assert "lockbox" in str(e.value) and "readiness" in str(e.value)


def test_synthetic_training_run_and_backward_compatible_loading(synth):
    if "files" not in synth:
        pytest.skip("split test did not run first")
    out = synth["base"] / "run_mobilenet"
    info = train_split(synth["files"][0], synth["snap"], out, arch="mobilenet_v3_large", epochs=1,
                       image_size=64, resize_size=72, batch_size=16, device="cpu")
    assert info["mode"] == "synthetic_pipeline_test" and info["banner"] == SYNTHETIC_BANNER
    ck = torch.load(out / "model.pth", map_location="cpu", weights_only=False)
    assert ck["architecture"] == "mobilenet_v3_large" and ck["synthetic"] and ck["pipeline_test"]
    assert ck["lockbox_seal"] and ck["snapshot_version"] == "SYNTH-T"
    pred = pd.read_csv(out / "predictions_test.csv")
    assert set(pred.columns) >= {"fabric_id", "band", *E.PCOLS}
    assert np.allclose(pred[E.PCOLS].sum(axis=1), 1, atol=1e-4)
    from inference.composition.predict import CompositionPredictor

    p = CompositionPredictor(out / "model.pth", device="cpu")
    img = snapshot.image_root(synth["snap"]) / snapshot.load_tables(synth["snap"])["manifest"].image_path.iloc[0]
    r = p.predict(img)
    assert list(r.probabilities) == CLASSES and abs(sum(r.probabilities.values()) - 1) < 1e-4


def test_fabric_aggregation_and_metrics():
    pred = pd.DataFrame({"fabric_id": ["A", "A", "B", "C"], "band": ["low", "low", "very_high", "medium"],
                         "p_low": [0.9, 0.1, 0.7, 0.1], "p_medium": [0.1, 0.3, 0.1, 0.8],
                         "p_high": [0, 0.2, 0.1, 0.1], "p_very_high": [0, 0.4, 0.1, 0]})
    fab = E.aggregate_fabrics(pred).set_index("fabric_id")
    assert fab.p_low["A"] == pytest.approx(0.5) and fab.predicted["A"] == "low" and fab.n_images["A"] == 2
    m = E.fabric_metrics(fab.reset_index())
    assert m["far_band_error_rate"] == pytest.approx(1 / 3)           # B: very_high predicted low
    lo, hi = E.bootstrap_ci(fab.reset_index(), n=200)
    assert 0 <= lo <= hi <= 1


def test_success_criteria_mechanics_flag_synthetic():
    summary = {"balanced_accuracy_mean": 0.7, "repeat_std": 0.02, "pooled_ci95": [0.6, 0.8],
               "pooled": {"per_band": {c: {"recall": 0.7, "precision": 0.7} for c in CLASSES},
                          "far_band_error_rate": 0.0, "ece": 0.05}}
    r = E.check_success(summary, {"colour_histogram": 0.4, "metadata": 0.3}, lockbox_bal_acc=0.65, synthetic=True)
    assert r["all_pass"] and "PIPELINE TEST" in r["note"]
    r2 = E.check_success({**summary, "pooled_ci95": [0.3, 0.8]}, {"colour_histogram": 0.65, "metadata": 0.3})
    assert r2["criteria"]["S1"] is False and r2["criteria"]["S2"] is False
