"""Protocol v1.1: R7 adjacent-band boundary rule, protocol versioning, R8 embedding-assisted review."""

import hashlib
import io
import json

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from data_collection.phase4 import registry as R
from training.phase5 import embeddings, readiness, snapshot
from training.phase5.protocol import PHASE5_DIR, load_protocol, protocol_sha256, protocol_version
from training.phase5.synthetic import make_synthetic_dataset


# ---- protocol version and history ------------------------------------------------ #
def test_protocol_is_v11_and_v10_is_archived_unchanged():
    p = load_protocol()
    assert protocol_version() == "1.1" and p["protocol_version"] == "1.1"
    g = p["readiness_gates"]
    assert g["R7_boundary_distance_pct_points"] == 3.0 and g["R7_max_boundary_share_per_band"] == 0.20
    assert g["R7_definition"].startswith("adjacent_band_distance")
    old = PHASE5_DIR / "protocol_history" / "protocol_v1.0.json"
    v10 = next(c for c in p["changelog"] if c["version"] == "1.0")
    assert hashlib.sha256(old.read_bytes()).hexdigest() == v10["sha256"]          # archived record intact
    assert json.loads(old.read_text())["protocol_version"] == "1.0"
    assert protocol_sha256() != v10["sha256"]
    md = (PHASE5_DIR / "PHASE5_PROTOCOL.md").read_text(encoding="utf-8")
    assert "Version 1.1" in md and "adjacent-band risk" in md


# ---- R7: all four bands ------------------------------------------------------------ #
@pytest.mark.parametrize("cotton,band,at_risk,dist", [
    (30.0, "low", False, 20.0), (35.0, "low", False, 15.0), (40.0, "low", False, 10.0),
    (50.0, "medium", False, 10.0), (58.0, "medium", False, 18.0), (66.0, "medium", False, 14.0),
    (80.0, "high", False, 14.0), (92.0, "high", False, 6.0), (94.9, "high", False, 3.1), (95.0, "high", True, 3.0),
    (98.0, "very_high", True, 3.0), (98.1, "very_high", False, 3.1), (98.5, "very_high", False, 3.5),
    (99.0, "very_high", False, 4.0), (100.0, "very_high", False, 5.0),
])
def test_r7_adjacent_band_rule_all_bands(cotton, band, at_risk, dist):
    assert readiness.adjacent_band_distance(cotton, band) == pytest.approx(dist)
    assert bool(readiness.boundary_flags(pd.Series([cotton]), pd.Series([band]), 3.0).iloc[0]) is at_risk


def test_r7_99_percent_never_fails_and_wider_distance_still_detects_risk():
    vh = readiness.boundary_flags(pd.Series([99.0] * 10), pd.Series(["very_high"] * 10), 3.0)
    assert not vh.any()                                                    # 99 % fabrics cannot fail R7 at 3 points
    wide = readiness.boundary_flags(pd.Series([40.0, 50.0, 66.0, 80.0]),
                                    pd.Series(["low", "medium", "medium", "high"]), 14.0)
    assert wide.tolist() == [True, True, True, True]                       # rule generalises to any distance
    assert readiness.boundary_flags(pd.Series([39.9]), pd.Series(["low"]), 10.0).iloc[0] == False  # noqa: E712


def test_r7_gap_edges_are_informational_only():
    c, b = pd.Series([30.0, 40.0, 99.0]), pd.Series(["low", "low", "very_high"])
    assert readiness.gap_edge_proximity(c, b, 3.0).tolist() == [True, True, True]
    assert readiness.boundary_flags(c, b, 3.0).tolist() == [False, False, False]


def test_r7_gate_share_on_snapshot(tmp_path):
    make_synthetic_dataset(tmp_path / "p4", fabrics_per_band=5, photos_per_fabric=1, seed=11, image_size=32)
    fab = R.load(tmp_path / "p4")[0]
    vh = fab.index[fab.notes.str.contains("SYNTHETIC")][-5:]               # last 5 = very_high band
    fab.loc[vh, "cotton_percentage"] = ["98.0", "99.0", "99.5", "100.0", "100.0"]
    fab.loc[vh, "other_fibre_percentage"] = ["2.0", "1.0", "0.5", "0.0", "0.0"]
    fab.loc[vh, "other_fibre_composition"] = ["polyester:2.0", "polyester:1.0", "polyester:0.5", "", ""]
    R.write_csv(R.paths(tmp_path / "p4")["fabrics"], fab)
    ver = R.load(tmp_path / "p4")[2]
    ver = ver.merge(fab[["fabric_id", "cotton_percentage"]], on="fabric_id", suffixes=("", "_f"))
    ver["reported_cotton_percentage"] = ver.cotton_percentage
    R.write_csv(R.paths(tmp_path / "p4")["verification"], ver[R.VERIFICATION_COLUMNS])
    snap = snapshot.create_snapshot(tmp_path / "p4", tmp_path / "s")
    r7 = readiness.evaluate(snap)["gates"]["R7"]
    assert r7["boundary_share_per_band"]["very_high"] == pytest.approx(0.2)   # only the 98.0 fabric
    assert r7["status"] == "PASS"


# ---- snapshot records protocol version and detects a new one ------------------------ #
def test_snapshot_records_protocol_version_and_detects_change(tmp_path, monkeypatch):
    make_synthetic_dataset(tmp_path / "p4", fabrics_per_band=1, photos_per_fabric=1, seed=12, image_size=32)
    snap = snapshot.create_snapshot(tmp_path / "p4", tmp_path / "s")
    meta = snapshot.load_meta(snap)
    assert meta["protocol_version"] == "1.1" and meta["protocol_sha256"] == protocol_sha256()
    assert snapshot.verify_snapshot(snap)["ok"]
    monkeypatch.setattr(snapshot, "protocol_sha256", lambda: "f" * 64)
    monkeypatch.setattr(snapshot, "protocol_version", lambda: "1.2")
    v = snapshot.verify_snapshot(snap)
    assert not v["ok"] and any("v1.1" in x and "v1.2" in x for x in v["problems"])
    assert snapshot.load_meta(snap)["protocol_version"] == "1.1"           # record is not rewritten


# ---- R8 ------------------------------------------------------------------------------ #
def _two_fabric_lookalike(tmp_path):
    """Synthetic dataset where two different fabrics carry near-identical photographs."""
    root = tmp_path / "p4"
    make_synthetic_dataset(root, fabrics_per_band=2, photos_per_fabric=1, seed=13, image_size=48, related_pairs=0)
    fab, img, _ = R.load(root)
    a, b = img.iloc[0], img[img.fabric_id != img.iloc[0].fabric_id].iloc[0]
    src = np.asarray(Image.open(root / a.image_path).convert("RGB")).astype(np.int16)
    noisy = np.clip(src + np.random.default_rng(1).integers(-2, 3, src.shape), 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(noisy).save(buf, format="PNG")
    (root / b.image_path).write_bytes(buf.getvalue())
    img.loc[img.image_id == b.image_id, "sha256"] = hashlib.sha256(buf.getvalue()).hexdigest()
    R.write_csv(R.paths(root)["images_csv"], img)
    return root, a.image_id, b.image_id, a.fabric_id, b.fabric_id


@pytest.mark.skipif(embeddings.cached_weights_path() is None, reason="cached ResNet-50 weights not available")
def test_r8_embedding_flags_with_scores_and_reasons_and_no_auto_merge(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.hub, "load_state_dict_from_url",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("download attempted")))
    root, ia, ib, fa, fb = _two_fabric_lookalike(tmp_path)
    registry_before = R.paths(root)["fabrics"].read_bytes()
    snap = snapshot.create_snapshot(root, tmp_path / "s")
    r8 = readiness.evaluate(snap)["gates"]["R8"]
    assert r8["embedding_check"]["available"] and "never downloaded" in r8["embedding_check"]["note"]
    cand = pd.read_csv(snap / "readiness" / "lookalike_candidates.csv")
    row = cand[((cand.image_a == ia) & (cand.image_b == ib)) | ((cand.image_a == ib) & (cand.image_b == ia))]
    assert len(row) == 1
    row = row.iloc[0]
    assert row.embedding_similarity >= 0.85 and row.colour_structure_similarity >= 0.92
    assert "embedding similarity" in row.reasons and "colour/structure similarity" in row.reasons
    assert r8["status"] == "FAIL" and r8["undecided"] >= 1                 # needs a human decision
    assert R.paths(root)["fabrics"].read_bytes() == registry_before        # nothing merged automatically
    # a fabric-pair decision resolves every image pair of those two fabrics
    pd.DataFrame([{"fabric_a": fa, "fabric_b": fb, "decision": "distinct", "reviewer": "QA", "date": "2026-11-25"}] +
                 [{"image_a": r.image_a, "image_b": r.image_b, "decision": "distinct", "reviewer": "QA", "date": "2026-11-25"}
                  for r in cand.itertuples() if {r.fabric_a, r.fabric_b} != {fa, fb}]
                 ).to_csv(snap / "readiness" / "lookalike_review.csv", index=False)
    assert readiness.evaluate(snap)["gates"]["R8"]["status"] == "PASS"
    pd.DataFrame([{"image_a": ia, "image_b": ib, "decision": "same", "reviewer": "QA", "date": "2026-11-25"}]
                 ).to_csv(snap / "readiness" / "lookalike_review.csv", index=False)
    assert readiness.evaluate(snap)["gates"]["R8"]["decided_same_fabric"] == 1


def test_r8_falls_back_without_cached_weights_and_never_downloads(tmp_path, monkeypatch):
    monkeypatch.setattr(embeddings, "cached_weights_path", lambda: None)
    monkeypatch.setattr(torch.hub, "load_state_dict_from_url",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("download attempted")))
    root, ia, ib, *_ = _two_fabric_lookalike(tmp_path)
    snap = snapshot.create_snapshot(root, tmp_path / "s")
    r8 = readiness.evaluate(snap)["gates"]["R8"]
    assert not r8["embedding_check"]["available"] and "unavailable" in r8["embedding_check"]["note"]
    assert r8["flagged_by_embedding"] == 0 and r8["flagged_by_colour_structure"] >= 1
    cand = pd.read_csv(snap / "readiness" / "lookalike_candidates.csv")
    assert cand.embedding_similarity.isna().all() and cand.reasons.str.contains("colour/structure").all()


def test_review_decisions_both_orders_and_ignores_invalid():
    rev = pd.DataFrame([{"image_a": "I1", "image_b": "I2", "decision": "Distinct"},
                        {"fabric_a": "F1", "fabric_b": "F2", "decision": "same"},
                        {"image_a": "I3", "image_b": "I4", "decision": "maybe"}])
    d = readiness.review_decisions(rev)
    assert readiness.decision_for({"image_a": "I2", "image_b": "I1", "fabric_a": "X", "fabric_b": "Y"}, d) == "distinct"
    assert readiness.decision_for({"image_a": "Z", "image_b": "W", "fabric_a": "F2", "fabric_b": "F1"}, d) == "same"
    assert readiness.decision_for({"image_a": "I3", "image_b": "I4", "fabric_a": "A", "fabric_b": "B"}, d) is None


# ---- synthetic learning test guards ------------------------------------------------- #
def test_recipe_overrides_refused_for_real_snapshot(tmp_path, monkeypatch):
    from training.phase5.train import TrainingRefused, train_split

    monkeypatch.setenv("BWORTH_PHASE4_ROOT", str(tmp_path / "real"))
    R.init_dataset(policy_version=2)
    snap = snapshot.create_snapshot(tmp_path / "real", tmp_path / "s")
    with pytest.raises(TrainingRefused):
        train_split(tmp_path / "x.csv", snap, tmp_path / "run", approved=True, recipe_overrides={"lr_backbone": 1e-3})


def test_synthetic_learning_refuses_real_folders_and_texture_encodes_class():
    from training.phase5 import synthetic_learning as SL

    for bad in (R.PROJECT_ROOT / "datasets" / "x", R.PROJECT_ROOT / "models" / "x"):
        with pytest.raises(ValueError, match="real dataset/model"):
            SL.run(bad)
    imgs = [np.asarray(Image.open(io.BytesIO(SL.learnable_texture(b, "grey", np.random.default_rng(0), 96))).convert("L"),
                       dtype=float) for b in range(4)]
    energy = [np.abs(np.diff(i, axis=1)).mean() + np.abs(np.diff(i, axis=0)).mean() for i in imgs]
    assert energy == sorted(energy)                                       # higher class -> finer stripes
    assert "SYNTHETIC" in SL.BANNER and "NOT evidence" in SL.BANNER
