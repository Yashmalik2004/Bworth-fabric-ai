"""Phase 4 schema v2 / verification policy v2: backward compatibility, migration, new checks."""

import io
import json

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from data_collection.phase4 import registry as R
from data_collection.phase4 import validate as V


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setenv("BWORTH_PHASE4_ROOT", str(tmp_path / "p4"))
    return tmp_path / "p4"


def png(seed):
    buf = io.BytesIO()
    Image.fromarray(np.random.default_rng(seed).integers(0, 255, (40, 40, 3), dtype=np.uint8)).save(buf, format="PNG")
    return buf.getvalue()


def lab(fid, cotton, reviewer="Ann", second=None, decision2="accepted", product=""):
    rid = R.add_verification({"fabric_id": fid, "evidence_type": "laboratory_test", "evidence_reference": "L1",
                              "reported_cotton_percentage": cotton, "test_method_standard": "ISO 1833",
                              "reviewed_by": reviewer, "review_decision": "accepted", "product_identifier": product},
                             document=("r.pdf", b"%PDF test"))
    if second:
        R.add_second_review(rid, second, decision2)
    return rid


def status(fid):
    fab, _, ver = R.load()
    return R.fabric_statuses(fab, ver).set_index("fabric_id").status[fid]


def checks(res, sev):
    i = res["issues"]
    return set(i[i.severity == sev].check)


# ---- legacy datasets keep working ------------------------------------------ #
def test_legacy_dataset_is_policy_v1_and_single_review_still_verifies(root):
    R.init_dataset()
    assert R.read_schema()["legacy"] and R.policy_version() == 1
    fid = R.add_fabric({"cotton_percentage": 65, "other_fibre_percentage": 35})
    lab(fid, 65)
    assert status(fid) == R.VERIFIED
    res = V.validate()
    assert "ERROR" not in set(res["issues"].severity)
    assert "legacy_verification_policy" in checks(res, "WARNING")


def test_legacy_csv_without_new_columns_is_not_invalid(root):
    p = R.paths()
    p["metadata"].mkdir(parents=True)
    pd.DataFrame([{**{c: "" for c in R.V1_COLUMNS["fabrics"]}, "fabric_id": "BWF-0001", "cotton_percentage": "65",
                   "other_fibre_percentage": "35"}]).to_csv(p["fabrics"], index=False)
    pd.DataFrame(columns=R.V1_COLUMNS["images_csv"]).to_csv(p["images_csv"], index=False)
    pd.DataFrame(columns=R.V1_COLUMNS["verification"]).to_csv(p["verification"], index=False)
    res = V.validate()
    assert "ERROR" not in set(res["issues"].severity)            # missing v2 columns never an error
    assert "legacy_schema" in checks(res, "WARNING")
    before = p["fabrics"].read_text()
    R.add_fabric({"cotton_percentage": 80, "other_fibre_percentage": 20})   # write path adds columns explicitly
    fab = pd.read_csv(p["fabrics"], dtype=str, keep_default_na=False)
    assert {"provenance_group", "source_product_id"} <= set(fab.columns)
    assert fab.set_index("fabric_id").cotton_percentage["BWF-0001"] == "65"   # existing value unchanged
    assert list(p["migration_backups"].iterdir()), "backup written before adding columns"
    assert (p["metadata"] / "migration_log.jsonl").is_file()
    assert R.policy_version() == 1                                             # columns added, policy NOT changed
    assert any(before.strip() in (b / "fabric_registry.csv").read_text() for b in p["migration_backups"].iterdir())


# ---- explicit migration ---------------------------------------------------- #
def test_migration_dry_run_writes_nothing_and_lists_status_changes(root):
    R.init_dataset()
    fid = R.add_fabric({"cotton_percentage": 65, "other_fibre_percentage": 35})
    lab(fid, 65)
    snap = {k: R.paths()[k].read_bytes() for k in ("fabrics", "images_csv", "verification")}
    rep = R.migrate(dry_run=True)
    assert rep["from_policy"] == 1 and rep["to_policy"] == 2
    assert rep["status_changes"] == [{"fabric_id": fid, "status_before": R.VERIFIED, "status_after": R.PENDING,
                                      "reasons_after": rep["status_changes"][0]["reasons_after"]}]
    assert "second review" in rep["status_changes"][0]["reasons_after"]
    assert {k: R.paths()[k].read_bytes() for k in snap} == snap and not R.paths()["schema"].is_file()


def test_migration_apply_then_second_review(root):
    R.init_dataset()
    fid = R.add_fabric({"cotton_percentage": 65, "other_fibre_percentage": 35})
    rid = lab(fid, 65, reviewer="Ann")
    R.migrate(dry_run=False)
    assert R.policy_version() == 2 and not R.read_schema()["legacy"]
    assert status(fid) == R.PENDING                                   # explicit, reported change
    with pytest.raises(ValueError, match="different person"):
        R.add_second_review(rid, "ann", "accepted")                   # case-insensitive same person
    R.add_second_review(rid, "Bob", "accepted")
    assert status(fid) == R.VERIFIED
    log = [json.loads(x) for x in (R.paths()["metadata"] / "migration_log.jsonl").read_text().splitlines()]
    assert any(e["action"] == "migrate" for e in log)


def test_second_review_rejection_and_same_person_in_csv(root):
    R.init_dataset(policy_version=2)
    fid = R.add_fabric({"cotton_percentage": 90, "other_fibre_percentage": 10})
    lab(fid, 90, second="Bob", decision2="rejected")
    assert status(fid) == R.REJECTED
    _, _, ver = R.load()
    ver.loc[0, ["second_reviewed_by", "second_review_decision"]] = ["Ann", "accepted"]   # hand-edited CSV
    R.write_csv(R.paths()["verification"], ver)
    res = V.validate()
    assert "second_review_not_independent" in checks(res, "ERROR")
    assert status(fid) != R.VERIFIED


# ---- new metadata and cross-fabric checks ---------------------------------- #
def test_new_fields_flow_into_manifest(root):
    R.init_dataset(policy_version=2)
    fid = R.add_fabric({"cotton_percentage": 99, "other_fibre_percentage": 1, "provenance_group": "PG-7",
                        "source_product_id": "ART-1"})
    lab(fid, 99, second="Bob")
    for k, t in enumerate(["full_fabric", "macro_texture", "natural_light", "alternate_angle"]):
        R.add_image(fid, f"{k}.png", png(k), {"image_type": t, "capture_session": "S1", "capture_device": "cam"})
    man = V.validate()["manifest"]
    assert set(man.provenance_group) == {"PG-7"} and set(man.source_product_id) == {"ART-1"}
    assert set(man.capture_session) == {"S1"} and set(man.band) == {"very_high"}


def test_v2_warns_on_missing_session_and_provenance(root):
    R.init_dataset(policy_version=2)
    fid = R.add_fabric({"cotton_percentage": 60, "other_fibre_percentage": 40})
    R.add_image(fid, "a.png", png(1), {"image_type": "full_fabric"})
    msgs = " ".join(V.validate()["issues"].message)
    assert "capture_session is empty" in msgs and "provenance_group" in msgs


def test_shared_product_identifier_warning_and_grouping(root):
    R.init_dataset(policy_version=2)
    a = R.add_fabric({"cotton_percentage": 60, "other_fibre_percentage": 40, "source_product_id": "ART-9"})
    b = R.add_fabric({"cotton_percentage": 60, "other_fibre_percentage": 40, "source_product_id": " art-9 "})
    assert "shared_product_identifier" in checks(V.validate(), "WARNING")
    fab = R.load()[0]
    fab["provenance_group"] = "PG-1"                                  # declared as one bolt -> no warning
    R.write_csv(R.paths()["fabrics"], fab)
    assert "shared_product_identifier" not in checks(V.validate(), "WARNING")


def test_shared_product_identifier_with_conflicting_verified_labels_is_error(root):
    R.init_dataset(policy_version=2)
    a = R.add_fabric({"cotton_percentage": 60, "other_fibre_percentage": 40})
    b = R.add_fabric({"cotton_percentage": 90, "other_fibre_percentage": 10})
    lab(a, 60, second="Bob", product="SKU-123")
    lab(b, 90, second="Bob", product="SKU-123")
    assert "contradictory_labels" in checks(V.validate(), "ERROR")


def test_entry_app_new_dataset_uses_policy_v2(root):
    from streamlit.testing.v1 import AppTest

    app = str(R.PROJECT_ROOT / "data_collection" / "phase4" / "entry_app.py")
    at = AppTest.from_file(app, default_timeout=120).run()
    assert not at.exception, at.exception
    assert R.policy_version() == 2                                    # brand-new dataset created under policy v2
    R.add_fabric({"cotton_percentage": 65, "other_fibre_percentage": 35})   # photo tab renders once a fabric exists
    at = AppTest.from_file(app, default_timeout=120).run()
    assert not at.exception, at.exception
    labels = [t.label for t in at.text_input]
    assert any("Provenance group" in x for x in labels) and any("Capture session" in x for x in labels)
