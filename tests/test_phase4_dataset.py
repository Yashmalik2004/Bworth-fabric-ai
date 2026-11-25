"""Phase 4 verified fabric dataset: registry, verification rules, validation checks, manifest, entry app."""

import io

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from data_collection.phase4 import registry as R
from data_collection.phase4 import validate as V


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setenv("BWORTH_PHASE4_ROOT", str(tmp_path / "phase4"))
    R.init_dataset()
    return tmp_path / "phase4"


def jpeg(seed, size=(64, 48), quality=92):
    arr = np.random.default_rng(seed).integers(0, 255, (size[1], size[0], 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def lab_evidence(fid, cotton, **kw):
    rec = {"fabric_id": fid, "evidence_type": "laboratory_test", "evidence_reference": "LAB-123",
           "issuing_party": "Test Lab", "reported_cotton_percentage": cotton, "test_method_standard": "ISO 1833-11",
           "reviewed_by": "QA", "review_decision": "accepted"}
    rec.update(kw)
    return R.add_verification(rec, document=("report.pdf", b"%PDF-1.4 test"))


def verified_fabric(cotton=65, other=35, seed=0, n_images=4):
    fid = R.add_fabric({"cotton_percentage": cotton, "other_fibre_percentage": other,
                        "other_fibre_composition": f"polyester:{other}" if other else "", "weave_or_knit": "woven",
                        "source": "Supplier A", "physical_sample_location": "Box 1"})
    lab_evidence(fid, cotton)
    types = ["full_fabric", "macro_texture", "natural_light", "alternate_angle", "controlled"]
    for k in range(n_images):
        R.add_image(fid, f"p{k}.jpg", jpeg(seed * 100 + k), {"image_type": types[k % 5], "lighting": "natural_daylight",
                                                            "camera_angle": "overhead_90", "capture_device": "phone"})
    return fid


def checks(res, sev=None):
    iss = res["issues"]
    return set(iss[iss.severity == sev].check) if sev else set(iss.check)


def test_templates_have_required_columns(root):
    p = R.paths()
    for key, cols in (("fabrics", R.FABRIC_COLUMNS), ("images_csv", R.IMAGE_COLUMNS), ("verification", R.VERIFICATION_COLUMNS)):
        assert list(pd.read_csv(p[key]).columns) == cols
    for c in ["fabric_id", "cotton_percentage", "other_fibre_percentage", "fabric_type", "weave_or_knit", "colour", "source",
              "verification_method", "verification_document", "collection_date", "notes"]:
        assert c in R.FABRIC_COLUMNS
    for c in ["image_id", "fabric_id", "image_path", "capture_device", "image_type", "lighting", "camera_angle",
              "resolution", "capture_date", "sha256"]:
        assert c in R.IMAGE_COLUMNS


@pytest.mark.parametrize("value,band", [(30, "low"), (40, "low"), (41, "out_of_band"), (50, "medium"), (66, "medium"),
                                        (80, "high"), (95, "high"), (97, "out_of_band"), (97.9, "out_of_band"),
                                        (98, "very_high"), (99, "very_high"), (99.5, "very_high"), (100, "very_high"),
                                        (100.0, "very_high"), ("100", None), (None, None)])
def test_band_mapping(value, band):
    if isinstance(value, str):                       # band_of expects numbers; registry strings go through to_float
        assert R.band_of(R.to_float(value)) == "very_high"
    else:
        assert R.band_of(value) == band


def test_very_high_band_definition_and_label():
    assert R.BANDS["very_high"] == (98.0, 100.0)
    assert R.BAND_LABELS["very_high"] == "Very high (98–100%)"
    assert R.TARGET_FABRICS_PER_BAND == 30


def test_verified_fabric_enters_manifest(root):
    fid = verified_fabric()
    res = V.validate()
    st = res["status"].set_index("fabric_id")
    assert st.status[fid] == R.VERIFIED and st.band[fid] == "medium"
    assert set(res["manifest"].fabric_id) == {fid} and len(res["manifest"]) == 4
    assert "ERROR" not in set(res["issues"].severity)


def test_incomplete_record_saved_as_pending_and_excluded(root):
    fid = R.add_fabric({"colour": "red"})                      # no composition at all
    res = V.validate()
    assert res["status"].set_index("fabric_id").status[fid] == R.INCOMPLETE
    assert "missing_composition" in checks(res, "PENDING")
    assert res["manifest"].empty


def test_care_label_never_verifies(root):
    fid = R.add_fabric({"cotton_percentage": 99, "other_fibre_percentage": 1})
    R.add_verification({"fabric_id": fid, "evidence_type": "garment_care_label", "evidence_reference": "label photo",
                        "reported_cotton_percentage": 99, "reviewed_by": "QA", "review_decision": "accepted"},
                       document=("label.jpg", jpeg(5)))
    res = V.validate()
    assert res["status"].set_index("fabric_id").status[fid] == R.PENDING
    assert "missing_verification" in checks(res, "PENDING") and res["manifest"].empty


@pytest.mark.parametrize("override,reason", [
    ({"evidence_type": "supplier_documentation", "product_identifier": ""}, "does not identify the product"),
    ({"review_decision": "pending"}, "review decision"),
    ({"reviewed_by": ""}, "reviewer"),
    ({"test_method_standard": ""}, "method"),
])
def test_incomplete_evidence_is_not_accepted(root, override, reason):
    fid = R.add_fabric({"cotton_percentage": 80, "other_fibre_percentage": 20})
    lab_evidence(fid, 80, **override)
    st = R.fabric_statuses(*[R.load()[i] for i in (0, 2)]).set_index("fabric_id")
    assert st.status[fid] == R.PENDING and reason in st.reasons[fid]


def test_evidence_without_document_is_not_accepted(root):
    fid = R.add_fabric({"cotton_percentage": 80, "other_fibre_percentage": 20})
    R.add_verification({"fabric_id": fid, "evidence_type": "laboratory_test", "evidence_reference": "LAB-1",
                        "reported_cotton_percentage": 80, "test_method_standard": "ISO 1833", "reviewed_by": "QA",
                        "review_decision": "accepted"})
    assert R.fabric_statuses(*[R.load()[i] for i in (0, 2)]).status.iloc[0] == R.PENDING


def test_conflicting_evidence_blocks_fabric(root):
    fid = verified_fabric(cotton=65, other=35, seed=1)
    lab_evidence(fid, 80)                                         # second lab disagrees by 15 points
    res = V.validate()
    assert res["status"].set_index("fabric_id").status[fid] == R.CONFLICT
    assert "contradictory_labels" in checks(res, "ERROR") and res["manifest"].empty


def test_registry_value_disagreeing_with_evidence_is_conflict(root):
    fid = R.add_fabric({"cotton_percentage": 65, "other_fibre_percentage": 35})
    lab_evidence(fid, 95)
    assert R.fabric_statuses(*[R.load()[i] for i in (0, 2)]).status.iloc[0] == R.CONFLICT


@pytest.mark.parametrize("fields,fragment", [
    ({"cotton_percentage": 120, "other_fibre_percentage": 0}, "not a number in 0-100"),
    ({"cotton_percentage": "abc", "other_fibre_percentage": 10}, "not a number in 0-100"),
    ({"cotton_percentage": 60, "other_fibre_percentage": 30}, "not 100"),
    ({"cotton_percentage": 60, "other_fibre_percentage": 40, "other_fibre_composition": "polyester:30"}, "sums to"),
    ({"cotton_percentage": 60, "other_fibre_percentage": 40, "other_fibre_composition": "polyester"}, "malformed"),
])
def test_invalid_percentages(root, fields, fragment):
    R.add_fabric(fields)
    iss = V.validate()["issues"]
    msgs = " ".join(iss[(iss.severity == "ERROR") & (iss.check == "invalid_percentage")].message)
    assert fragment in msgs


def test_duplicate_photo_rejected_at_entry(root):
    a = R.add_fabric({"cotton_percentage": 65, "other_fibre_percentage": 35})
    b = R.add_fabric({"cotton_percentage": 95, "other_fibre_percentage": 5})
    data = jpeg(7)
    R.add_image(a, "x.jpg", data, {"image_type": "full_fabric"})
    with pytest.raises(ValueError, match="duplicate photograph"):
        R.add_image(b, "x.jpg", data, {"image_type": "full_fabric"})


def test_same_photo_under_two_fabrics_with_different_labels_detected(root):
    a = verified_fabric(cotton=65, other=35, seed=2, n_images=1)
    b = verified_fabric(cotton=95, other=5, seed=3, n_images=1)
    fab, img, ver = R.load()
    row = img.iloc[0].copy()                                       # hand-edited CSV: copy a's photo row to fabric b
    row["image_id"], row["fabric_id"] = "BWI-000099", b
    R.write_csv(R.paths()["images_csv"], pd.concat([img, row.to_frame().T], ignore_index=True))
    res = V.validate()
    assert "contradictory_labels" in checks(res, "ERROR")
    assert "BWI-000099" not in set(res["manifest"].image_id)


def test_contradictory_fabric_id_rows(root):
    fid = R.add_fabric({"cotton_percentage": 65, "other_fibre_percentage": 35})
    fab = R.load()[0]
    dup = fab.iloc[0].copy(); dup["cotton_percentage"], dup["other_fibre_percentage"] = "95", "5"
    R.write_csv(R.paths()["fabrics"], pd.concat([fab, dup.to_frame().T], ignore_index=True))
    assert "contradictory_labels" in checks(V.validate(), "ERROR")


def test_image_path_hash_and_fabric_id_errors(root):
    fid = verified_fabric(seed=4)
    p = R.paths()
    fab, img, ver = R.load()
    img.loc[0, "image_path"] = "images/missing.jpg"                # wrong path
    img.loc[1, "fabric_id"] = "BWF-9999"                           # unknown fabric
    (p["root"] / img.loc[2, "image_path"]).write_bytes(jpeg(999))  # file replaced -> hash mismatch
    R.write_csv(p["images_csv"], img)
    res = V.validate()
    errs = checks(res, "ERROR")
    assert {"incorrect_image_path", "missing_fabric_id", "hash_mismatch"} <= errs
    assert len(res["manifest"]) == 1                               # only the untouched photo survives


def test_near_duplicate_across_fabrics_warned(root):
    a = R.add_fabric({"cotton_percentage": 65, "other_fibre_percentage": 35})
    b = R.add_fabric({"cotton_percentage": 95, "other_fibre_percentage": 5})
    base = np.random.default_rng(11).integers(0, 255, (48, 64, 3), dtype=np.uint8)
    for fid, q in ((a, 95), (b, 70)):                              # same picture, different JPEG encoding
        buf = io.BytesIO(); Image.fromarray(base).save(buf, format="JPEG", quality=q)
        R.add_image(fid, "x.jpg", buf.getvalue(), {"image_type": "full_fabric"})
    assert "near_duplicate_across_fabrics" in checks(V.validate(), "WARNING")


def test_insufficient_samples_and_capture_warnings(root):
    verified_fabric(seed=5, n_images=2)
    res = V.validate()
    w = checks(res, "WARNING")
    assert {"insufficient_samples_per_band", "capture_count"} <= w
    assert all(not v["sufficient"] for v in res["bands"].values())


def test_validator_outputs_and_exit_code(root):
    verified_fabric(seed=6)
    assert V.main(["--root", str(root)]) == 0
    p = R.paths()
    for f in ("dataset_audit_report.md", "audit_results.json", "issues.csv"):
        assert (p["audit"] / f).is_file()
    assert (p["manifests"] / "verified_training_manifest.csv").is_file()
    R.add_fabric({"cotton_percentage": 150})
    assert V.main(["--root", str(root)]) == 1


def test_entry_app_registers_pending_fabric(root):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(R.PROJECT_ROOT / "data_collection" / "phase4" / "entry_app.py"), default_timeout=120).run()
    assert not at.exception, at.exception
    at.text_input[0].set_value("65")
    at.text_input[1].set_value("35")
    at.button[0].click().run()
    assert not at.exception, at.exception
    fab = R.load()[0]
    assert len(fab) == 1 and fab.cotton_percentage.iloc[0] == "65"
    assert any("pending verification" in s.value for s in at.success)


# ---- Very high band extended to 100 % (Phase 4 update) --------------------- #
@pytest.mark.parametrize("cotton,other,comp", [(100, 0, ""), (100, "", ""), (98, 2, "polyester:2"), (99, 1, "elastane:1")])
def test_valid_very_high_compositions(root, cotton, other, comp):
    R.add_fabric({"cotton_percentage": cotton, "other_fibre_percentage": other, "other_fibre_composition": comp})
    iss = V.validate()["issues"]
    assert "invalid_percentage" not in set(iss.check)


@pytest.mark.parametrize("cotton,other,comp", [(100, 5, "polyester:5"), (100, 0, "polyester:5"), (98, 5, "polyester:5")])
def test_invalid_fibre_totals_rejected(root, cotton, other, comp):
    R.add_fabric({"cotton_percentage": cotton, "other_fibre_percentage": other, "other_fibre_composition": comp})
    res = V.validate()
    assert "invalid_percentage" in checks(res, "ERROR")
    assert res["manifest"].empty


def test_100_percent_cotton_still_requires_accepted_evidence(root):
    plain = R.add_fabric({"cotton_percentage": 100, "other_fibre_percentage": 0})
    labelled = R.add_fabric({"cotton_percentage": 100, "other_fibre_percentage": 0})
    R.add_verification({"fabric_id": labelled, "evidence_type": "garment_care_label", "evidence_reference": "label",
                        "reported_cotton_percentage": 100, "reviewed_by": "QA", "review_decision": "accepted"},
                       document=("label.jpg", jpeg(21)))
    res = V.validate()
    st = res["status"].set_index("fabric_id")
    assert st.status[plain] == R.PENDING and st.status[labelled] == R.PENDING
    assert st.band[plain] == "very_high"                         # in band, but not verified
    assert res["manifest"].empty


def test_verified_100_percent_cotton_in_manifest_and_counted(root):
    fid = verified_fabric(cotton=100, other=0, seed=30)
    res = V.validate()
    st = res["status"].set_index("fabric_id")
    assert st.status[fid] == R.VERIFIED and st.band[fid] == "very_high"
    man = res["manifest"]
    assert set(man.fabric_id) == {fid} and set(man.band) == {"very_high"} and len(man) == 4
    assert res["bands"]["very_high"]["verified_fabrics_in_manifest"] == 1
    assert "out_of_band" not in set(res["issues"].check)


def test_verified_97_percent_is_not_very_high(root):
    fid = verified_fabric(cotton=97, other=3, seed=31)
    res = V.validate()
    assert res["status"].set_index("fabric_id").band[fid] == "out_of_band"
    assert "out_of_band" in checks(res, "WARNING")
    assert res["manifest"].empty and res["bands"]["very_high"]["verified_fabrics_in_manifest"] == 0
