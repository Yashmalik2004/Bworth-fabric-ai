"""Validate the Phase 4 verified fabric dataset, write the audit report and the verified training manifest.

    python -m data_collection.phase4.validate            # datasets/phase4_verified (or $BWORTH_PHASE4_ROOT)
    python -m data_collection.phase4.validate --root PATH

Severity:
  ERROR    data-integrity problem; the affected fabric/image is kept out of the training manifest
  PENDING  record incomplete or unverified (allowed to be saved; never enters the training manifest)
  WARNING  capture/quality problem worth fixing; does not by itself block training use
Exit code 1 if any ERROR exists.

Outputs:
  metadata/manifests/verified_training_manifest.csv   only VERIFIED, in-band fabrics; error-free images
  metadata/manifests/fabric_status.csv                 computed status + reasons for every fabric
  documentation/audit/dataset_audit_report.md          human-readable audit report
  documentation/audit/audit_results.json, issues.csv   machine-readable results
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from data_collection.phase4 import registry as R

NEAR_DUP_HAMMING = 8  # of 256 dHash bits (same threshold used in the CottonFabricImageBD audit)


def dhash_bits(path: Path) -> np.ndarray:
    from PIL import Image, ImageOps

    with Image.open(path) as im:
        g = np.asarray(ImageOps.exif_transpose(im).convert("L").resize((17, 16), Image.BILINEAR), dtype=np.int16)
    return (g[:, 1:] > g[:, :-1]).flatten()


def validate(root: Path | None = None) -> dict:
    p = R.paths(root)
    fab, img, ver = R.load(root)
    issues: list[dict] = []

    def add(sev, check, msg, fabric_id="", image_id="", record_id=""):
        issues.append({"severity": sev, "check": check, "fabric_id": fabric_id, "image_id": image_id,
                       "record_id": record_id, "message": msg})

    # ---- schema ------------------------------------------------------------ #
    for key, cols in (("fabrics", R.FABRIC_COLUMNS), ("images_csv", R.IMAGE_COLUMNS), ("verification", R.VERIFICATION_COLUMNS)):
        if not p[key].is_file():
            add("ERROR", "schema", f"missing file {p[key].name}")
            continue
        present = set(pd.read_csv(p[key], nrows=0).columns)
        for c in cols:
            if c not in present:
                add("ERROR", "schema", f"{p[key].name}: missing column '{c}'")

    # ---- fabric registry --------------------------------------------------- #
    for f in fab.itertuples():
        fid = f.fabric_id
        if not str(fid).strip():
            add("ERROR", "missing_fabric_id", "fabric registry row without fabric_id")
            continue
        if not R.FABRIC_ID_RE.match(fid):
            add("ERROR", "fabric_id_format", f"'{fid}' does not match BWF-0000", fid)
        cot = R.to_float(f.cotton_percentage)
        oth = R.to_float(f.other_fibre_percentage)
        if cot is None:
            add("PENDING", "missing_composition", "cotton_percentage is empty", fid)
        elif pd.isna(cot) or not 0 <= cot <= 100:
            add("ERROR", "invalid_percentage", f"cotton_percentage '{f.cotton_percentage}' is not a number in 0-100", fid)
        if oth is not None and (pd.isna(oth) or not 0 <= oth <= 100):
            add("ERROR", "invalid_percentage", f"other_fibre_percentage '{f.other_fibre_percentage}' is not a number in 0-100", fid)
        if cot is not None and oth is not None and not pd.isna(cot) and not pd.isna(oth) and abs(cot + oth - 100) > R.SUM_TOLERANCE:
            add("ERROR", "invalid_percentage", f"cotton {cot} + other fibres {oth} = {cot + oth}, not 100", fid)
        if cot is not None and oth is None and not pd.isna(cot) and cot < 100:
            add("PENDING", "missing_composition", "other_fibre_percentage is empty", fid)
        try:
            fibres = R.parse_fibres(f.other_fibre_composition)
            if fibres is not None and oth is not None and not pd.isna(oth) and abs(sum(fibres.values()) - oth) > R.SUM_TOLERANCE:
                add("ERROR", "invalid_percentage", f"other_fibre_composition sums to {sum(fibres.values())}, other_fibre_percentage is {oth}", fid)
            if fibres is not None and "cotton" in fibres:
                add("ERROR", "invalid_percentage", "other_fibre_composition must not list cotton", fid)
        except ValueError as exc:
            add("ERROR", "invalid_percentage", f"other_fibre_composition: {exc}", fid)
        if f.weave_or_knit and f.weave_or_knit not in R.WEAVE_OR_KNIT:
            add("ERROR", "vocabulary", f"weave_or_knit '{f.weave_or_knit}' not in {R.WEAVE_OR_KNIT}", fid)
        if f.verification_method and f.verification_method not in R.EVIDENCE_TYPES:
            add("ERROR", "vocabulary", f"verification_method '{f.verification_method}' not in {R.EVIDENCE_TYPES}", fid)
        if f.verification_document and not (p["root"] / f.verification_document).is_file():
            add("ERROR", "missing_document", f"verification_document '{f.verification_document}' not found", fid)
        if not str(f.source).strip():
            add("WARNING", "provenance", "source is empty", fid)
        if not str(f.physical_sample_location).strip():
            add("WARNING", "provenance", "physical_sample_location is empty (sample cannot be re-tested)", fid)

    dup = fab[fab.fabric_id.str.strip() != ""].groupby("fabric_id")
    for fid, g in dup:
        if len(g) > 1:
            vals = sorted(set(g.cotton_percentage))
            if len(vals) > 1:
                add("ERROR", "contradictory_labels", f"fabric_id appears {len(g)} times with different cotton values {vals}", fid)
            else:
                add("ERROR", "duplicate_fabric_id", f"fabric_id appears {len(g)} times in the registry", fid)
    known = set(fab.fabric_id)

    # ---- verification records ---------------------------------------------- #
    for r in ver.itertuples():
        if r.fabric_id not in known:
            add("ERROR", "missing_fabric_id", f"verification record refers to unknown fabric '{r.fabric_id}'", r.fabric_id, record_id=r.record_id)
        if r.evidence_type not in R.EVIDENCE_TYPES:
            add("ERROR", "vocabulary", f"evidence_type '{r.evidence_type}' not in {R.EVIDENCE_TYPES}", r.fabric_id, record_id=r.record_id)
        if r.review_decision and r.review_decision not in R.REVIEW_DECISIONS:
            add("ERROR", "vocabulary", f"review_decision '{r.review_decision}' not in {R.REVIEW_DECISIONS}", r.fabric_id, record_id=r.record_id)
        if r.document_path and not (p["root"] / r.document_path).is_file():
            add("ERROR", "missing_document", f"document '{r.document_path}' not found", r.fabric_id, record_id=r.record_id)
        if ver.record_id.tolist().count(r.record_id) > 1:
            add("ERROR", "duplicate_record_id", f"record_id {r.record_id} used more than once", r.fabric_id, record_id=r.record_id)

    status = R.fabric_statuses(fab.drop_duplicates("fabric_id"), ver, p["root"])
    for s in status.itertuples():
        if s.status == R.PENDING:
            add("PENDING", "missing_verification", s.reasons, s.fabric_id)
        elif s.status == R.CONFLICT:
            add("ERROR", "contradictory_labels", s.reasons, s.fabric_id)
        elif s.status == R.REJECTED:
            add("PENDING", "rejected_evidence", "all evidence was rejected: " + s.reasons, s.fabric_id)
        elif s.status == R.VERIFIED and s.band == "out_of_band":
            add("WARNING", "out_of_band", f"verified cotton {s.cotton_percentage}% lies outside the four task bands", s.fabric_id)

    # ---- image registry ---------------------------------------------------- #
    images_dir = p["images"].resolve()
    hashes = {}
    ok_images = []
    for im in img.itertuples():
        iid, fid = im.image_id, im.fabric_id
        bad = False
        if not R.IMAGE_ID_RE.match(str(iid)):
            add("ERROR", "image_id_format", f"'{iid}' does not match BWI-000000", fid, iid); bad = True
        if img.image_id.tolist().count(iid) > 1:
            add("ERROR", "duplicate_image_id", f"image_id {iid} used more than once", fid, iid); bad = True
        if not str(fid).strip():
            add("ERROR", "missing_fabric_id", "image without fabric_id", "", iid); bad = True
        elif fid not in known:
            add("ERROR", "missing_fabric_id", f"image refers to unknown fabric '{fid}'", fid, iid); bad = True
        path = (p["root"] / str(im.image_path)).resolve()
        if not str(im.image_path).strip() or not path.is_file():
            add("ERROR", "incorrect_image_path", f"image file not found: '{im.image_path}'", fid, iid); bad = True
            continue
        if images_dir not in path.parents:
            add("ERROR", "incorrect_image_path", f"image path is outside images/: '{im.image_path}'", fid, iid); bad = True
        elif path.parent.name != fid:
            add("WARNING", "image_location", f"image stored under images/{path.parent.name}/, not images/{fid}/", fid, iid)
        actual = R.sha256_file(path)
        if im.sha256 and im.sha256 != actual:
            add("ERROR", "hash_mismatch", "file content does not match recorded sha256 (file changed or wrong file)", fid, iid); bad = True
        if not im.sha256:
            add("ERROR", "hash_mismatch", "sha256 not recorded", fid, iid); bad = True
        hashes.setdefault(actual, []).append((iid, fid))
        try:
            from PIL import Image
            with Image.open(path) as pi:
                w, h = pi.size
            if im.resolution and im.resolution != f"{w}x{h}":
                add("WARNING", "resolution_mismatch", f"recorded {im.resolution}, file is {w}x{h}", fid, iid)
        except Exception as exc:  # noqa: BLE001
            add("ERROR", "unreadable_image", f"cannot read image: {exc}", fid, iid); bad = True
        if im.image_type not in R.IMAGE_TYPES:
            add("ERROR", "vocabulary", f"image_type '{im.image_type}' not in {R.IMAGE_TYPES}", fid, iid); bad = True
        if im.lighting and im.lighting not in R.LIGHTING:
            add("ERROR", "vocabulary", f"lighting '{im.lighting}' not in {R.LIGHTING}", fid, iid); bad = True
        if im.camera_angle and im.camera_angle not in R.CAMERA_ANGLES:
            add("ERROR", "vocabulary", f"camera_angle '{im.camera_angle}' not in {R.CAMERA_ANGLES}", fid, iid); bad = True
        if not str(im.capture_device).strip():
            add("WARNING", "capture_metadata", "capture_device is empty", fid, iid)
        if not bad:
            ok_images.append(iid)

    cot_of = dict(zip(fab.fabric_id, fab.cotton_percentage))
    for h, items in hashes.items():
        if len(items) > 1:
            fids = sorted({f for _, f in items})
            ids = ", ".join(i for i, _ in items)
            if len(fids) > 1 and len({cot_of.get(f) for f in fids}) > 1:
                for i, f in items:
                    add("ERROR", "contradictory_labels", f"identical photograph ({ids}) registered under fabrics {fids} with different compositions", f, i)
            else:
                for i, f in items:
                    add("ERROR", "duplicate_image_hash", f"identical photograph registered more than once ({ids}; fabrics {fids})", f, i)
    dup_imgs = {i for h, items in hashes.items() if len(items) > 1 for i, _ in items}

    # near-identical photographs filed under different fabrics (the CottonFabricImageBD failure mode)
    files = [(im.image_id, im.fabric_id, (p["root"] / im.image_path)) for im in img.itertuples()
             if str(im.image_path).strip() and (p["root"] / im.image_path).is_file()]
    if len(files) > 1:
        try:
            H = np.stack([dhash_bits(f) for _, _, f in files])
            D = (H[:, None, :] != H[None, :, :]).sum(-1)
            for a, b in zip(*np.triu_indices(len(files), 1)):
                if D[a, b] <= NEAR_DUP_HAMMING and files[a][1] != files[b][1] and files[a][0] not in dup_imgs:
                    add("WARNING", "near_duplicate_across_fabrics",
                        f"{files[a][0]} ({files[a][1]}) and {files[b][0]} ({files[b][1]}) look like the same photograph "
                        f"(dHash distance {int(D[a, b])}); check the fabric IDs", files[a][1], files[a][0])
        except Exception as exc:  # noqa: BLE001
            add("WARNING", "near_duplicate_check", f"near-duplicate check skipped: {exc}")

    # disk files not registered
    registered = {(p["root"] / x).resolve() for x in img.image_path if str(x).strip()}
    for f in p["images"].rglob("*"):
        if f.is_file() and f.resolve() not in registered and f.name != ".gitkeep":
            add("WARNING", "unregistered_file", f"file on disk not in image_registry: {f.relative_to(p['root']).as_posix()}")

    # ---- capture completeness ---------------------------------------------- #
    for fid in sorted(known):
        sub = img[img.fabric_id == fid]
        n = len(sub)
        if n < R.MIN_IMAGES_PER_FABRIC:
            add("WARNING", "capture_count", f"{n} photographs (protocol: {R.MIN_IMAGES_PER_FABRIC}-{R.MAX_IMAGES_PER_FABRIC})", fid)
        elif n > R.MAX_IMAGES_PER_FABRIC:
            add("WARNING", "capture_count", f"{n} photographs (protocol: {R.MIN_IMAGES_PER_FABRIC}-{R.MAX_IMAGES_PER_FABRIC}); extra photos add no independent samples", fid)
        for t in R.REQUIRED_IMAGE_TYPES:
            if n and t not in set(sub.image_type):
                add("WARNING", "capture_types", f"no '{t}' photograph", fid)

    # ---- manifest ---------------------------------------------------------- #
    iss = pd.DataFrame(issues, columns=["severity", "check", "fabric_id", "image_id", "record_id", "message"])
    fabric_errors = set(iss[(iss.severity == "ERROR") & (iss.image_id == "")].fabric_id)
    image_errors = set(iss[(iss.severity == "ERROR") & (iss.image_id != "")].image_id)
    eligible = status[(status.status == R.VERIFIED) & (status.band.isin(list(R.BANDS))) & (~status.fabric_id.isin(fabric_errors))]
    man = img[img.fabric_id.isin(eligible.fabric_id) & img.image_id.isin(ok_images) & ~img.image_id.isin(image_errors)]
    man = man.merge(eligible[["fabric_id", "cotton_percentage", "band", "accepted_records"]], on="fabric_id")
    man = man[["image_id", "fabric_id", "image_path", "cotton_percentage", "band", "image_type", "sha256", "accepted_records"]]
    man = man.rename(columns={"accepted_records": "verification_records"})
    status["blocking_errors"] = status.fabric_id.isin(fabric_errors)
    status["in_training_manifest"] = status.fabric_id.isin(man.fabric_id)

    # ---- band progress ----------------------------------------------------- #
    bands = {}
    for b in R.BANDS:
        bands[b] = {
            "verified_fabrics_in_manifest": int(man[man.band == b].fabric_id.nunique()),
            "verified_photographs_in_manifest": int((man.band == b).sum()),
            "pending_fabrics": int(((status.band == b) & (status.status != R.VERIFIED)).sum()),
            "target_fabrics": R.TARGET_FABRICS_PER_BAND,
        }
        bands[b]["sufficient"] = bands[b]["verified_fabrics_in_manifest"] >= R.TARGET_FABRICS_PER_BAND
        if not bands[b]["sufficient"]:
            add("WARNING", "insufficient_samples_per_band",
                f"{R.BAND_LABELS[b]}: {bands[b]['verified_fabrics_in_manifest']} verified independent fabrics, target {R.TARGET_FABRICS_PER_BAND}")
    iss = pd.DataFrame(issues, columns=["severity", "check", "fabric_id", "image_id", "record_id", "message"])

    return {"generated": datetime.now().isoformat(timespec="seconds"), "root": (p["root"].resolve().relative_to(R.PROJECT_ROOT).as_posix() if R.PROJECT_ROOT in p["root"].resolve().parents else str(p["root"])),
            "counts": {"fabrics": int(fab.fabric_id.nunique()), "fabric_rows": int(len(fab)), "images": int(len(img)),
                       "verification_records": int(len(ver)),
                       "fabrics_by_status": status.status.value_counts().to_dict(),
                       "fabrics_in_manifest": int(man.fabric_id.nunique()), "photographs_in_manifest": int(len(man))},
            "bands": bands, "issues": iss, "status": status, "manifest": man}


def write_outputs(res: dict, root: Path | None = None) -> dict[str, Path]:
    p = R.init_dataset(root)
    out = {"manifest": p["manifests"] / "verified_training_manifest.csv", "status": p["manifests"] / "fabric_status.csv",
           "report": p["audit"] / "dataset_audit_report.md", "json": p["audit"] / "audit_results.json",
           "issues": p["audit"] / "issues.csv"}
    R.write_csv(out["manifest"], res["manifest"])
    R.write_csv(out["status"], res["status"])
    R.write_csv(out["issues"], res["issues"])
    iss = res["issues"]
    summary = {k: v for k, v in res.items() if k not in ("issues", "status", "manifest")}
    summary["issues_by_severity"] = iss.severity.value_counts().to_dict()
    summary["issues_by_check"] = iss.groupby(["severity", "check"]).size().reset_index(name="n").to_dict(orient="records")
    out["json"].write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    c = res["counts"]
    lines = [
        "# Phase 4 verified fabric dataset: audit report", "",
        f"Generated {res['generated']} by `python -m data_collection.phase4.validate`. Do not edit by hand: re-run the validator.", "",
        "## Summary", "",
        "| | Count |", "|---|---|",
        f"| Physical fabrics registered | {c['fabrics']} |",
        f"| Photographs registered | {c['images']} |",
        f"| Verification records | {c['verification_records']} |",
        f"| **Fabrics in verified training manifest** | **{c['fabrics_in_manifest']}** |",
        f"| Photographs in verified training manifest | {c['photographs_in_manifest']} |",
        "", "Fabric status: " + (", ".join(f"{k} {v}" for k, v in c["fabrics_by_status"].items()) or "no fabrics registered yet"), "",
        "## Progress per band (independent physical fabrics, not photographs)", "",
        "| Band | Verified fabrics | Target | Pending | Photographs | Sufficient |", "|---|---|---|---|---|---|",
    ]
    for b, v in res["bands"].items():
        lines.append(f"| {R.BAND_LABELS[b]} | {v['verified_fabrics_in_manifest']} | {v['target_fabrics']} | {v['pending_fabrics']} | "
                     f"{v['verified_photographs_in_manifest']} | {'yes' if v['sufficient'] else '**no**'} |")
    lines += ["", "## Issues", ""]
    if iss.empty:
        lines.append("No issues.")
    else:
        lines += ["| Severity | Check | Count |", "|---|---|---|"]
        for (s, ch), n in iss.groupby(["severity", "check"]).size().items():
            lines.append(f"| {s} | {ch} | {n} |")
        lines += ["", "First 40 issues (all in `issues.csv`):", "", "| Severity | Check | Fabric | Image | Record | Message |",
                  "|---|---|---|---|---|---|"]
        for r in iss.head(40).itertuples():
            lines.append(f"| {r.severity} | {r.check} | {r.fabric_id} | {r.image_id} | {r.record_id} | {str(r.message).replace('|', '/')} |")
    lines += ["", "## Rules applied", "",
              "* A fabric enters the training manifest only if it is **VERIFIED** (accepted laboratory, manufacturer or "
              "product-identified supplier evidence, document on file, reviewer recorded, evidence agreeing within "
              f"{R.LAB_AGREEMENT_TOLERANCE} points), lies in a task band, and has no ERROR. Garment care labels never verify.",
              "* Each image must exist under `images/`, match its recorded SHA-256, and not duplicate another photograph.",
              "* Band targets count **independent physical fabrics**. Several photographs of one fabric are one sample.",
              "* Use `fabric_id` as the grouping key for any train/validation/test split."]
    out["report"].write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=None)
    a = ap.parse_args(argv)
    R.init_dataset(a.root)
    res = validate(a.root)
    out = write_outputs(res, a.root)
    sev = res["issues"].severity.value_counts().to_dict()
    print(f"fabrics {res['counts']['fabrics']} | images {res['counts']['images']} | in manifest: "
          f"{res['counts']['fabrics_in_manifest']} fabrics / {res['counts']['photographs_in_manifest']} photos | issues {sev}")
    for b, v in res["bands"].items():
        print(f"  {R.BAND_LABELS[b].replace(chr(8211), '-'):<20} verified {v['verified_fabrics_in_manifest']:>3}/{v['target_fabrics']}  pending {v['pending_fabrics']}")
    print(f"report: {out['report']}")
    return 1 if sev.get("ERROR", 0) else 0


if __name__ == "__main__":
    sys.exit(main())
