"""Phase 4 verified fabric dataset: schema, storage and verification-status rules.

Everything that decides whether a fabric may be used for training lives here, so the data-entry
interface, the validator and the manifest builder all apply the same rules.

Dataset root: datasets/phase4_verified (override with env var BWORTH_PHASE4_ROOT, used by tests).
"""

from __future__ import annotations

import csv
import hashlib
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def dataset_root() -> Path:
    return Path(os.environ.get("BWORTH_PHASE4_ROOT", PROJECT_ROOT / "datasets" / "phase4_verified"))


def paths(root: Path | None = None) -> dict[str, Path]:
    r = Path(root) if root else dataset_root()
    return {
        "root": r,
        "images": r / "images",
        "metadata": r / "metadata",
        "documents": r / "metadata" / "verification_documents",
        "manifests": r / "metadata" / "manifests",
        "documentation": r / "documentation",
        "audit": r / "documentation" / "audit",
        "fabrics": r / "metadata" / "fabric_registry.csv",
        "images_csv": r / "metadata" / "image_registry.csv",
        "verification": r / "metadata" / "verification_records.csv",
    }


# --------------------------------------------------------------------------- #
# Schema and controlled vocabularies
# --------------------------------------------------------------------------- #
FABRIC_COLUMNS = [
    "fabric_id", "cotton_percentage", "other_fibre_percentage", "fabric_type", "weave_or_knit", "colour",
    "source", "verification_method", "verification_document", "collection_date", "notes",
    # extensions (documented in DATA_DICTIONARY.md)
    "other_fibre_composition",   # e.g. "polyester:35|elastane:5"; must sum to other_fibre_percentage
    "physical_sample_location",  # where the physical swatch is stored (needed for re-testing)
]
IMAGE_COLUMNS = [
    "image_id", "fabric_id", "image_path", "capture_device", "image_type", "lighting", "camera_angle",
    "resolution", "capture_date", "sha256",
    "notes",
]
VERIFICATION_COLUMNS = [
    "record_id", "fabric_id", "evidence_type", "evidence_reference", "issuing_party", "product_identifier",
    "document_path", "reported_cotton_percentage", "reported_other_fibres", "test_method_standard",
    "reviewed_by", "review_date", "review_decision", "notes",
]

FABRIC_ID_RE = re.compile(r"^BWF-\d{4}$")
IMAGE_ID_RE = re.compile(r"^BWI-\d{6}$")
RECORD_ID_RE = re.compile(r"^BWV-\d{5}$")

WEAVE_OR_KNIT = ["woven", "knit", "non_woven", "unknown"]
IMAGE_TYPES = ["full_fabric", "macro_texture", "natural_light", "alternate_angle", "controlled"]
REQUIRED_IMAGE_TYPES = ["full_fabric", "macro_texture"]
LIGHTING = ["natural_daylight", "controlled_led_5000k", "controlled_other", "indoor_mixed", "unknown"]
CAMERA_ANGLES = ["overhead_90", "angled_45", "angled_other", "unknown"]

# Evidence that CAN support a composition label (VERIFICATION_PROTOCOL.md §2)
ACCEPTED_EVIDENCE = {
    "laboratory_test": "Laboratory fibre-composition test (e.g. ISO 1833 / AATCC 20A) with report number",
    "manufacturer_documentation": "Traceable manufacturer documentation (technical data sheet / certificate) for the identified product",
    "supplier_documentation": "Supplier documentation (invoice/spec sheet) that identifies this exact product/lot",
}
# Evidence that can be RECORDED but never verifies a label on its own
NON_ACCEPTED_EVIDENCE = {
    "garment_care_label": "Printed care/composition label on a garment (not trusted: labels are often wrong)",
    "seller_listing": "Online listing or verbal claim without product documentation",
    "visual_estimate": "Visual or tactile judgement",
    "none": "No evidence",
}
EVIDENCE_TYPES = list(ACCEPTED_EVIDENCE) + list(NON_ACCEPTED_EVIDENCE)
REVIEW_DECISIONS = ["accepted", "rejected", "pending"]
# documentation that must identify the specific product to be acceptable
NEEDS_PRODUCT_ID = {"manufacturer_documentation", "supplier_documentation"}

# Bands of the Phase 4 collection task (inclusive). Very high includes 100 % cotton (Phase 4 decision, 2026-11-25;
# earlier experiments used 98-99 % and keep their own definitions). Values in the gaps are kept but are out of band.
BANDS = {
    "low": (30.0, 40.0),
    "medium": (50.0, 66.0),
    "high": (80.0, 95.0),
    "very_high": (98.0, 100.0),
}
BAND_LABELS = {"low": "Low (30–40%)", "medium": "Medium (50–66%)", "high": "High (80–95%)",
               "very_high": "Very high (98–100%)"}
TARGET_FABRICS_PER_BAND = 30
MIN_IMAGES_PER_FABRIC = 4
MAX_IMAGES_PER_FABRIC = 6
LAB_AGREEMENT_TOLERANCE = 3.0   # percentage points: max disagreement between accepted evidence / registry value
SUM_TOLERANCE = 0.5

# Computed fabric statuses
VERIFIED = "VERIFIED"
PENDING = "PENDING_VERIFICATION"
CONFLICT = "CONFLICTING_EVIDENCE"
REJECTED = "REJECTED"
INCOMPLETE = "INCOMPLETE_RECORD"


def band_of(cotton: float | None) -> str | None:
    if cotton is None or pd.isna(cotton):
        return None
    for b, (lo, hi) in BANDS.items():
        if lo <= float(cotton) <= hi:
            return b
    return "out_of_band"


def to_float(x) -> float | None:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    s = str(x).strip().rstrip("%")
    if s == "":
        return None
    try:
        return float(s)
    except ValueError:
        return float("nan")


def parse_fibres(spec) -> dict[str, float] | None:
    """'polyester:35|elastane:5' -> {'polyester': 35.0, 'elastane': 5.0}; None if empty; raises ValueError if malformed."""
    if spec is None or (isinstance(spec, float) and pd.isna(spec)) or str(spec).strip() == "":
        return None
    out = {}
    for part in str(spec).split("|"):
        name, _, val = part.partition(":")
        if not name.strip() or val.strip() == "":
            raise ValueError(f"malformed fibre entry {part!r}")
        out[name.strip().lower()] = float(val)
    return out


# --------------------------------------------------------------------------- #
# Storage
# --------------------------------------------------------------------------- #
def init_dataset(root: Path | None = None) -> dict[str, Path]:
    p = paths(root)
    for k in ("images", "documents", "manifests", "audit"):
        p[k].mkdir(parents=True, exist_ok=True)
    for key, cols in (("fabrics", FABRIC_COLUMNS), ("images_csv", IMAGE_COLUMNS), ("verification", VERIFICATION_COLUMNS)):
        if not p[key].is_file():
            write_csv(p[key], pd.DataFrame(columns=cols))
    return p


def read_csv(path: Path, columns: list[str]) -> pd.DataFrame:
    if not Path(path).is_file():
        return pd.DataFrame(columns=columns)
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    for c in columns:
        if c not in df.columns:
            df[c] = ""
    return df


def write_csv(path: Path, df: pd.DataFrame) -> None:
    """Atomic write (temp file + replace) so a crash never leaves a half-written registry."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(fd)
    df.to_csv(tmp, index=False, quoting=csv.QUOTE_MINIMAL)
    os.replace(tmp, path)


def load(root: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    p = paths(root)
    return (read_csv(p["fabrics"], FABRIC_COLUMNS), read_csv(p["images_csv"], IMAGE_COLUMNS),
            read_csv(p["verification"], VERIFICATION_COLUMNS))


def next_id(existing, prefix: str, width: int) -> str:
    nums = [int(m.group(1)) for x in existing for m in [re.match(rf"^{prefix}-(\d+)$", str(x))] if m]
    return f"{prefix}-{(max(nums) + 1 if nums else 1):0{width}d}"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def add_fabric(fields: dict, root: Path | None = None) -> str:
    p = init_dataset(root)
    fab, _, _ = load(root)
    fid = fields.get("fabric_id") or next_id(fab.fabric_id, "BWF", 4)
    if fid in set(fab.fabric_id):
        raise ValueError(f"{fid} already exists")
    row = {c: "" for c in FABRIC_COLUMNS}
    row.update({k: ("" if v is None else str(v)) for k, v in fields.items() if k in FABRIC_COLUMNS})
    row["fabric_id"] = fid
    row["collection_date"] = row["collection_date"] or date.today().isoformat()
    write_csv(p["fabrics"], pd.concat([fab, pd.DataFrame([row])], ignore_index=True))
    return fid


def add_verification(fields: dict, document: tuple[str, bytes] | None = None, root: Path | None = None) -> str:
    p = init_dataset(root)
    fab, _, ver = load(root)
    if fields.get("fabric_id") not in set(fab.fabric_id):
        raise ValueError(f"unknown fabric_id {fields.get('fabric_id')!r}")
    rid = next_id(ver.record_id, "BWV", 5)
    row = {c: "" for c in VERIFICATION_COLUMNS}
    row.update({k: ("" if v is None else str(v)) for k, v in fields.items() if k in VERIFICATION_COLUMNS})
    row["record_id"] = rid
    row["review_decision"] = row["review_decision"] or "pending"
    if document is not None:
        name, data = document
        d = p["documents"] / row["fabric_id"]
        d.mkdir(parents=True, exist_ok=True)
        target = d / f"{rid}{Path(name).suffix.lower()}"
        target.write_bytes(data)
        row["document_path"] = target.relative_to(p["root"]).as_posix()
    write_csv(p["verification"], pd.concat([ver, pd.DataFrame([row])], ignore_index=True))
    return rid


def add_image(fabric_id: str, filename: str, data: bytes, fields: dict, root: Path | None = None) -> str:
    """Store one photograph under images/<fabric_id>/ and register it. Rejects byte-identical duplicates."""
    from io import BytesIO

    from PIL import Image

    p = init_dataset(root)
    fab, img, _ = load(root)
    if fabric_id not in set(fab.fabric_id):
        raise ValueError(f"unknown fabric_id {fabric_id!r}")
    digest = sha256_bytes(data)
    dup = img[img.sha256 == digest]
    if len(dup):
        raise ValueError(f"duplicate photograph: identical to {dup.image_id.iloc[0]} ({dup.fabric_id.iloc[0]})")
    try:
        with Image.open(BytesIO(data)) as im:
            im.verify()
        with Image.open(BytesIO(data)) as im:
            w, h = im.size
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"not a readable image: {exc}") from exc
    iid = next_id(img.image_id, "BWI", 6)
    ext = Path(filename).suffix.lower() or ".jpg"
    d = p["images"] / fabric_id
    d.mkdir(parents=True, exist_ok=True)
    target = d / f"{iid}{ext}"
    target.write_bytes(data)
    row = {c: "" for c in IMAGE_COLUMNS}
    row.update({k: ("" if v is None else str(v)) for k, v in fields.items() if k in IMAGE_COLUMNS})
    row.update({"image_id": iid, "fabric_id": fabric_id, "image_path": target.relative_to(p["root"]).as_posix(),
                "resolution": f"{w}x{h}", "sha256": digest,
                "capture_date": row["capture_date"] or date.today().isoformat()})
    write_csv(p["images_csv"], pd.concat([img, pd.DataFrame([row])], ignore_index=True))
    return iid


# --------------------------------------------------------------------------- #
# Verification status (the only route into the training manifest)
# --------------------------------------------------------------------------- #
@dataclass
class FabricStatus:
    fabric_id: str
    status: str
    band: str | None
    cotton: float | None
    accepted_records: list[str]
    reasons: list[str]


def record_is_acceptable(r: pd.Series, root: Path) -> tuple[bool, str]:
    """An evidence record can verify a label only if every rule below holds."""
    if r.evidence_type not in ACCEPTED_EVIDENCE:
        return False, f"{r.record_id}: evidence type '{r.evidence_type}' cannot verify a label"
    if r.review_decision != "accepted":
        return False, f"{r.record_id}: review decision is '{r.review_decision or 'pending'}'"
    if not str(r.evidence_reference).strip():
        return False, f"{r.record_id}: no evidence reference (report/document number)"
    if not str(r.document_path).strip() or not (root / r.document_path).is_file():
        return False, f"{r.record_id}: evidence document missing"
    if r.evidence_type in NEEDS_PRODUCT_ID and not str(r.product_identifier).strip():
        return False, f"{r.record_id}: documentation does not identify the product"
    if r.evidence_type == "laboratory_test" and not str(r.test_method_standard).strip():
        return False, f"{r.record_id}: laboratory test method/standard not recorded"
    rc = to_float(r.reported_cotton_percentage)
    if rc is None or pd.isna(rc) or not 0 <= rc <= 100:
        return False, f"{r.record_id}: reported cotton percentage missing or invalid"
    if not str(r.reviewed_by).strip():
        return False, f"{r.record_id}: reviewer not recorded"
    return True, ""


def fabric_statuses(fab: pd.DataFrame, ver: pd.DataFrame, root: Path | None = None) -> pd.DataFrame:
    root = paths(root)["root"]
    out = []
    for f in fab.itertuples():
        reasons = []
        cotton = to_float(f.cotton_percentage)
        recs = ver[ver.fabric_id == f.fabric_id]
        acc = []
        for r in recs.itertuples():
            ok, why = record_is_acceptable(pd.Series(r._asdict()), root)
            if ok:
                acc.append(r)
            elif why:
                reasons.append(why)
        if cotton is None or pd.isna(cotton) or not 0 <= cotton <= 100:
            status = INCOMPLETE
            reasons.insert(0, "cotton percentage missing or invalid")
        elif (recs.review_decision == "rejected").any() and not acc:
            status = REJECTED
        elif not acc:
            status = PENDING
            reasons.insert(0, "no accepted verification evidence")
        else:
            vals = [to_float(r.reported_cotton_percentage) for r in acc]
            if max(vals) - min(vals) > LAB_AGREEMENT_TOLERANCE:
                status = CONFLICT
                reasons.insert(0, f"accepted evidence disagrees: {sorted(vals)}")
            elif any(abs(v - cotton) > LAB_AGREEMENT_TOLERANCE for v in vals):
                status = CONFLICT
                reasons.insert(0, f"registry cotton {cotton} differs from verified evidence {sorted(vals)}")
            else:
                status = VERIFIED
                reasons = []
        out.append({"fabric_id": f.fabric_id, "status": status, "cotton_percentage": cotton,
                    "band": band_of(cotton) if status != INCOMPLETE else None,
                    "accepted_records": ";".join(r.record_id for r in acc),
                    "reasons": " | ".join(reasons)})
    return pd.DataFrame(out, columns=["fabric_id", "status", "cotton_percentage", "band", "accepted_records", "reasons"])
