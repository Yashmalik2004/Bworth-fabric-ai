"""Synthetic Phase 4 datasets for PIPELINE TESTS ONLY (E0). Never real ground truth.

Builds a dataset through the real Phase 4 registry API (policy v2, two reviewers), with procedurally generated
images whose texture frequency depends on the band but whose colour, supplier, device, lighting and session are
independent of the band. Every fabric is marked SYNTHETIC and the dataset schema file carries "synthetic": true.
Refuses to write into the real Phase 4 dataset folder.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from PIL import Image

from data_collection.phase4 import registry as R
from training.phase5.protocol import CLASSES, PROJECT_ROOT, bands

REAL_DATASET = (PROJECT_ROOT / "datasets" / "phase4_verified").resolve()
COLOURS = ["navy blue", "red", "forest green", "white", "black", "grey", "yellow", "pink", "brown", "orange"]
SOURCES = [f"Synthetic Supplier {c}" for c in "ABCDEFGH"]
DEVICES = ["SynthCam A", "SynthCam B", "SynthCam C"]
LIGHTS = ["natural_daylight", "controlled_led_5000k"]
TYPES = ["full_fabric", "macro_texture", "natural_light", "alternate_angle", "controlled"]
RGB = {"navy blue": (30, 40, 110), "red": (180, 30, 40), "forest green": (30, 100, 50), "white": (235, 235, 230),
       "black": (25, 25, 25), "grey": (128, 128, 128), "yellow": (220, 200, 40), "pink": (230, 150, 170),
       "brown": (110, 70, 40), "orange": (230, 120, 30)}


def texture(band_idx: int, colour: str, rng: np.random.Generator, size: int = 96) -> bytes:
    """Stripe texture: frequency encodes the band (learnable signal), colour is random (no colour confound)."""
    y, x = np.mgrid[0:size, 0:size].astype(np.float32)
    freq = (band_idx + 1) * 0.12 + rng.normal(0, 0.01)
    angle = rng.uniform(0, np.pi)
    wave = 0.5 + 0.5 * np.sin(freq * (x * np.cos(angle) + y * np.sin(angle)) + rng.uniform(0, 6.28))
    base = np.array(RGB[colour], dtype=np.float32) * (0.7 + 0.3 * rng.random())
    img = base[None, None, :] * (0.55 + 0.45 * wave[..., None]) + rng.normal(0, 8, (size, size, 3))
    buf = io.BytesIO()
    Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(buf, format="PNG")
    return buf.getvalue()


def make_synthetic_dataset(root: Path, fabrics_per_band: int = 12, photos_per_fabric: int = 4, seed: int = 0,
                           related_pairs: int = 2, image_size: int = 96, texture_fn=None, note: str = "SYNTHETIC pipeline-test dataset") -> dict:
    root = Path(root).resolve()
    if root == REAL_DATASET or REAL_DATASET in root.parents:
        raise ValueError("refusing to write synthetic data into the real Phase 4 dataset")
    if R.paths(root)["fabrics"].is_file():
        raise ValueError(f"{root} already contains a dataset")
    rng = np.random.default_rng(seed)
    p = R.init_dataset(root, policy_version=R.CURRENT_POLICY_VERSION)
    R.write_schema(p, R.CURRENT_POLICY_VERSION, note=note, extra={"synthetic": True})
    texture_fn = texture_fn or texture
    band_ranges = bands()
    created = []
    for bi, band in enumerate(CLASSES):
        lo, hi = band_ranges[band]
        pg_ids = [f"PG-{band[:2].upper()}{k:02d}" for k in range(related_pairs)]
        for k in range(fabrics_per_band):
            cotton = round(float(rng.uniform(lo, hi)), 1)
            # the first 2*related_pairs fabrics of each band come in related pairs (same provenance group)
            prov = pg_ids[k // 2] if k < 2 * related_pairs else ""
            fid = R.add_fabric({
                "cotton_percentage": cotton, "other_fibre_percentage": round(100 - cotton, 1),
                "other_fibre_composition": f"polyester:{round(100 - cotton, 1)}" if cotton < 100 else "",
                "fabric_type": str(rng.choice(["poplin", "twill", "jersey", "rib"])),
                "weave_or_knit": str(rng.choice(["woven", "knit"])), "colour": str(rng.choice(COLOURS)),
                "source": str(rng.choice(SOURCES)), "physical_sample_location": f"SYNTH box {bi}/{k}",
                "provenance_group": prov, "source_product_id": f"SYN-ART-{bi}{k:03d}",
                "verification_method": "laboratory_test", "notes": "SYNTHETIC - pipeline test only"}, root=root)
            rid = R.add_verification({
                "fabric_id": fid, "evidence_type": "laboratory_test", "evidence_reference": f"SYN-LAB-{fid}",
                "issuing_party": "Synthetic Lab", "reported_cotton_percentage": cotton,
                "test_method_standard": "ISO 1833 (synthetic)", "reviewed_by": "Reviewer One",
                "review_decision": "accepted", "notes": "SYNTHETIC"},
                document=("report.pdf", b"%PDF-1.4 SYNTHETIC " + fid.encode()), root=root)
            R.add_second_review(rid, "Reviewer Two", "accepted", root=root)
            colour = R.load(root)[0].set_index("fabric_id").colour[fid]
            for j in range(photos_per_fabric):
                R.add_image(fid, f"s{j}.png", texture_fn(bi, colour, rng, image_size), {
                    "image_type": TYPES[j % len(TYPES)], "lighting": str(rng.choice(LIGHTS)),
                    "camera_angle": "overhead_90", "capture_device": str(rng.choice(DEVICES)),
                    "capture_session": f"SYN-S{int(rng.integers(0, 6))}", "notes": "SYNTHETIC"}, root=root)
            created.append(fid)
    return {"root": str(root), "fabrics": len(created), "synthetic": True}
