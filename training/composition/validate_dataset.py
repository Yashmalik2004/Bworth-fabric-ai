"""Audit the CottonFabricImageBD dataset before training.

Checks performed:
  * class folders in Original/ and Augmented/ and whether they match
  * image counts per class and file extensions
  * unreadable / corrupted images (Original/ is fully decoded, Augmented/ is
    header-verified unless --full is given)
  * image sizes and colour modes
  * exact duplicates (MD5) within and across classes and folders
  * near-duplicates among Original/ images (perceptual difference hash)
  * whether Augmented/ filenames can be mapped back to their source originals

Usage:
    python -m training.composition.validate_dataset [--full]

Writes evaluation/composition/dataset_report.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import (  # noqa: E402
    AUGMENTED_DIR,
    EVALUATION_DIR,
    IMAGE_EXTENSIONS,
    ORIGINAL_DIR,
    class_sort_key,
)


def list_classes(root: Path) -> list[str]:
    if not root.is_dir():
        return []
    return sorted((p.name for p in root.iterdir() if p.is_dir()), key=class_sort_key)


def dhash(img: Image.Image, size: int = 16) -> np.ndarray:
    g = np.asarray(img.convert("L").resize((size + 1, size), Image.BILINEAR), dtype=np.int16)
    return (g[:, 1:] > g[:, :-1]).flatten()


def inspect_file(path: Path, full_decode: bool, want_hash: bool) -> dict:
    rec = {"path": str(path), "ok": True, "error": None, "size": None, "mode": None}
    try:
        data = path.read_bytes()
        rec["md5"] = hashlib.md5(data).hexdigest()
        with Image.open(path) as im:
            rec["size"] = im.size
            rec["mode"] = im.mode
            if full_decode:
                im.load()
                if want_hash:
                    rec["dhash"] = dhash(im)
            else:
                im.verify()
    except Exception as exc:  # noqa: BLE001 - we want to report every failure
        rec["ok"] = False
        rec["error"] = f"{type(exc).__name__}: {exc}"
    return rec


def audit(full: bool = False, workers: int = 8, near_dup_threshold: int = 10) -> dict:
    report: dict = {"original_dir": str(ORIGINAL_DIR), "augmented_dir": str(AUGMENTED_DIR)}

    orig_classes = list_classes(ORIGINAL_DIR)
    aug_classes = list_classes(AUGMENTED_DIR)
    report["classes"] = orig_classes
    report["num_classes"] = len(orig_classes)
    report["augmented_classes_match_original"] = orig_classes == aug_classes

    records: dict[str, dict[str, list[dict]]] = {"Original": {}, "Augmented": {}}
    non_images: list[str] = []
    for split, root, classes, decode in (
        ("Original", ORIGINAL_DIR, orig_classes, True),
        ("Augmented", AUGMENTED_DIR, aug_classes, full),
    ):
        for cls in classes:
            files = sorted((root / cls).iterdir())
            imgs = [f for f in files if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS]
            non_images += [str(f) for f in files if f not in imgs]
            want_hash = split == "Original"
            with ThreadPoolExecutor(workers) as ex:
                records[split][cls] = list(ex.map(lambda p: inspect_file(p, decode, want_hash), imgs))
            print(f"  scanned {split}/{cls}: {len(imgs)} images", flush=True)

    # Counts
    report["counts"] = {
        cls: {
            "original": len(records["Original"].get(cls, [])),
            "augmented": len(records["Augmented"].get(cls, [])),
        }
        for cls in orig_classes
    }
    report["total_original"] = sum(v["original"] for v in report["counts"].values())
    report["total_augmented"] = sum(v["augmented"] for v in report["counts"].values())
    report["non_image_files"] = non_images

    # Corruption, sizes, modes
    all_recs = [(s, c, r) for s in records for c in records[s] for r in records[s][c]]
    report["unreadable"] = [{"path": r["path"], "error": r["error"]} for _, _, r in all_recs if not r["ok"]]
    report["image_sizes"] = {
        s: {f"{w}x{h}": n for (w, h), n in Counter(tuple(r["size"]) for c in records[s] for r in records[s][c] if r["size"]).items()}
        for s in records
    }
    report["image_modes"] = {
        s: dict(Counter(r["mode"] for c in records[s] for r in records[s][c] if r["mode"])) for s in records
    }
    report["augmented_decode_mode"] = "full decode" if full else "header verify only (use --full to decode)"

    # Exact duplicates
    by_md5: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for s, c, r in all_recs:
        if r.get("md5"):
            by_md5[r["md5"]].append((s, c, Path(r["path"]).name))
    dup_groups = [v for v in by_md5.values() if len(v) > 1]
    report["exact_duplicate_groups"] = {
        "total_groups": len(dup_groups),
        "within_original": sum(1 for g in dup_groups if all(x[0] == "Original" for x in g)),
        "within_augmented": sum(1 for g in dup_groups if all(x[0] == "Augmented" for x in g)),
        "original_vs_augmented": sum(1 for g in dup_groups if len({x[0] for x in g}) > 1),
        "cross_class": sum(1 for g in dup_groups if len({x[1] for x in g}) > 1),
        "examples": [[f"{s}/{c}/{n}" for s, c, n in g] for g in dup_groups[:25]],
    }

    # Near duplicates among originals (perceptual hash, Hamming distance)
    orig = [(c, Path(r["path"]).name, r["dhash"]) for c in records["Original"] for r in records["Original"][c] if "dhash" in r]
    near = []
    if orig:
        H = np.stack([h for _, _, h in orig]).astype(np.uint8)
        for i in range(len(orig)):
            d = (H[i + 1:] != H[i]).sum(axis=1)
            for j in np.nonzero(d <= near_dup_threshold)[0]:
                k = i + 1 + int(j)
                near.append((orig[i][0], orig[i][1], orig[k][0], orig[k][1], int(d[j])))
    report["near_duplicates_original"] = {
        "hamming_threshold_of_256": near_dup_threshold,
        "pairs": len(near),
        "cross_class_pairs": sum(1 for a in near if a[0] != a[2]),
        "examples": [f"{a}/{b} ~ {c}/{d} (d={e})" for a, b, c, d, e in near[:25]],
    }

    # Augmented -> original mapping
    pat_orig = re.compile(r"^original_(\d+)$", re.I)
    pat_aug = re.compile(r"^augmented_(\d+)$", re.I)
    odd_aug = [f"{c}/{Path(r['path']).name}" for c in records["Augmented"] for r in records["Augmented"][c]
               if not pat_aug.match(Path(r["path"]).stem)]
    odd_orig = [f"{c}/{Path(r['path']).name}" for c in records["Original"] for r in records["Original"][c]
                if not pat_orig.match(Path(r["path"]).stem)]
    report["filename_anomalies"] = {"original": odd_orig, "augmented": odd_aug}
    report["augmented_to_original_mapping"] = {
        "mappable": False,
        "reason": (
            "Augmented files are named augmented_<global sequential index> and contain no reference "
            "to the source original; read_me.pdf documents no mapping. Augmented images are rotated/"
            "cropped versions of originals, so they cannot be split safely. They are therefore NOT "
            "used for training; augmentation is applied on the fly to training originals only."
        ),
    }
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--full", action="store_true", help="fully decode Augmented/ images too (slower)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", type=Path, default=EVALUATION_DIR / "dataset_report.json")
    args = ap.parse_args()

    if not ORIGINAL_DIR.is_dir():
        print(f"ERROR: dataset not found at {ORIGINAL_DIR}")
        return 1
    print(f"Auditing {ORIGINAL_DIR.parent} ...")
    report = audit(full=args.full, workers=args.workers)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))

    print(f"\nClasses ({report['num_classes']}): {', '.join(report['classes'])}")
    print(f"{'class':<10}{'original':>10}{'augmented':>11}")
    for c, v in report["counts"].items():
        print(f"{c:<10}{v['original']:>10}{v['augmented']:>11}")
    print(f"{'TOTAL':<10}{report['total_original']:>10}{report['total_augmented']:>11}")
    print(f"Unreadable images: {len(report['unreadable'])}")
    print(f"Non-image files:   {len(report['non_image_files'])}")
    print(f"Image sizes:       {report['image_sizes']}")
    print(f"Exact duplicates:  {json.dumps({k: v for k, v in report['exact_duplicate_groups'].items() if k != 'examples'})}")
    print(f"Near-duplicate original pairs: {report['near_duplicates_original']['pairs']} "
          f"(cross-class: {report['near_duplicates_original']['cross_class_pairs']})")
    print(f"Filename anomalies: {report['filename_anomalies']}")
    print(f"Augmented->original mapping possible: {report['augmented_to_original_mapping']['mappable']}")
    print(f"\nReport written to {args.out}")
    return 0 if not report["unreadable"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
