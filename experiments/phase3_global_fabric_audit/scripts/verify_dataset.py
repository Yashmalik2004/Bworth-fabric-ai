"""Verify a local copy of CottonFabricImageBD against the Stage A inventory (SHA-256 of every file).

    python experiments/phase3_global_fabric_audit/scripts/verify_dataset.py [--original-only]

Exit code 0 = every inventoried file is present with the recorded SHA-256 and no extra files exist.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
INVENTORY = PROJECT_ROOT / "experiments/phase3_global_fabric_audit/audit/complete_image_inventory.csv"
DATASET = PROJECT_ROOT / "datasets/compositions/CottonFabricImageBD"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--original-only", action="store_true", help="check only Original/ (1,300 files)")
    args = ap.parse_args()
    inv = pd.read_csv(INVENTORY, usecols=["original_path", "sha256", "subset"])
    if args.original_only:
        inv = inv[inv.subset == "Original"]
    missing, mismatched = [], []
    for path, digest in zip(inv.original_path, inv.sha256):
        p = PROJECT_ROOT / path
        if not p.is_file():
            missing.append(path)
        elif sha256(p) != digest:
            mismatched.append(path)
    subsets = ["Original"] if args.original_only else ["Original", "Augmented"]
    on_disk = {p.relative_to(PROJECT_ROOT).as_posix() for s in subsets
               for p in (DATASET / s).rglob("*") if p.is_file()} if DATASET.is_dir() else set()
    extra = sorted(on_disk - set(inv.original_path))
    print(f"checked {len(inv)} files | missing {len(missing)} | SHA-256 mismatch {len(mismatched)} | extra {len(extra)}")
    for label, items in (("missing", missing), ("mismatch", mismatched), ("extra", extra)):
        for x in items[:10]:
            print(f"  {label}: {x}")
    return 0 if not (missing or mismatched or extra) else 1


if __name__ == "__main__":
    raise SystemExit(main())
