"""Frozen, hash-verified dataset snapshots: the only input Phase 5 experiments may read.

create_snapshot refuses unless the Phase 4 validator reports zero errors. It never modifies the Phase 4 dataset.
Contents of <out>/<version>/:
  source/{fabric_registry,image_registry,verification_records}.csv, source/schema_version.json
  manifest.csv (verified training manifest), fabric_status.csv, issues.csv
  snapshot.json (metadata, protocol hash, counts), SHA256SUMS.txt (every frozen file + every image)
Frozen files are made read-only. verify_snapshot recomputes every hash, including every image.
"""

from __future__ import annotations

import json
import os
import shutil
import stat
from datetime import datetime
from pathlib import Path

import pandas as pd

from data_collection.phase4 import registry as R
from data_collection.phase4.validate import validate
from training.phase5.protocol import PROJECT_ROOT, PROTOCOL_FILE, protocol_sha256, protocol_version, sha256_file

FROZEN = ["source/fabric_registry.csv", "source/image_registry.csv", "source/verification_records.csv",
          "source/schema_version.json", "manifest.csv", "fabric_status.csv", "issues.csv", "snapshot.json"]


class SnapshotRefused(RuntimeError):
    pass


def _rel(p: Path) -> str:
    p = Path(p).resolve()
    try:
        return p.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return p.as_posix()


def _abs(p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else PROJECT_ROOT / q


def create_snapshot(phase4_root: Path, out_root: Path, version: str | None = None) -> Path:
    phase4_root = Path(phase4_root).resolve()
    res = validate(phase4_root)
    n_err = int((res["issues"].severity == "ERROR").sum())
    if n_err:
        raise SnapshotRefused(f"Phase 4 validator reports {n_err} ERROR(s); fix them before snapshotting")
    version = version or "P4S-" + datetime.now().strftime("%Y%m%dT%H%M%S")
    snap = Path(out_root) / version
    if snap.exists():
        raise SnapshotRefused(f"snapshot {snap} already exists (snapshots are immutable)")
    (snap / "source").mkdir(parents=True)
    p = R.paths(phase4_root)
    for key, name in (("fabrics", "fabric_registry.csv"), ("images_csv", "image_registry.csv"),
                      ("verification", "verification_records.csv")):
        shutil.copy2(p[key], snap / "source" / name)
    schema = R.read_schema(phase4_root)
    (snap / "source" / "schema_version.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")
    res["manifest"].to_csv(snap / "manifest.csv", index=False)
    res["status"].to_csv(snap / "fabric_status.csv", index=False)
    res["issues"].to_csv(snap / "issues.csv", index=False)

    # every manifest image must still match its registered hash
    img_hash = {}
    for r in res["manifest"].itertuples():
        f = phase4_root / r.image_path
        h = sha256_file(f)
        if h != r.sha256:
            shutil.rmtree(snap)
            raise SnapshotRefused(f"{r.image_id}: file hash differs from registry")
        img_hash[r.image_path] = h

    meta = {
        "snapshot_version": version, "created": datetime.now().isoformat(timespec="seconds"),
        "phase4_root": _rel(phase4_root), "schema": schema,
        "verification_policy_version": R.policy_version(phase4_root),
        "synthetic": bool(schema.get("synthetic", False)),
        "protocol_file": _rel(PROTOCOL_FILE), "protocol_sha256": protocol_sha256(),
        "protocol_version": protocol_version(),
        "counts": {"manifest_images": int(len(res["manifest"])), "manifest_fabrics": int(res["manifest"].fabric_id.nunique()),
                   "registered_fabrics": res["counts"]["fabrics"], "registered_images": res["counts"]["images"],
                   "validator_errors": n_err,
                   "issues_by_severity": res["issues"].severity.value_counts().to_dict()},
        "bands": res["bands"],
    }
    (snap / "snapshot.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    lines = [f"{sha256_file(snap / f)}  {f}" for f in FROZEN]
    lines += [f"{h}  images:{rel}" for rel, h in sorted(img_hash.items())]
    (snap / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for f in FROZEN + ["SHA256SUMS.txt"]:
        os.chmod(snap / f, stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)
    return snap


def load_meta(snap: Path) -> dict:
    return json.loads((Path(snap) / "snapshot.json").read_text(encoding="utf-8"))


def verify_snapshot(snap: Path) -> dict:
    snap = Path(snap)
    meta = load_meta(snap)
    root = _abs(meta["phase4_root"])
    bad, checked = [], 0
    for line in (snap / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        h, _, name = line.partition("  ")
        f = root / name[len("images:"):] if name.startswith("images:") else snap / name
        checked += 1
        if not f.is_file():
            bad.append(f"missing: {name}")
        elif sha256_file(f) != h:
            bad.append(f"hash mismatch: {name}")
    proto_ok = meta["protocol_sha256"] == protocol_sha256()
    if not proto_ok:
        bad.append(f"protocol changed since the snapshot was taken: snapshot used protocol "
                   f"v{meta.get('protocol_version', '1.0 (unrecorded)')} ({meta['protocol_sha256'][:12]}), current is "
                   f"v{protocol_version()} ({protocol_sha256()[:12]}); create a new snapshot")
    return {"ok": not bad, "checked": checked, "problems": bad, "snapshot_version": meta["snapshot_version"],
            "protocol_version": meta.get("protocol_version"), "current_protocol_version": protocol_version(),
            "synthetic": meta["synthetic"]}


def load_tables(snap: Path) -> dict[str, pd.DataFrame]:
    snap = Path(snap)
    rd = lambda f: pd.read_csv(snap / f, dtype=str, keep_default_na=False)  # noqa: E731
    return {"manifest": rd("manifest.csv"), "status": rd("fabric_status.csv"), "issues": rd("issues.csv"),
            "fabrics": rd("source/fabric_registry.csv"), "images": rd("source/image_registry.csv"),
            "verification": rd("source/verification_records.csv")}


def image_root(snap: Path) -> Path:
    return _abs(load_meta(snap)["phase4_root"])
