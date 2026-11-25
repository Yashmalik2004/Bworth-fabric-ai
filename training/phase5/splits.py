"""Lockbox sealing and fabric-disjoint development splits (provenance groups never cross a split).

seal_lockbox: ~20 % of provenance groups per band, seeded, written once and sealed by SHA-256 (immutable).
make_cv_splits: repeated StratifiedGroupKFold on the non-lockbox groups; each training fold holds out an inner
validation split (by group) for early stopping; the test fold is never used for model selection.
check_split: hard integrity checks run before any split file is written.
"""

from __future__ import annotations

import json
import os
import stat
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from training.phase5.protocol import CLASSES, canonical_sha256, load_protocol, protocol_sha256
from training.phase5.provenance import build_groups
from training.phase5.snapshot import load_meta, load_tables, verify_snapshot


class SplitError(RuntimeError):
    pass


def fabric_table(snap: Path) -> pd.DataFrame:
    """One row per manifest fabric with band and provenance group; refuses groups spanning bands."""
    t = load_tables(snap)
    man = t["manifest"]
    fab = man.groupby("fabric_id").agg(band=("band", "first"), cotton=("cotton_percentage", "first"),
                                       n_images=("image_id", "size")).reset_index()
    g = build_groups(t["fabrics"], t["verification"], fab.fabric_id)
    fab = fab.merge(g, on="fabric_id")
    multi = fab.groupby("provenance_id").band.nunique()
    if (multi > 1).any():
        raise SplitError(f"provenance groups spanning several bands: {sorted(multi[multi > 1].index)}")
    return fab


def seal_lockbox(snap: Path, fraction: float | None = None, seed: int | None = None) -> dict:
    snap = Path(snap)
    if (snap / "lockbox.json").exists():
        raise SplitError("lockbox already sealed for this snapshot (immutable); create a new snapshot to change it")
    v = verify_snapshot(snap)
    if not v["ok"]:
        raise SplitError(f"snapshot does not verify: {v['problems'][:3]}")
    sp = load_protocol()["splits"]
    fraction = sp["lockbox_fraction"] if fraction is None else fraction
    seed = sp["lockbox_seed"] if seed is None else seed
    fab = fabric_table(snap)
    rng = np.random.default_rng(seed)
    chosen = []
    for band in CLASSES:
        groups = fab[fab.band == band].groupby("provenance_id").fabric_id.size()
        if len(groups) < 2:
            continue
        order = list(groups.index)
        rng.shuffle(order)
        target = max(1, int(round(fraction * groups.sum())))
        got = 0
        for gid in order:                       # whole groups only; never exceed the target once one is taken
            if got >= target:
                break
            if got > 0 and got + groups[gid] > target:
                continue
            chosen.append(gid)
            got += groups[gid]
    lb = fab[fab.provenance_id.isin(chosen)]
    content = {"snapshot_version": load_meta(snap)["snapshot_version"], "protocol_sha256": protocol_sha256(),
               "seed": seed, "fraction": fraction, "created": datetime.now().isoformat(timespec="seconds"),
               "provenance_groups": sorted(chosen), "fabrics": sorted(lb.fabric_id),
               "fabrics_per_band": lb.band.value_counts().reindex(CLASSES, fill_value=0).astype(int).to_dict()}
    seal = canonical_sha256(content)
    (snap / "lockbox.json").write_text(json.dumps(content, indent=2), encoding="utf-8")
    (snap / "lockbox.seal").write_text(seal + "\n", encoding="utf-8")
    for f in ("lockbox.json", "lockbox.seal"):
        os.chmod(snap / f, stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)
    return {"seal": seal, **content}


def verify_lockbox(snap: Path) -> dict:
    snap = Path(snap)
    if not (snap / "lockbox.json").is_file() or not (snap / "lockbox.seal").is_file():
        return {"ok": False, "problem": "lockbox not sealed"}
    content = json.loads((snap / "lockbox.json").read_text(encoding="utf-8"))
    seal = (snap / "lockbox.seal").read_text(encoding="utf-8").strip()
    ok = canonical_sha256(content) == seal and content["snapshot_version"] == load_meta(snap)["snapshot_version"]
    return {"ok": ok, "seal": seal, "problem": "" if ok else "lockbox content does not match its seal",
            "fabrics": content["fabrics"], "provenance_groups": content["provenance_groups"]}


def check_split(df: pd.DataFrame, lockbox_fabrics: set) -> dict:
    """Hard integrity checks on one split assignment (image rows with fabric_id, provenance_id, split)."""
    problems = []
    for col in ("fabric_id", "provenance_id"):
        n = df.groupby(col).split.nunique()
        if (n > 1).any():
            problems.append(f"{int((n > 1).sum())} {col} values in more than one split")
    leaked = set(df.fabric_id) & set(lockbox_fabrics)
    if leaked:
        problems.append(f"lockbox fabrics present in development split: {sorted(leaked)[:5]}")
    if df.image_id.duplicated().any():
        problems.append("duplicate image rows")
    if "sha256" in df and df.sha256.duplicated().any():
        problems.append("identical image file appears twice")
    presence = {s: sorted(set(df[df.split == s].band)) for s in ("train", "val", "test")}
    return {"ok": not problems, "problems": problems, "bands_present": presence}


def make_cv_splits(snap: Path, out_dir: Path, repeats: int | None = None, folds: int | None = None,
                   seed: int | None = None, inner_fraction: float | None = None) -> list[Path]:
    snap, out_dir = Path(snap), Path(out_dir)
    sp = load_protocol()["splits"]
    repeats = sp["cv_repeats"] if repeats is None else repeats
    folds = sp["cv_folds"] if folds is None else folds
    seed = sp["cv_seed"] if seed is None else seed
    inner_fraction = sp["inner_validation_fraction"] if inner_fraction is None else inner_fraction
    lb = verify_lockbox(snap)
    if not lb["ok"]:
        raise SplitError(f"cannot build splits: {lb['problem']}")
    fab = fabric_table(snap)
    dev = fab[~fab.fabric_id.isin(lb["fabrics"])].reset_index(drop=True)
    min_groups = dev.groupby("band").provenance_id.nunique()
    if (min_groups < folds).any():
        raise SplitError(f"too few provenance groups per band for {folds} folds: {min_groups.to_dict()}")
    man = load_tables(snap)["manifest"].merge(fab[["fabric_id", "provenance_id"]], on="fabric_id")
    out_dir.mkdir(parents=True, exist_ok=True)
    written, summary = [], []
    n_inner = max(2, int(round(1 / inner_fraction)))
    for r in range(repeats):
        outer = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed + r)
        for k, (tr_idx, te_idx) in enumerate(outer.split(dev, y=dev.band, groups=dev.provenance_id)):
            assign = pd.Series("train", index=dev.index)
            assign.iloc[te_idx] = "test"
            tr = dev.iloc[tr_idx]
            inner = StratifiedGroupKFold(n_splits=n_inner, shuffle=True, random_state=seed + 1000 * (r + 1) + k)
            _, va_rel = next(inner.split(tr, y=tr.band, groups=tr.provenance_id))
            assign.iloc[tr_idx[va_rel]] = "val"
            fsplit = dict(zip(dev.fabric_id, assign))
            df = man[man.fabric_id.isin(dev.fabric_id)].copy()
            df["split"] = df.fabric_id.map(fsplit)
            chk = check_split(df, set(lb["fabrics"]))
            if not chk["ok"]:
                raise SplitError(f"repeat {r} fold {k}: {chk['problems']}")
            f = out_dir / f"rep{r}_fold{k}.csv"
            df.to_csv(f, index=False)
            written.append(f)
            for s in ("train", "val", "test"):
                sub = df[df.split == s]
                summary.append({"repeat": r, "fold": k, "split": s, "images": len(sub), "fabrics": sub.fabric_id.nunique(),
                                "groups": sub.provenance_id.nunique(),
                                **{f"fabrics_{b}": int(sub[sub.band == b].fabric_id.nunique()) for b in CLASSES}})
    pd.DataFrame(summary).to_csv(out_dir / "split_summary.csv", index=False)
    (out_dir / "splits.json").write_text(json.dumps({
        "snapshot_version": load_meta(snap)["snapshot_version"], "lockbox_seal": lb["seal"],
        "protocol_sha256": protocol_sha256(), "repeats": repeats, "folds": folds, "seed": seed,
        "inner_validation_fraction": inner_fraction, "files": [f.name for f in written],
        "integrity": "every provenance group and fabric in exactly one split; no lockbox fabric present"}, indent=2),
        encoding="utf-8")
    return written
