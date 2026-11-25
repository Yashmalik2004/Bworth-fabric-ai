"""Train one Phase 5 model on one development split (fabric-disjoint), with hard guards.

Guards (training_guard):
  * synthetic snapshots: allowed; every output is stamped PIPELINE TEST.
  * real snapshots: require snapshot + lockbox verification, readiness READY, unchanged protocol AND
    explicit approval (approved=True). Otherwise TrainingRefused.
Images are loaded only from the snapshot manifest, and every file hash is re-checked before training.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from training.composition.dataset import FabricDataset
from training.composition.train import run_epoch, save_checkpoint, set_seed
from training.composition.transforms import build_eval_transform, build_train_transform
from training.phase5.models import build_model, parameter_groups
from training.phase5.protocol import CLASSES, LABEL_SCHEME, SYNTHETIC_BANNER, load_protocol, protocol_sha256, sha256_file
from training.phase5.snapshot import image_root, load_meta, verify_snapshot
from training.phase5.splits import verify_lockbox


class TrainingRefused(RuntimeError):
    pass


def training_guard(snap: Path, approved: bool = False) -> dict:
    meta = load_meta(snap)
    if meta["synthetic"]:
        return {"mode": "synthetic_pipeline_test", "banner": SYNTHETIC_BANNER}
    problems = []
    v = verify_snapshot(snap)
    if not v["ok"]:
        problems.append(f"snapshot does not verify: {v['problems'][:3]}")
    lb = verify_lockbox(snap)
    if not lb["ok"]:
        problems.append(f"lockbox: {lb['problem']}")
    rf = Path(snap) / "readiness" / "readiness.json"
    overall = json.loads(rf.read_text())["overall"] if rf.is_file() else "NOT_EVALUATED"
    if overall != "READY":
        problems.append(f"readiness is {overall}, not READY")
    if meta["protocol_sha256"] != protocol_sha256():
        problems.append("protocol changed since the snapshot")
    if not approved:
        problems.append("explicit owner approval flag not given")
    if problems:
        raise TrainingRefused("real-data training refused: " + "; ".join(problems))
    return {"mode": "real", "banner": ""}


def load_split(split_csv: Path, snap: Path, verify_hashes: bool = True) -> pd.DataFrame:
    df = pd.read_csv(split_csv, dtype=str, keep_default_na=False)
    root = image_root(snap)
    df["path"] = [str(root / p) for p in df.image_path]
    df["target"] = df.band.map({c: i for i, c in enumerate(CLASSES)}).astype(int)
    if verify_hashes:
        bad = [r.image_id for r in df.itertuples() if sha256_file(Path(r.path)) != r.sha256]
        if bad:
            raise TrainingRefused(f"image hashes changed since snapshot: {bad[:5]}")
    return df


def train_split(split_csv: Path, snap: Path, out_dir: Path, arch: str = "resnet50", seed: int = 42,
                approved: bool = False, pretrained: bool = False, epochs: int | None = None,
                image_size: int | None = None, resize_size: int | None = None, batch_size: int | None = None,
                num_workers: int = 0, device: str | None = None, recipe_overrides: dict | None = None) -> dict:
    guard = training_guard(snap, approved)
    rec = dict(load_protocol()["training_reference_recipe"])
    if recipe_overrides:
        # only synthetic pipeline tests may deviate from the frozen recipe; real data always uses the protocol
        if guard["mode"] != "synthetic_pipeline_test":
            raise TrainingRefused("recipe_overrides are allowed only for synthetic pipeline tests")
        unknown = set(recipe_overrides) - set(rec)
        if unknown:
            raise ValueError(f"unknown recipe keys: {sorted(unknown)}")
        rec.update(recipe_overrides)
    epochs = epochs or rec["epochs"]
    batch_size = batch_size or rec["batch_size"]
    pre = {"resize_size": resize_size or rec["resize_size"], "image_size": image_size or rec["image_size"],
           "mean": rec["mean"], "std": rec["std"]}
    set_seed(seed)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    df = load_split(split_csv, snap)
    tr, va, te = (df[df.split == s].reset_index(drop=True) for s in ("train", "val", "test"))
    kw = dict(num_workers=num_workers)
    tl = DataLoader(FabricDataset(tr, build_train_transform(pre, rec["color_aug"])), batch_size=batch_size, shuffle=True,
                    drop_last=len(tr) > batch_size, **kw)
    vl = DataLoader(FabricDataset(va, build_eval_transform(pre)), batch_size=batch_size * 2, **kw)
    counts = np.bincount(tr.target, minlength=len(CLASSES)).astype(float)
    w = torch.tensor(counts.sum() / (len(CLASSES) * np.maximum(counts, 1)), dtype=torch.float32, device=dev)
    crit = nn.CrossEntropyLoss(weight=w, label_smoothing=rec["label_smoothing"])
    eval_crit = nn.CrossEntropyLoss(weight=w)
    model = build_model(arch, len(CLASSES), pretrained=pretrained).to(dev)
    opt = torch.optim.AdamW(parameter_groups(model, arch, rec["lr_backbone"], rec["lr_head"]), weight_decay=rec["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    use_amp = dev.type == "cuda"
    scaler = torch.amp.GradScaler(dev.type, enabled=use_amp)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ck_path = out_dir / "model.pth"
    meta = load_meta(snap)
    hist, best, stale = [], {"val_acc": -1.0, "val_loss": float("inf"), "epoch": 0}, 0
    for ep in range(1, epochs + 1):
        t0 = time.time()
        trl, tra = run_epoch(model, tl, crit, dev, use_amp, opt, scaler)
        val, vaa = run_epoch(model, vl, eval_crit, dev, use_amp)
        sched.step()
        hist.append({"epoch": ep, "train_loss": trl, "train_acc": tra, "val_loss": val, "val_acc": vaa, "seconds": time.time() - t0})
        if vaa > best["val_acc"] or (vaa == best["val_acc"] and val < best["val_loss"]):
            best, stale = {"val_acc": vaa, "val_loss": val, "epoch": ep}, 0
            save_checkpoint(ck_path, {
                "model_state_dict": model.state_dict(), "architecture": arch, "label_scheme": LABEL_SCHEME,
                "class_to_idx": {c: i for i, c in enumerate(CLASSES)}, "idx_to_class": dict(enumerate(CLASSES)),
                "preprocessing": pre, "config": {"arch": arch, "seed": seed, "pretrained": pretrained, "epochs": epochs,
                                                 "batch_size": batch_size, "color_aug": rec["color_aug"], "dropout": 0.3,
                                                 "split_file": Path(split_csv).name,
                                                 "recipe_overrides": recipe_overrides or {}},
                "best_epoch": ep, "val_acc": vaa, "val_loss": val, "trained": True,
                "synthetic": meta["synthetic"], "pipeline_test": meta["synthetic"],
                "snapshot_version": meta["snapshot_version"], "protocol_sha256": meta["protocol_sha256"],
                "lockbox_seal": verify_lockbox(snap).get("seal", "")})
        else:
            stale += 1
        if stale >= rec["early_stopping_patience"]:
            break
    pd.DataFrame(hist).to_csv(out_dir / "history.csv", index=False)
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    from training.composition.model import load_model_from_checkpoint

    model = load_model_from_checkpoint(ck, dev)
    tf = build_eval_transform(pre)
    probs = []
    with torch.inference_mode():
        for x, _ in DataLoader(FabricDataset(te, tf), batch_size=batch_size * 2):
            probs.append(torch.softmax(model(x.to(dev)).float(), 1).cpu().numpy())
    probs = np.concatenate(probs)
    pred = te[["image_id", "fabric_id", "provenance_id", "band"]].copy()
    for j, c in enumerate(CLASSES):
        pred[f"p_{c}"] = probs[:, j]
    pred.to_csv(out_dir / "predictions_test.csv", index=False)
    info = {"mode": guard["mode"], "banner": guard["banner"], "arch": arch, "seed": seed, "best_epoch": best["epoch"],
            "epochs_run": len(hist), "n_train": len(tr), "n_val": len(va), "n_test": len(te), "checkpoint": str(ck_path)}
    (out_dir / "run_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info
