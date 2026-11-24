"""Train the ResNet-50 cotton-percentage classifier.

Usage (from the project root):
    python -m training.composition.train                       # swatch-aware split (default)
    python -m training.composition.train --split-strategy random \
        --run-name resnet50_random_split \
        --checkpoint-path models/composition/resnet50_random_split.pth

Any field of ``configs.config.CompositionConfig`` can be overridden with
``--field-name value``. Only train and validation data are used here; the test
split is evaluated once, after training, from the best checkpoint
(``training/composition/evaluate.py``; run automatically unless --no-evaluate).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import fields
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import CompositionConfig  # noqa: E402
from training.composition.dataset import FabricDataset, discover_classes, prepare_splits, resolve_path  # noqa: E402
from training.composition.model import build_model, parameter_groups  # noqa: E402
from training.composition.transforms import build_eval_transform, build_train_transform  # noqa: E402
from utils.device import amp_enabled, describe_device, get_device  # noqa: E402
from utils.plotting import plot_training_curves  # noqa: E402


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_config(argv: list[str] | None = None) -> tuple[CompositionConfig, argparse.Namespace]:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    defaults = CompositionConfig()
    for f in fields(CompositionConfig):
        if f.name in ("mean", "std"):
            continue
        val = getattr(defaults, f.name)
        flag = "--" + f.name.replace("_", "-")
        if isinstance(val, bool):
            ap.add_argument(flag, type=lambda s: s.lower() in ("1", "true", "yes"), default=None,
                            metavar="BOOL")
        else:
            ap.add_argument(flag, type=type(val), default=None)
    ap.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    ap.add_argument("--rebuild-splits", action="store_true", help="recompute the split CSV")
    ap.add_argument("--no-evaluate", action="store_true", help="skip the final test-set evaluation")
    args = ap.parse_args(argv)
    overrides = {f.name: getattr(args, f.name) for f in fields(CompositionConfig)
                 if f.name not in ("mean", "std") and getattr(args, f.name) is not None}
    if "splits_path" not in overrides:
        overrides["splits_path"] = ""
    cfg = CompositionConfig(**overrides)
    for attr in ("checkpoint_path", "output_dir", "splits_path", "data_dir"):
        setattr(cfg, attr, str(resolve_path(getattr(cfg, attr))))
    return cfg, args


def run_epoch(model, loader, criterion, device, use_amp, optimizer=None, scaler=None) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(x)
                loss = criterion(logits, y)
            if training:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            total_loss += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            n += x.size(0)
    return total_loss / max(n, 1), correct / max(n, 1)


def save_checkpoint(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def train(cfg: CompositionConfig, device: torch.device, rebuild_splits: bool = False) -> dict:
    set_seed(cfg.seed)
    out_dir = Path(cfg.output_dir) / cfg.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    classes = discover_classes(cfg.data_dir)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    df = prepare_splits(cfg, rebuild=rebuild_splits)
    train_df, val_df = df[df.split == "train"], df[df.split == "val"]
    if train_df.empty or val_df.empty:
        raise RuntimeError("Train or validation split is empty; check the dataset and split settings.")

    pre = cfg.preprocessing()
    train_ds = FabricDataset(train_df, build_train_transform(pre))
    val_ds = FabricDataset(val_df, build_eval_transform(pre))
    loader_kw = dict(num_workers=cfg.num_workers, pin_memory=device.type == "cuda",
                     persistent_workers=cfg.num_workers > 0)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=len(train_ds) > cfg.batch_size, **loader_kw)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size * 2, shuffle=False, **loader_kw)

    weight = None
    if cfg.class_weighted_loss:
        counts = np.bincount(train_df["label_idx"], minlength=len(classes)).astype(float)
        w = np.where(counts > 0, counts.sum() / (len(classes) * np.maximum(counts, 1)), 0.0)
        weight = torch.tensor(w, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weight, label_smoothing=cfg.label_smoothing)
    eval_criterion = nn.CrossEntropyLoss(weight=weight)

    model = build_model(len(classes), pretrained=cfg.pretrained, dropout=cfg.dropout).to(device)
    optimizer = torch.optim.AdamW(parameter_groups(model, cfg.lr_backbone, cfg.lr_head), weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    use_amp = amp_enabled(device, cfg.amp)
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    print(f"Device: {describe_device(device)} | AMP: {use_amp}")
    print(f"Train: {len(train_ds)} images | Val: {len(val_ds)} images | Classes: {len(classes)}")

    history: list[dict] = []
    best = {"val_acc": -1.0, "val_loss": float("inf"), "epoch": 0}
    ckpt_path = Path(cfg.checkpoint_path)
    stale = 0
    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, device, use_amp, optimizer, scaler)
        va_loss, va_acc = run_epoch(model, val_loader, eval_criterion, device, use_amp)
        scheduler.step()
        history.append({"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc,
                        "val_loss": va_loss, "val_acc": va_acc, "seconds": time.time() - t0})
        improved = va_acc > best["val_acc"] or (va_acc == best["val_acc"] and va_loss < best["val_loss"])
        flag = ""
        if improved:
            best = {"val_acc": va_acc, "val_loss": va_loss, "epoch": epoch}
            stale = 0
            flag = "  * saved best"
            save_checkpoint(ckpt_path, {
                "model_state_dict": model.state_dict(),
                "architecture": cfg.architecture,
                "class_to_idx": class_to_idx,
                "idx_to_class": {i: c for c, i in class_to_idx.items()},
                "preprocessing": pre,
                "config": cfg.to_dict(),
                "best_epoch": epoch,
                "val_acc": va_acc,
                "val_loss": va_loss,
                "train_acc": tr_acc,
                "split_counts": df.groupby("split").size().to_dict(),
                "trained": True,
                "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "torch_version": torch.__version__,
            })
        else:
            stale += 1
        print(f"epoch {epoch:3d}/{cfg.epochs} | train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
              f"val loss {va_loss:.4f} acc {va_acc:.4f} | {history[-1]['seconds']:.0f}s{flag}", flush=True)
        if stale >= cfg.early_stopping_patience:
            print(f"Early stopping: no validation improvement for {stale} epochs.")
            break

    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
    plot_training_curves(history, best["epoch"], out_dir / "training_curves.png")
    summary = {
        "run_name": cfg.run_name,
        "best_epoch": best["epoch"],
        "best_val_acc": best["val_acc"],
        "best_val_loss": best["val_loss"],
        "train_acc_at_best_epoch": history[best["epoch"] - 1]["train_acc"],
        "epochs_run": len(history),
        "checkpoint": str(ckpt_path),
        "device": describe_device(device),
        "amp": use_amp,
        "config": cfg.to_dict(),
    }
    (out_dir / "training_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Best epoch {best['epoch']}: val acc {best['val_acc']:.4f}, val loss {best['val_loss']:.4f}")
    print(f"Checkpoint: {ckpt_path}\nTraining outputs: {out_dir}")
    return summary


def main(argv: list[str] | None = None) -> int:
    cfg, args = parse_config(argv)
    device = get_device(args.device)
    train(cfg, device, rebuild_splits=args.rebuild_splits)
    if not args.no_evaluate:
        from training.composition.evaluate import evaluate_checkpoint

        evaluate_checkpoint(cfg.checkpoint_path, device=device, num_workers=cfg.num_workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
