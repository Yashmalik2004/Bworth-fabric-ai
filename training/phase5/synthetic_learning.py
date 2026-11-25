"""SYNTHETIC — PIPELINE TEST: learnability check of the Phase 5 training pipeline.

Separate from E0. Builds a synthetic dataset with a deliberately learnable texture signal (stripe frequency doubles
from class to class; colour, supplier, device, lighting and session are random), then runs the REAL pipeline:
snapshot -> lockbox -> fabric-disjoint grouped CV -> guarded training -> fabric-level evaluation.

It shows the code can learn a learnable signal (training loss falls, held-out fabric accuracy rises above chance).
It is NOT evidence of fabric-composition accuracy: no real images, real paths or real labels are used.

    python -m training.phase5.synthetic_learning --work <scratch dir> [--epochs 20] [--device cuda]
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from training.phase5 import snapshot, splits
from training.phase5.evaluate import aggregate_fabrics, bootstrap_ci, fabric_metrics
from training.phase5.protocol import CLASSES, PROJECT_ROOT
from training.phase5.synthetic import RGB, make_synthetic_dataset
from training.phase5.train import train_split

BANNER = "SYNTHETIC — PIPELINE TEST (learnability check) — NOT evidence of fabric-composition accuracy"
FREQUENCIES = [0.08, 0.16, 0.32, 0.64]   # rad/px, factor 2 apart: survives the <=1.6x random-crop zoom
# From-scratch recipe (no pretrained weights at this stage): the protocol's backbone LR (1e-4) is a fine-tuning
# rate for ImageNet weights and barely moves a randomly initialised network. Allowed only on synthetic snapshots.
FROM_SCRATCH = {"lr_backbone": 1e-3, "lr_head": 1e-3, "early_stopping_patience": 30}


def learnable_texture(band_idx: int, colour: str, rng: np.random.Generator, size: int = 96) -> bytes:
    y, x = np.mgrid[0:size, 0:size].astype(np.float32)
    angle = rng.uniform(0, np.pi)
    wave = np.sin(FREQUENCIES[band_idx] * (x * np.cos(angle) + y * np.sin(angle)) + rng.uniform(0, 6.28))
    base = np.array(RGB[colour], dtype=np.float32)
    img = base[None, None, :] * (0.6 + 0.4 * (wave[..., None] > 0)) + rng.normal(0, 6, (size, size, 3))
    buf = io.BytesIO()
    Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(buf, format="PNG")
    return buf.getvalue()


def _guard_work_dir(work: Path) -> None:
    work = work.resolve()
    real = [(PROJECT_ROOT / "datasets").resolve(), (PROJECT_ROOT / "models").resolve()]
    if any(r == work or r in work.parents for r in real):
        raise ValueError("refusing to run the synthetic learning test inside real dataset/model folders")


def run(work: Path, fabrics_per_band: int = 24, photos: int = 4, folds: int = 3, epochs: int = 30,
        arch: str = "resnet50", device: str | None = None, seed: int = 7) -> dict:
    work = Path(work)
    _guard_work_dir(work)
    work.mkdir(parents=True, exist_ok=False)
    (work / "README.md").write_text(f"# {BANNER}\n", encoding="utf-8")
    make_synthetic_dataset(work / "phase4_synthetic_learnable", fabrics_per_band=fabrics_per_band,
                           photos_per_fabric=photos, seed=seed, image_size=96, texture_fn=learnable_texture,
                           note="SYNTHETIC — PIPELINE TEST (learnability check)")
    snap = snapshot.create_snapshot(work / "phase4_synthetic_learnable", work / "snapshots", version="SYNTH-LEARN")
    splits.seal_lockbox(snap)
    files = splits.make_cv_splits(snap, work / "splits", repeats=1, folds=folds)
    runs, histories = {}, {}
    for f in files:
        out = work / "runs" / f.stem
        train_split(f, snap, out, arch=arch, seed=42, epochs=epochs, image_size=64, resize_size=72,
                    batch_size=16, device=device, recipe_overrides=FROM_SCRATCH)
        runs[f.stem] = pd.read_csv(out / "predictions_test.csv")
        histories[f.stem] = pd.read_csv(out / "history.csv")
    per_fold = {k: fabric_metrics(aggregate_fabrics(v)) for k, v in runs.items()}
    pooled = aggregate_fabrics(pd.concat(runs.values()))
    pm = fabric_metrics(pooled)
    lo, hi = bootstrap_ci(pooled, n=2000)
    loss = {k: {"first_epoch_train_loss": float(h.train_loss.iloc[0]), "min_train_loss": float(h.train_loss.min()),
                "last_epoch_train_loss": float(h.train_loss.iloc[-1]), "epochs_run": int(len(h)),
                "best_val_acc": float(h.val_acc.max()), "first_val_acc": float(h.val_acc.iloc[0])}
            for k, h in histories.items()}
    res = {
        "banner": BANNER, "arch": arch, "pretrained": False, "recipe_overrides": FROM_SCRATCH,
        "epochs_max": epochs, "folds": folds,
        "fabrics_per_band": fabrics_per_band, "photos_per_fabric": photos, "chance_balanced_accuracy": 1 / len(CLASSES),
        "training_loss": loss,
        "training_loss_decreased_every_fold": all(v["min_train_loss"] < v["first_epoch_train_loss"] for v in loss.values()),
        "held_out_fabric_balanced_accuracy_per_fold": {k: v["balanced_accuracy"] for k, v in per_fold.items()},
        "held_out_fabric_balanced_accuracy_pooled": pm["balanced_accuracy"], "pooled_ci95": [lo, hi],
        "above_chance": bool(lo > 1 / len(CLASSES)),
        "confusion_matrix_pooled": pm["confusion_matrix"],
    }
    (work / "SYNTHETIC_PIPELINE_TEST_learning_results.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    _plot(histories, work / "SYNTHETIC_PIPELINE_TEST_learning_curves.png")
    return res


def _plot(histories: dict, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), facecolor="#fcfcfb")
    colours = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
    for i, (name, h) in enumerate(histories.items()):
        axes[0].plot(h.epoch, h.train_loss, color=colours[i % 5], linewidth=2, label=name)
        axes[1].plot(h.epoch, h.val_acc, color=colours[i % 5], linewidth=2, label=name)
    axes[1].axhline(0.25, color="#898781", linestyle="--", linewidth=1)
    axes[1].text(1, 0.27, "chance", color="#52514e", fontsize=8)
    for ax, t in ((axes[0], "Training loss"), (axes[1], "Inner-validation accuracy (images)")):
        ax.set_title(t, loc="left", fontsize=11)
        ax.set_xlabel("Epoch")
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle(BANNER, fontsize=10, color="#d03b3b")
    fig.tight_layout()
    fig.savefig(out, dpi=130, facecolor="#fcfcfb")
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--fabrics-per-band", type=int, default=24)
    ap.add_argument("--device", default=None)
    ap.add_argument("--arch", default="resnet50")
    a = ap.parse_args(argv)
    res = run(a.work, fabrics_per_band=a.fabrics_per_band, epochs=a.epochs, device=a.device, arch=a.arch)
    print(json.dumps(res, indent=2))
    print(BANNER)
    return 0


if __name__ == "__main__":
    sys.exit(main())
