"""Evaluate a trained checkpoint on the held-out test split.

Usage:
    python -m training.composition.evaluate [--checkpoint models/composition/best_model.pth]

Outputs (in evaluation/composition/<run_name>/):
    metrics_test.json, metrics_val.json, classification_report_test.txt,
    per_class_metrics_test.csv, predictions_test.csv,
    confusion_matrix_test.png, confusion_matrix_test_normalized.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import DEFAULT_CHECKPOINT, cotton_label  # noqa: E402
from training.composition.dataset import FabricDataset, resolve_path  # noqa: E402
from training.composition.model import load_model_from_checkpoint  # noqa: E402
from training.composition.transforms import build_eval_transform  # noqa: E402
from utils.device import amp_enabled, get_device  # noqa: E402
from utils.plotting import plot_confusion_matrix  # noqa: E402


def cotton_value(class_name: str) -> int:
    return int("".join(ch for ch in class_name if ch.isdigit()))


@torch.inference_mode()
def predict_frame(model, frame: pd.DataFrame, preprocessing: dict, device, batch_size: int = 32,
                  num_workers: int = 0) -> np.ndarray:
    ds = FabricDataset(frame, build_eval_transform(preprocessing))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    probs = []
    use_amp = amp_enabled(device)
    for x, _ in loader:
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(x.to(device))
        probs.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
    return np.concatenate(probs)


def compute_metrics(y_true: np.ndarray, probs: np.ndarray, classes: list[str]) -> dict:
    y_pred = probs.argmax(1)
    labels = list(range(len(classes)))
    top3 = np.argsort(-probs, axis=1)[:, :3]
    values = np.array([cotton_value(c) for c in classes])
    abs_err = np.abs(values[y_pred] - values[y_true])
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    return {
        "n_images": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "top3_accuracy": float(np.mean([t in row for t, row in zip(y_true, top3)])),
        "mean_abs_error_cotton_pct_points": float(abs_err.mean()),
        "within_5_pct_points": float(np.mean(abs_err <= 5)),
        "within_10_pct_points": float(np.mean(abs_err <= 10)),
        "per_class": {
            c: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
            for i, c in enumerate(classes)
        },
    }


def evaluate_checkpoint(checkpoint_path: str | Path = DEFAULT_CHECKPOINT, device: torch.device | None = None,
                        num_workers: int = 0, splits_path: str | None = None) -> dict:
    device = device or get_device()
    ckpt_path = resolve_path(checkpoint_path)
    if not ckpt_path.is_file() or ckpt_path.stat().st_size == 0:
        raise FileNotFoundError(f"No trained checkpoint at {ckpt_path}. Train first.")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    classes = [ckpt["idx_to_class"][i] for i in range(len(ckpt["idx_to_class"]))]
    model = load_model_from_checkpoint(ckpt, device)

    df = pd.read_csv(resolve_path(splits_path or cfg["splits_path"]))
    out_dir = resolve_path(cfg["output_dir"]) / cfg["run_name"]
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for split in ("val", "test"):
        frame = df[df.split == split].reset_index(drop=True)
        probs = predict_frame(model, frame, ckpt["preprocessing"], device, num_workers=num_workers)
        y_true = frame["label_idx"].to_numpy()
        m = compute_metrics(y_true, probs, classes)
        m.update({"split": split, "split_strategy": cfg["split_strategy"], "checkpoint": str(ckpt_path),
                  "best_epoch": ckpt["best_epoch"]})
        results[split] = m
        (out_dir / f"metrics_{split}.json").write_text(json.dumps(m, indent=2))
        if split != "test":
            continue

        y_pred = probs.argmax(1)
        pretty = [cotton_label(c) for c in classes]
        report = classification_report(y_true, y_pred, labels=list(range(len(classes))),
                                       target_names=pretty, digits=3, zero_division=0)
        header = (f"Test-set classification report ({cfg['split_strategy']} split, "
                  f"{len(frame)} images, best epoch {ckpt['best_epoch']})\n\n")
        (out_dir / "classification_report_test.txt").write_text(header + report)
        pd.DataFrame(m["per_class"]).T.rename_axis("class").to_csv(out_dir / "per_class_metrics_test.csv")
        pred_df = frame[["path", "label", "group"]].copy()
        pred_df["predicted"] = [classes[i] for i in y_pred]
        pred_df["score"] = probs.max(1)
        pred_df["correct"] = pred_df["predicted"] == pred_df["label"]
        pred_df.to_csv(out_dir / "predictions_test.csv", index=False)
        cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
        np.savetxt(out_dir / "confusion_matrix_test.csv", cm, fmt="%d", delimiter=",")
        short = [c.replace("cotton", "%") for c in classes]
        plot_confusion_matrix(cm, short, out_dir / "confusion_matrix_test.png",
                              f"Test confusion matrix (counts, n={len(frame)})")
        plot_confusion_matrix(cm, short, out_dir / "confusion_matrix_test_normalized.png",
                              "Test confusion matrix (row-normalised recall)", normalize=True)
        print(header + report)

    for split in ("val", "test"):
        m = results[split]
        print(f"[{split.upper():4s}] acc {m['accuracy']:.4f} | macro-F1 {m['macro_f1']:.4f} | "
              f"top-3 {m['top3_accuracy']:.4f} | MAE {m['mean_abs_error_cotton_pct_points']:.1f} pts "
              f"| n={m['n_images']}")
    print(f"Evaluation outputs: {out_dir}")
    return results


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    ap.add_argument("--splits-path", default=None, help="override the split CSV stored in the checkpoint")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    evaluate_checkpoint(args.checkpoint, get_device(args.device), splits_path=args.splits_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
