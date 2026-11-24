"""Evaluate a trained checkpoint on the held-out test split.

Usage:
    python -m training.composition.evaluate [--checkpoint models/composition/best_model.pth]
        [--splits-path processed/composition/splits_group_v2.csv] [--output-name NAME]

Outputs (in evaluation/composition/<output-name or run_name>/):
    metrics_test.json, metrics_val.json, classification_report_test.txt,
    per_class_metrics_test.csv, predictions_test.csv (with per-class probabilities),
    confusion_matrix_test.png, confusion_matrix_test_normalized.png
For a 13-class checkpoint the metrics also contain ``band4``: the same predictions
scored in the four cotton bands (13-class probabilities summed within each band),
so 13-class and 4-class models can be compared on the same task.
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
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import BAND_DEFINITIONS, DEFAULT_CHECKPOINT, FINE_TO_BAND, display_label  # noqa: E402
from training.composition.dataset import FabricDataset, apply_label_scheme, resolve_path  # noqa: E402
from training.composition.model import load_model_from_checkpoint  # noqa: E402
from training.composition.transforms import build_eval_transform  # noqa: E402
from utils.device import amp_enabled, get_device  # noqa: E402
from utils.plotting import plot_confusion_matrix  # noqa: E402

BANDS = list(BAND_DEFINITIONS)


def cotton_value(class_name: str) -> int | None:
    digits = "".join(ch for ch in class_name if ch.isdigit())
    return int(digits) if digits else None


def short_label(class_name: str) -> str:
    if class_name in BAND_DEFINITIONS:
        return {"low_cotton": "Low", "medium_cotton": "Medium", "high_cotton": "High",
                "very_high_cotton": "Very high"}[class_name]
    return class_name.replace("cotton", "%")


@torch.inference_mode()
def predict_frame(model, frame: pd.DataFrame, preprocessing: dict, device, batch_size: int = 32,
                  num_workers: int = 0, amp: bool = True) -> np.ndarray:
    ds = FabricDataset(frame, build_eval_transform(preprocessing))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    probs = []
    use_amp = amp_enabled(device, amp)
    for x, _ in loader:
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(x.to(device))
        probs.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
    return np.concatenate(probs)


def fine_to_band_probs(probs: np.ndarray, fine_classes: list[str]) -> np.ndarray:
    """Sum 13-class probabilities within each cotton band."""
    out = np.zeros((len(probs), len(BANDS)))
    for j, c in enumerate(fine_classes):
        out[:, BANDS.index(FINE_TO_BAND[c])] += probs[:, j]
    return out


def compute_metrics(y_true: np.ndarray, probs: np.ndarray, classes: list[str]) -> dict:
    y_pred = probs.argmax(1)
    labels = list(range(len(classes)))
    k = min(3, len(classes) - 1)
    topk = np.argsort(-probs, axis=1)[:, :k]
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    m = {
        "n_images": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)) if len(set(y_true)) > 1 else float("nan"),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        f"top{k}_accuracy": float(np.mean([t in row for t, row in zip(y_true, topk)])),
        "per_class": {
            c: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
            for i, c in enumerate(classes)
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }
    values = [cotton_value(c) for c in classes]
    if all(v is not None for v in values):
        v = np.array(values)
        err = np.abs(v[y_pred] - v[y_true])
        m["mean_abs_error_cotton_pct_points"] = float(err.mean())
    return m


def score_predictions(frame: pd.DataFrame, probs: np.ndarray, classes: list[str], scheme: str) -> dict:
    """Metrics in the model's own label space, plus 4-band metrics for 13-class models."""
    y = frame["target"].to_numpy()
    m = compute_metrics(y, probs, classes)
    if scheme == "fine13":
        y_band = frame["label"].map(FINE_TO_BAND).map({b: i for i, b in enumerate(BANDS)}).to_numpy()
        m["band4"] = compute_metrics(y_band, fine_to_band_probs(probs, classes), BANDS)
    return m


def evaluate_checkpoint(checkpoint_path: str | Path = DEFAULT_CHECKPOINT, device: torch.device | None = None,
                        num_workers: int = 0, splits_path: str | None = None,
                        output_name: str | None = None, verbose: bool = True) -> dict:
    device = device or get_device()
    ckpt_path = resolve_path(checkpoint_path)
    if not ckpt_path.is_file() or ckpt_path.stat().st_size == 0:
        raise FileNotFoundError(f"No trained checkpoint at {ckpt_path}. Train first.")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    scheme = ckpt.get("label_scheme", cfg.get("label_scheme", "fine13"))
    idx_to_class = {int(k): v for k, v in ckpt["idx_to_class"].items()}
    classes = [idx_to_class[i] for i in range(len(idx_to_class))]
    model = load_model_from_checkpoint(ckpt, device)

    split_file = resolve_path(splits_path or cfg["splits_path"])
    df = apply_label_scheme(pd.read_csv(split_file), scheme, classes)
    out_dir = resolve_path(cfg["output_dir"]) / (output_name or cfg["run_name"])
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for split in ("val", "test"):
        frame = df[df.split == split].reset_index(drop=True)
        probs = predict_frame(model, frame, ckpt["preprocessing"], device, num_workers=num_workers)
        m = score_predictions(frame, probs, classes, scheme)
        m.update({"split": split, "label_scheme": scheme, "split_file": str(split_file),
                  "checkpoint": str(ckpt_path), "best_epoch": ckpt["best_epoch"]})
        results[split] = m
        (out_dir / f"metrics_{split}.json").write_text(json.dumps(m, indent=2))
        if split != "test":
            continue

        y_true, y_pred = frame["target"].to_numpy(), probs.argmax(1)
        names = [display_label(c) for c in classes]
        report = classification_report(y_true, y_pred, labels=list(range(len(classes))),
                                       target_names=names, digits=3, zero_division=0)
        header = (f"Test-set classification report ({scheme}, split file {split_file.name}, "
                  f"{len(frame)} images, best epoch {ckpt['best_epoch']})\n\n")
        (out_dir / "classification_report_test.txt").write_text(header + report)
        pd.DataFrame(m["per_class"]).T.rename_axis("class").to_csv(out_dir / "per_class_metrics_test.csv")
        pred_df = frame[["path", "label", "group"]].copy()
        pred_df["true"] = [classes[i] for i in y_true]
        pred_df["predicted"] = [classes[i] for i in y_pred]
        pred_df["score"] = probs.max(1)
        pred_df["correct"] = pred_df["predicted"] == pred_df["true"]
        for j, c in enumerate(classes):
            pred_df[f"p_{c}"] = probs[:, j]
        pred_df.to_csv(out_dir / "predictions_test.csv", index=False)
        cm = np.array(m["confusion_matrix"])
        np.savetxt(out_dir / "confusion_matrix_test.csv", cm, fmt="%d", delimiter=",")
        short = [short_label(c) for c in classes]
        plot_confusion_matrix(cm, short, out_dir / "confusion_matrix_test.png",
                              f"Test confusion matrix (counts, n={len(frame)})")
        plot_confusion_matrix(cm, short, out_dir / "confusion_matrix_test_normalized.png",
                              "Test confusion matrix (row-normalised recall)", normalize=True)
        if "band4" in m:
            bcm = np.array(m["band4"]["confusion_matrix"])
            plot_confusion_matrix(bcm, [short_label(b) for b in BANDS], out_dir / "confusion_matrix_test_band4.png",
                                  f"Test confusion matrix, scored in 4 bands (n={len(frame)})")
        if verbose:
            print(header + report)

    if verbose:
        for split in ("val", "test"):
            m = results[split]
            line = (f"[{split.upper():4s}] acc {m['accuracy']:.4f} | bal-acc {m['balanced_accuracy']:.4f} | "
                    f"macro-F1 {m['macro_f1']:.4f} | n={m['n_images']}")
            if "band4" in m:
                line += f" || 4-band acc {m['band4']['accuracy']:.4f} bal-acc {m['band4']['balanced_accuracy']:.4f}"
            print(line)
        print(f"Evaluation outputs: {out_dir}")
    return results


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    ap.add_argument("--splits-path", default=None, help="override the split CSV stored in the checkpoint")
    ap.add_argument("--output-name", default=None, help="output folder name (default: the run name)")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    evaluate_checkpoint(args.checkpoint, get_device(args.device), splits_path=args.splits_path,
                        output_name=args.output_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
