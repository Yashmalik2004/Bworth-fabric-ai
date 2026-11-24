"""5-fold swatch-grouped cross-validation of the Phase-2 experiment grid.

    python -m training.composition.cross_validate run [--configs ...] [--folds 0 1 2 3 4]
    python -m training.composition.cross_validate aggregate

Grid: label scheme {fine13, band4} x colour augmentation {standard, strong}; every
config sees identical folds (processed/composition/cv/fold{k}.csv, built by
experiment_splits.py). Each fold: test = fold k, validation = fold k+1 (early
stopping / checkpoint selection), train = the other three.

Resumable: a fold whose predictions already exist is skipped. Fold checkpoints are
temporary and deleted after scoring; only predictions, metrics and curves are kept
in evaluation/composition/cv/<config>/fold<k>/.

13-class models are also scored in the 4-band space (probabilities summed per band),
which is how the two label schemes are compared.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import BAND_DEFINITIONS, EVALUATION_DIR, FINE_TO_BAND, CompositionConfig  # noqa: E402
from training.composition.dataset import apply_label_scheme  # noqa: E402
from training.composition.evaluate import (  # noqa: E402
    compute_metrics,
    fine_to_band_probs,
    predict_frame,
    short_label,
)
from training.composition.experiment_splits import CV_DIR, N_FOLDS  # noqa: E402
from training.composition.model import load_model_from_checkpoint  # noqa: E402
from training.composition.train import train  # noqa: E402
from utils.device import get_device  # noqa: E402
from utils.plotting import plot_confusion_matrix  # noqa: E402

CONFIGS = {
    "fine13_standard": {"label_scheme": "fine13", "color_aug": "standard"},
    "fine13_strong": {"label_scheme": "fine13", "color_aug": "strong"},
    "band4_standard": {"label_scheme": "band4", "color_aug": "standard"},
    "band4_strong": {"label_scheme": "band4", "color_aug": "strong"},
}
CV_OUT = EVALUATION_DIR / "cv"
TMP_CKPT = CV_DIR / "tmp_fold_model.pth"
BANDS = list(BAND_DEFINITIONS)


def run_fold(name: str, k: int, device: torch.device) -> None:
    out = CV_OUT / name / f"fold{k}"
    if (out / "predictions_test.csv").is_file():
        print(f"[skip] {name} fold {k} already done", flush=True)
        return
    cfg = CompositionConfig(
        **CONFIGS[name],
        splits_path=str(CV_DIR / f"fold{k}.csv"),
        run_name=f"cv/{name}/fold{k}",
        checkpoint_path=str(TMP_CKPT),
        output_dir=str(EVALUATION_DIR),
        num_workers=1,  # each Windows worker re-imports torch; CV runs many trainings back to back
    )
    print(f"\n=== {name} | fold {k} ===", flush=True)
    train(cfg, device)

    ck = torch.load(TMP_CKPT, map_location="cpu", weights_only=False)
    classes = [ck["idx_to_class"][i] for i in range(len(ck["idx_to_class"]))]
    model = load_model_from_checkpoint(ck, device)
    df = apply_label_scheme(pd.read_csv(cfg.splits_path), cfg.label_scheme, classes)
    test = df[df.split == "test"].reset_index(drop=True)
    probs = predict_frame(model, test, ck["preprocessing"], device)
    pred = test[["path", "label", "group"]].copy()
    pred["fold"] = k
    pred["true"] = [classes[i] for i in test["target"]]
    pred["predicted"] = [classes[i] for i in probs.argmax(1)]
    for j, c in enumerate(classes):
        pred[f"p_{c}"] = probs[:, j]
    pred.to_csv(out / "predictions_test.csv", index=False)
    (out / "fold_info.json").write_text(json.dumps({"best_epoch": ck["best_epoch"], "val_acc": ck["val_acc"],
                                                    "train_acc_at_best": ck["train_acc"]}, indent=2))
    del model
    TMP_CKPT.unlink(missing_ok=True)
    if device.type == "cuda":
        torch.cuda.empty_cache()


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def band_view(pred: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """(y_true_band, probs_band) for predictions of either scheme."""
    y = pred["label"].map(FINE_TO_BAND).map({b: i for i, b in enumerate(BANDS)}).to_numpy()
    if f"p_{BANDS[0]}" in pred:
        p = pred[[f"p_{b}" for b in BANDS]].to_numpy()
    else:
        fine = [c[2:] for c in pred.columns if c.startswith("p_")]
        p = fine_to_band_probs(pred[[f"p_{c}" for c in fine]].to_numpy(), fine)
    return y, p


def fine_view(pred: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    fine = [c[2:] for c in pred.columns if c.startswith("p_")]
    idx = {c: i for i, c in enumerate(fine)}
    return pred["label"].map(idx).to_numpy(), pred[[f"p_{c}" for c in fine]].to_numpy(), fine


def load_predictions(name: str) -> pd.DataFrame | None:
    parts = [pd.read_csv(p) for p in sorted((CV_OUT / name).glob("fold*/predictions_test.csv"))]
    return pd.concat(parts, ignore_index=True) if parts else None


def aggregate() -> dict:
    summary: dict = {"folds": N_FOLDS, "configs": {}}
    per_fold_rows, group_rows = [], []
    for name in CONFIGS:
        pred = load_predictions(name)
        if pred is None:
            continue
        folds_done = sorted(pred.fold.unique().tolist())
        entry: dict = {"folds_completed": folds_done, "n_predictions": len(pred)}
        fold_m = []
        for k in folds_done:
            fp = pred[pred.fold == k]
            y, p = band_view(fp)
            m = compute_metrics(y, p, BANDS)
            info = json.loads((CV_OUT / name / f"fold{k}" / "fold_info.json").read_text())
            row = {"config": name, "fold": k, "n": len(fp), "band4_accuracy": m["accuracy"],
                   "band4_balanced_accuracy": m["balanced_accuracy"], "band4_macro_f1": m["macro_f1"],
                   "best_epoch": info["best_epoch"], "val_acc_own_labels": info["val_acc"],
                   "train_acc_own_labels": info["train_acc_at_best"]}
            if not name.startswith("band4"):
                yf, pf, fine = fine_view(fp)
                mf = compute_metrics(yf, pf, fine)
                row.update({"fine13_accuracy": mf["accuracy"], "fine13_balanced_accuracy": mf["balanced_accuracy"],
                            "fine13_macro_f1": mf["macro_f1"]})
            per_fold_rows.append(row)
            fold_m.append(row)
        fm = pd.DataFrame(fold_m)
        for col in [c for c in fm.columns if c.startswith(("band4_", "fine13_"))]:
            entry[f"{col}_mean"] = float(fm[col].mean())
            entry[f"{col}_std"] = float(fm[col].std(ddof=1)) if len(fm) > 1 else float("nan")
        # pooled out-of-fold (every image predicted once, by a model that never saw its swatch)
        y, p = band_view(pred)
        pooled = compute_metrics(y, p, BANDS)
        entry["pooled_band4"] = {k: pooled[k] for k in ("accuracy", "balanced_accuracy", "macro_f1", "per_class",
                                                        "confusion_matrix")}
        plot_confusion_matrix(np.array(pooled["confusion_matrix"]), [short_label(b) for b in BANDS],
                              CV_OUT / name / "confusion_matrix_band4_pooled.png",
                              f"{name}: pooled out-of-fold, 4 bands (n={len(pred)})")
        plot_confusion_matrix(np.array(pooled["confusion_matrix"]), [short_label(b) for b in BANDS],
                              CV_OUT / name / "confusion_matrix_band4_pooled_normalized.png",
                              f"{name}: pooled out-of-fold, 4 bands (row-normalised)", normalize=True)
        if not name.startswith("band4"):
            yf, pf, fine = fine_view(pred)
            pf_m = compute_metrics(yf, pf, fine)
            entry["pooled_fine13"] = {k: pf_m[k] for k in ("accuracy", "balanced_accuracy", "macro_f1")}
            plot_confusion_matrix(np.array(pf_m["confusion_matrix"]), [short_label(c) for c in fine],
                                  CV_OUT / name / "confusion_matrix_fine13_pooled.png",
                                  f"{name}: pooled out-of-fold, 13 classes (n={len(pred)})")
        # per swatch group, band-level correctness (and fine-level for 13-class)
        pred = pred.assign(band_true=[BANDS[i] for i in y], band_pred=[BANDS[i] for i in p.argmax(1)])
        pred["band_correct"] = pred.band_true == pred.band_pred
        if not name.startswith("band4"):
            pred["fine_correct"] = pred.predicted == pred.label
        for (lab, g), gd in pred.groupby(["label", "group"]):
            group_rows.append({"config": name, "label": lab, "group": g, "fold": int(gd.fold.iloc[0]), "n": len(gd),
                               "band_correct": int(gd.band_correct.sum()),
                               "fine_correct": int(gd.fine_correct.sum()) if "fine_correct" in gd else None,
                               "band_predictions": dict(gd.band_pred.value_counts())})
        summary["configs"][name] = entry

    pf = pd.DataFrame(per_fold_rows)
    CV_OUT.mkdir(parents=True, exist_ok=True)
    pf.to_csv(CV_OUT / "per_fold_metrics.csv", index=False)
    pd.DataFrame(group_rows).to_csv(CV_OUT / "per_group_results.csv", index=False)

    # paired per-fold comparisons (same folds, same test images)
    comps = {}
    piv = pf.pivot(index="fold", columns="config", values="band4_balanced_accuracy")
    for a, b in [("band4_standard", "fine13_standard"), ("band4_strong", "band4_standard"),
                 ("fine13_strong", "fine13_standard"), ("band4_strong", "fine13_standard")]:
        if a in piv and b in piv:
            d = (piv[a] - piv[b]).dropna()
            comps[f"{a} - {b}"] = {"metric": "band4_balanced_accuracy", "mean_diff": float(d.mean()),
                                   "per_fold": d.round(4).tolist(), "folds_better": int((d > 0).sum()),
                                   "folds_compared": int(len(d))}
    summary["paired_comparisons"] = comps
    (CV_OUT / "cv_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(pf.round(4).to_string())
    print(json.dumps(comps, indent=2))
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["run", "aggregate", "fold"])
    ap.add_argument("--configs", nargs="*", default=list(CONFIGS))
    ap.add_argument("--folds", nargs="*", type=int, default=list(range(N_FOLDS)))
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    if args.command == "fold":            # one (config, fold) in this process
        run_fold(args.configs[0], args.folds[0], get_device(args.device))
        return 0
    if args.command == "run":
        # Each (config, fold) runs in a fresh subprocess so memory is fully released between runs.
        for k in args.folds:              # fold-major: every config gets fold 0 before fold 1, ...
            for name in args.configs:
                if (CV_OUT / name / f"fold{k}" / "predictions_test.csv").is_file():
                    print(f"[skip] {name} fold {k} already done", flush=True)
                    continue
                cmd = [sys.executable, "-m", "training.composition.cross_validate", "fold",
                       "--configs", name, "--folds", str(k)] + (["--device", args.device] if args.device else [])
                rc = subprocess.call(cmd, cwd=PROJECT_ROOT)
                if rc != 0:
                    print(f"[fail] {name} fold {k} exited with {rc}", flush=True)
                    return rc
    aggregate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
