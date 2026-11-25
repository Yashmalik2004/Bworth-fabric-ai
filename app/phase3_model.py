"""Phase 3 corrected three-class model: checkpoint selection, display labels and app registry entry.

    python -m app.phase3_model select      # (re)derive the checkpoint choice -> configs/phase3_app_model.json

Selection rule (agreed with the project owner):
  among the 10 Stage B *baseline* checkpoints (5 folds x 2 seeds), take the highest validation accuracy,
  ties broken by lower validation loss. Test-set results are never used for selection. Every candidate is
  validated (variant, 3 classes in the expected order, label scheme, strong colour augmentation,
  preprocessing) and the chosen file is pinned by SHA-256.
Read-only with respect to Stage B: checkpoints, reports and evaluation outputs are never modified.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

STAGE_B = PROJECT_ROOT / "experiments" / "phase3_global_fabric_audit" / "stage_b"
SELECTION_FILE = PROJECT_ROOT / "configs" / "phase3_app_model.json"
MODEL_KEY = "phase3_band3"

EXPECTED_CLASSES = ["medium_cotton", "high_cotton", "very_high_cotton"]
EXPECTED_PREPROCESSING = {"resize_size": 366, "image_size": 320,
                          "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
CLASS_LABELS = {
    "medium_cotton": "Medium cotton: 50–66%",
    "high_cotton": "High cotton: 80–95%",
    "very_high_cotton": "Very high cotton: 98–99%",
}
RANGE_NOTE = ("The percentage ranges are the cotton labels present in the training dataset, "
              "not continuous measurement ranges.")
DISCLAIMER = ("Experimental visual estimate. This model has limited validation on independent physical fabrics. "
              "It does not chemically verify fibre composition or determine exact cotton percentage. "
              "Do not use this result for automatic garment valuation or token allocation.")
LOW_NOTE = ("Low cotton (30–40%) is not predicted by this model: the class is currently not evaluable because "
            "the verified dataset contains too few independent physical-fabric samples (2 fabrics).")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(p: Path) -> str:
    return Path(p).resolve().relative_to(PROJECT_ROOT).as_posix()


def select() -> dict:
    import pandas as pd
    import torch

    candidates, problems = [], []
    for p in sorted((STAGE_B / "models" / "baseline").glob("*.pth")):
        ck = torch.load(p, map_location="cpu", weights_only=False)
        classes = [ck["idx_to_class"][i] for i in range(len(ck["idx_to_class"]))]
        rec = {"checkpoint": rel(p), "variant": ck.get("variant"), "fold": ck.get("fold"), "seed": ck.get("seed"),
               "best_epoch": ck.get("best_epoch"), "val_acc": float(ck["val_acc"]), "val_loss": float(ck["val_loss"]),
               "label_scheme": ck.get("label_scheme"), "classes": classes,
               "color_aug": ck["config"].get("color_aug"), "preprocessing": ck["preprocessing"]}
        ok = (rec["variant"] == "baseline" and classes == EXPECTED_CLASSES and rec["label_scheme"] == "band3_stageB"
              and rec["color_aug"] == "strong" and rec["preprocessing"] == EXPECTED_PREPROCESSING and ck.get("trained"))
        rec["valid"] = bool(ok)
        if not ok:
            problems.append(rec["checkpoint"])
        candidates.append(rec)
    if len(candidates) != 10 or problems:
        raise RuntimeError(f"Unexpected Stage B baseline checkpoints: n={len(candidates)}, invalid={problems}")
    chosen = sorted(candidates, key=lambda r: (-r["val_acc"], r["val_loss"]))[0]

    runs = pd.read_csv(STAGE_B / "reports" / "per_run_metrics.csv")
    base = runs[(runs.variant == "baseline") & (runs.test_subset == "all_test")]
    pc = pd.read_csv(STAGE_B / "reports" / "per_class_pooled_metrics.csv")
    pc = pc[(pc.variant == "baseline") & (pc.test_subset == "all_test")]
    ds = pd.read_csv(STAGE_B / "splits" / "stage_b_dataset_3class.csv")
    selection = {
        "model_key": MODEL_KEY,
        "title": "3-class corrected (Phase 3)",
        "badge": "EXPERIMENTAL",
        "checkpoint": chosen["checkpoint"],
        "checkpoint_sha256": sha256(PROJECT_ROOT / chosen["checkpoint"]),
        "selection_rule": "highest validation accuracy among the 10 Stage B baseline checkpoints; ties -> lower "
                          "validation loss; test results not used",
        "selection_caveat": "Its validation fold is the easiest fold, so its own validation score (shown below) is "
                            "not representative; use the cross-validated metrics.",
        "chosen": chosen,
        "candidates": candidates,
        "held_out_samples": rel(STAGE_B / "splits" / f"baseline_fold{chosen['fold']}.csv"),
        "trained_on": "3 of 5 fabric-disjoint folds of the strict corrected dataset (not a final all-data model)",
        "cv": {
            "runs": int(len(base)),
            "balanced_accuracy_mean": float(base.balanced_accuracy.mean()),
            "balanced_accuracy_std": float(base.balanced_accuracy.std(ddof=1)),
            "accuracy_mean": float(base.accuracy.mean()), "accuracy_std": float(base.accuracy.std(ddof=1)),
            "macro_f1_mean": float(base.macro_f1.mean()), "macro_f1_std": float(base.macro_f1.std(ddof=1)),
            "per_class": {r["class"]: {"precision": float(r.precision), "recall": float(r.recall)}
                          for r in pc.to_dict("records") for r in [pd.Series(r)]},
        },
        "fabrics_per_class": ds.groupby("label").global_fabric_id.nunique().reindex(EXPECTED_CLASSES).astype(int).to_dict(),
        "photos_per_class": ds.label.value_counts().reindex(EXPECTED_CLASSES).astype(int).to_dict(),
    }
    return selection


def write_selection() -> dict:
    sel = select()
    SELECTION_FILE.write_text(json.dumps(sel, indent=2), encoding="utf-8")
    return sel


def load_selection(verify_hash: bool = True) -> dict | None:
    """The recorded selection, or None if missing / checkpoint missing / checksum mismatch."""
    if not SELECTION_FILE.is_file():
        return None
    sel = json.loads(SELECTION_FILE.read_text(encoding="utf-8"))
    ck = PROJECT_ROOT / sel["checkpoint"]
    if not ck.is_file():
        return None
    if verify_hash and sha256(ck) != sel["checkpoint_sha256"]:
        return None
    return sel


def app_spec(sel: dict) -> dict:
    cv = sel["cv"]
    return {
        "title": sel["title"],
        "badge": sel["badge"],
        "kind": MODEL_KEY,
        "checkpoint": PROJECT_ROOT / sel["checkpoint"],
        "metrics": PROJECT_ROOT / "__no_v2_metrics__.json",   # not evaluated on the v2 split
        "cv_config": None,
        "labels": CLASS_LABELS,
        "description": ("Predicts one of 3 cotton bands: medium (50–66%), high (80–95%), very high (98–99%). "
                        f"Trained on verified, fabric-disjoint data; checkpoint {Path(sel['checkpoint']).name} "
                        f"(fold {sel['chosen']['fold']}, seed {sel['chosen']['seed']})."),
        "cv": cv,
        "selection": sel,
        "sample_splits": PROJECT_ROOT / sel["held_out_samples"],
    }


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "select":
        s = write_selection()
        print(json.dumps({k: s[k] for k in ("checkpoint", "checkpoint_sha256", "selection_rule", "held_out_samples",
                                            "cv", "fabrics_per_class")}, indent=2))
        print("chosen:", s["chosen"])
    else:
        print(__doc__)
