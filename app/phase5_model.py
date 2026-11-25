"""Phase 5 four-band model in the Streamlit app: selection record, labels, disclaimer, fabric-level aggregation.

Scaffolding only: no Phase 5 model exists yet, so configs/phase5_app_model.json is absent and the app does not
offer the model. Once a real model passes the protocol, create the record with:

    python -m app.phase5_model select --checkpoint <model.pth> [--summary <cv summary json>]

Synthetic (pipeline-test) checkpoints are refused unless --allow-synthetic is given (tests only); if one is ever
selected, the app shows a PIPELINE TEST banner. Env var BWORTH_PHASE5_SELECTION overrides the record path (tests).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.phase3_model import DISCLAIMER  # noqa: E402  (identical wording for every experimental model)

MODEL_KEY = "phase5_band4"
LABEL_SCHEME = "phase5_band4"
EXPECTED_CLASSES = ["low", "medium", "high", "very_high"]
CLASS_LABELS = {
    "low": "Low cotton: 30–40%",
    "medium": "Medium cotton: 50–66%",
    "high": "High cotton: 80–95%",
    "very_high": "Very high cotton: 98–100%",
}
RANGE_NOTE = ("The percentage ranges are the verified cotton labels of the Phase 4 dataset bands, "
              "not continuous measurement ranges.")
SYNTHETIC_WARNING = ("PIPELINE TEST MODEL: trained on synthetic images to test the software. Its predictions are "
                     "meaningless for real fabrics.")


def selection_file() -> Path:
    return Path(os.environ.get("BWORTH_PHASE5_SELECTION", PROJECT_ROOT / "configs" / "phase5_app_model.json"))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _rel(p: Path) -> str:
    p = Path(p).resolve()
    try:
        return p.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return p.as_posix()


def write_selection(checkpoint: Path, summary: dict | None = None, allow_synthetic: bool = False,
                    out: Path | None = None) -> dict:
    import torch

    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    classes = [ck["idx_to_class"][i] for i in range(len(ck["idx_to_class"]))]
    if ck.get("label_scheme") != LABEL_SCHEME or classes != EXPECTED_CLASSES or not ck.get("trained"):
        raise ValueError("not a Phase 5 four-band checkpoint")
    if ck.get("synthetic") and not allow_synthetic:
        raise ValueError("refusing to select a synthetic pipeline-test checkpoint for the app")
    sel = {"model_key": MODEL_KEY, "title": "4-class verified (Phase 5)", "badge": "EXPERIMENTAL",
           "checkpoint": _rel(checkpoint), "checkpoint_sha256": sha256(Path(checkpoint)),
           "architecture": ck.get("architecture", "resnet50"), "synthetic": bool(ck.get("synthetic", False)),
           "snapshot_version": ck.get("snapshot_version"), "protocol_sha256": ck.get("protocol_sha256"),
           "lockbox_seal": ck.get("lockbox_seal"), "cv": summary}
    f = out or selection_file()
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(sel, indent=2, default=str), encoding="utf-8")
    return sel


def load_selection(verify_hash: bool = True) -> dict | None:
    f = selection_file()
    if not f.is_file():
        return None
    sel = json.loads(f.read_text(encoding="utf-8"))
    ck = Path(sel["checkpoint"])
    ck = ck if ck.is_absolute() else PROJECT_ROOT / ck
    if not ck.is_file() or (verify_hash and sha256(ck) != sel["checkpoint_sha256"]):
        return None
    sel["_checkpoint_path"] = str(ck)
    return sel


def app_spec(sel: dict) -> dict:
    title = ("[PIPELINE TEST] " if sel.get("synthetic") else "") + sel["title"]
    return {
        "title": title, "badge": "PIPELINE TEST" if sel.get("synthetic") else sel["badge"], "kind": MODEL_KEY,
        "checkpoint": Path(sel["_checkpoint_path"]), "metrics": PROJECT_ROOT / "__no_v2_metrics__.json", "cv_config": None,
        "labels": CLASS_LABELS, "cv": sel.get("cv"), "selection": sel, "synthetic": bool(sel.get("synthetic")),
        "disclaimer": DISCLAIMER, "range_note": RANGE_NOTE, "low_note": None,
        "description": ("Predicts one of 4 cotton bands from verified Phase 4 fabrics: low (30–40%), medium (50–66%), "
                        f"high (80–95%), very high (98–100%). Architecture: {sel.get('architecture')}; "
                        f"dataset snapshot {sel.get('snapshot_version')}."),
    }


def aggregate_fabric_prediction(predictor, images) -> dict:
    """Fabric-level prediction from several photographs of ONE fabric (mean of class probabilities),
    matching how Phase 5 models are evaluated."""
    results = [predictor.predict(im) for im in images]
    if not results:
        raise ValueError("no images")
    classes = list(results[0].probabilities)
    mean = {c: sum(r.probabilities[c] for r in results) / len(results) for c in classes}
    best = max(mean, key=mean.get)
    return {"predicted_class": best, "score": mean[best], "probabilities": mean, "n_images": len(results)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["select"])
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--summary", type=Path, default=None)
    ap.add_argument("--allow-synthetic", action="store_true")
    a = ap.parse_args(argv)
    summary = json.loads(a.summary.read_text()) if a.summary else None
    print(json.dumps(write_selection(a.checkpoint, summary, a.allow_synthetic), indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
