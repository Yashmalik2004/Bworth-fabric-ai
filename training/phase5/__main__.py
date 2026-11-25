"""Phase 5 command line.

    python -m training.phase5 snapshot   --phase4-root datasets/phase4_verified
    python -m training.phase5 verify     --snapshot experiments/phase5/snapshots/<version>
    python -m training.phase5 readiness  --snapshot ...
    python -m training.phase5 seal-lockbox --snapshot ...
    python -m training.phase5 splits     --snapshot ...
    python -m training.phase5 e0         [--work experiments/phase5/pipeline_tests/e0_<time>]   # synthetic only

Real-data training is not exposed here yet; train.train_split refuses it unless READY + explicit approval.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from training.phase5.protocol import PHASE5_DIR, SYNTHETIC_BANNER


def run_e0(work: Path, fabrics_per_band: int = 12, folds: int = 3, repeats: int = 1, epochs: int = 2,
           arch: str = "resnet50", image_size: int = 64) -> dict:
    """End-to-end synthetic pipeline test. Results are NOT model performance."""
    import pandas as pd

    from training.phase5 import readiness, snapshot, splits
    from training.phase5.baselines import predict_baseline
    from training.phase5.evaluate import aggregate_fabrics, check_success, fabric_metrics, summarize_runs
    from training.phase5.synthetic import make_synthetic_dataset
    from training.phase5.train import train_split

    work = Path(work)
    work.mkdir(parents=True, exist_ok=False)
    (work / "README.md").write_text(f"# {SYNTHETIC_BANNER}\n\nE0 pipeline test output. Every number here comes from "
                                    "synthetic images and says nothing about real fabric composition performance.\n",
                                    encoding="utf-8")
    make_synthetic_dataset(work / "phase4_synthetic", fabrics_per_band=fabrics_per_band, photos_per_fabric=4, seed=0,
                           image_size=96)
    snap = snapshot.create_snapshot(work / "phase4_synthetic", work / "snapshots", version="SYNTH-E0")
    pre = readiness.evaluate(snap)
    splits.seal_lockbox(snap)
    post = readiness.evaluate(snap)
    files = splits.make_cv_splits(snap, work / "splits", repeats=repeats, folds=folds)
    tables = snapshot.load_tables(snap)
    root = snapshot.image_root(snap)
    runs, rep, base_preds = {}, {}, {"majority": [], "colour_histogram": [], "metadata": []}
    for f in files:
        df = pd.read_csv(f, dtype=str, keep_default_na=False)
        tr, te = df[df.split != "test"], df[df.split == "test"]
        for kind in base_preds:
            p = predict_baseline(kind, tr, te, tables["fabrics"], root)
            d = te[["fabric_id", "band"]].copy()
            for j, c in enumerate(["low", "medium", "high", "very_high"]):
                d[f"p_{c}"] = p[:, j]
            base_preds[kind].append(d)
        info = train_split(f, snap, work / "runs" / f.stem, arch=arch, seed=42, epochs=epochs,
                           image_size=image_size, resize_size=int(image_size * 1.15), batch_size=16)
        runs[f.stem] = pd.read_csv(work / "runs" / f.stem / "predictions_test.csv")
        rep[f.stem] = int(f.stem.split("_")[0][3:])
    summary = summarize_runs(runs, rep)
    base_ba = {k: fabric_metrics(aggregate_fabrics(pd.concat(v)))["balanced_accuracy"] for k, v in base_preds.items()}
    crit = check_success(summary, base_ba, synthetic=True)
    out = {"banner": SYNTHETIC_BANNER, "readiness_before_lockbox": pre["overall"], "readiness_after_lockbox": post["overall"],
           "gate_status": post["gate_status"], "splits": len(files), "model": arch,
           "balanced_accuracy_mean": summary["balanced_accuracy_mean"], "pooled_ci95": summary["pooled_ci95"],
           "baselines_balanced_accuracy": base_ba, "success_criteria_mechanics": crit}
    (work / "E0_PIPELINE_TEST_RESULTS.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["snapshot", "verify", "readiness", "seal-lockbox", "splits", "e0"])
    ap.add_argument("--phase4-root", type=Path, default=Path("datasets/phase4_verified"))
    ap.add_argument("--snapshot", type=Path)
    ap.add_argument("--out", type=Path, default=PHASE5_DIR / "snapshots")
    ap.add_argument("--work", type=Path, default=None)
    a = ap.parse_args(argv)
    from training.phase5 import readiness, snapshot, splits

    if a.command == "snapshot":
        print(snapshot.create_snapshot(a.phase4_root, a.out))
    elif a.command == "verify":
        print(json.dumps(snapshot.verify_snapshot(a.snapshot), indent=2))
    elif a.command == "readiness":
        r = readiness.evaluate(a.snapshot)
        print(json.dumps({"overall": r["overall"], "gates": r["gate_status"]}, indent=2))
    elif a.command == "seal-lockbox":
        print(json.dumps(splits.seal_lockbox(a.snapshot), indent=2))
    elif a.command == "splits":
        print(splits.make_cv_splits(a.snapshot, Path(a.snapshot) / "splits"))
    elif a.command == "e0":
        work = a.work or PHASE5_DIR / "pipeline_tests" / ("e0_" + datetime.now().strftime("%Y%m%dT%H%M%S"))
        print(json.dumps(run_e0(work), indent=2, default=str))
        print(SYNTHETIC_BANNER)
    return 0


if __name__ == "__main__":
    sys.exit(main())
