"""Phase 5 evaluation: fabric-level aggregation, metrics, bootstrap CIs and success criteria S1-S7.

Input predictions are image-level rows with fabric_id, band (truth) and p_<class> columns. The primary unit is the
fabric: probabilities of a fabric's photographs are averaged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

from training.phase5.protocol import CLASSES, load_protocol

PCOLS = [f"p_{c}" for c in CLASSES]


def aggregate_fabrics(pred: pd.DataFrame) -> pd.DataFrame:
    g = pred.groupby("fabric_id")
    out = g[PCOLS].mean()
    out["band"] = g.band.first()
    if g.band.nunique().max() > 1:
        raise ValueError("a fabric has rows with different true bands")
    out["predicted"] = [CLASSES[i] for i in out[PCOLS].to_numpy().argmax(1)]
    out["n_images"] = g.size()
    return out.reset_index()


def expected_calibration_error(conf: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> float:
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def fabric_metrics(fab: pd.DataFrame) -> dict:
    idx = {c: i for i, c in enumerate(CLASSES)}
    y = fab.band.map(idx).to_numpy()
    yp = fab.predicted.map(idx).to_numpy()
    labels = list(range(len(CLASSES)))
    p, r, f, s = precision_recall_fscore_support(y, yp, labels=labels, zero_division=0)
    far = {(idx["low"], idx["very_high"]), (idx["very_high"], idx["low"])}
    conf = fab[PCOLS].to_numpy().max(1)
    present = sorted(set(y))
    return {
        "n_fabrics": int(len(fab)),
        "balanced_accuracy": float(balanced_accuracy_score(y, yp)) if len(present) > 1 else float("nan"),
        "macro_f1": float(f1_score(y, yp, labels=labels, average="macro", zero_division=0)),
        "accuracy": float((y == yp).mean()),
        "per_band": {c: {"precision": float(p[i]), "recall": float(r[i]), "support": int(s[i])} for i, c in enumerate(CLASSES)},
        "far_band_error_rate": float(np.mean([(a, b) in far for a, b in zip(y, yp)])),
        "ece": expected_calibration_error(conf, (y == yp).astype(float)),
        "confusion_matrix": confusion_matrix(y, yp, labels=labels).tolist(),
    }


def bootstrap_ci(fab: pd.DataFrame, n: int | None = None, seed: int = 0, alpha: float = 0.05) -> tuple[float, float]:
    """95 % CI of fabric balanced accuracy, resampling fabrics within each band (stratified bootstrap)."""
    n = n or load_protocol()["metrics"]["bootstrap_resamples"]
    rng = np.random.default_rng(seed)
    idx = {c: i for i, c in enumerate(CLASSES)}
    y = fab.band.map(idx).to_numpy()
    yp = fab.predicted.map(idx).to_numpy()
    by_band = [np.flatnonzero(y == b) for b in sorted(set(y))]
    vals = []
    for _ in range(n):
        take = np.concatenate([rng.choice(ix, len(ix), replace=True) for ix in by_band])
        vals.append(balanced_accuracy_score(y[take], yp[take]))
    return float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))


def summarize_runs(run_preds: dict[str, pd.DataFrame], repeat_of: dict[str, int]) -> dict:
    """run_preds: run name -> image-level predictions on that run's test fold."""
    per_run = {name: fabric_metrics(aggregate_fabrics(p)) for name, p in run_preds.items()}
    ba = pd.Series({k: v["balanced_accuracy"] for k, v in per_run.items()})
    rep = pd.Series(repeat_of)
    per_repeat = ba.groupby(rep.reindex(ba.index)).mean()
    pooled = aggregate_fabrics(pd.concat(run_preds.values()).groupby(["fabric_id", "band"], as_index=False)[PCOLS].mean())
    pm = fabric_metrics(pooled)
    lo, hi = bootstrap_ci(pooled)
    return {"per_run": per_run, "balanced_accuracy_mean": float(ba.mean()),
            "balanced_accuracy_std": float(ba.std(ddof=1)) if len(ba) > 1 else 0.0,
            "repeat_std": float(per_repeat.std(ddof=1)) if len(per_repeat) > 1 else 0.0,
            "pooled": pm, "pooled_ci95": [lo, hi]}


def check_success(summary: dict, baseline_bal_acc: dict[str, float] | None = None,
                  lockbox_bal_acc: float | None = None, synthetic: bool = False) -> dict:
    s = load_protocol()["success_criteria"]
    pm, (lo, hi) = summary["pooled"], summary["pooled_ci95"]
    res = {
        "S1": summary["balanced_accuracy_mean"] >= s["S1_fabric_balanced_accuracy_mean_min"] and lo >= s["S1_fabric_balanced_accuracy_ci95_lower_min"],
        "S2": None if not baseline_bal_acc else all(summary["balanced_accuracy_mean"] - v >= s["S2_min_margin_over_colour_and_metadata_baselines"]
                                                    for k, v in baseline_bal_acc.items() if k in ("colour_histogram", "metadata")),
        "S3": all(v["recall"] >= s["S3_min_recall_per_band"] and v["precision"] >= s["S3_min_precision_per_band"] for v in pm["per_band"].values()),
        "S4": pm["far_band_error_rate"] <= s["S4_max_far_band_error_rate"],
        "S5": summary["repeat_std"] <= s["S5_max_std_across_repeats"],
        "S6": None if lockbox_bal_acc is None else (lo <= lockbox_bal_acc <= hi),
        "S7": pm["ece"] <= s["S7_max_expected_calibration_error"],
    }
    return {"criteria": res, "all_pass": all(v is True for v in res.values()),
            "note": ("PIPELINE TEST on synthetic data: criteria evaluated only to exercise the code; NOT model performance"
                     if synthetic else "")}
