"""Phase 3 Stage B: corrected three-class experiment (strict dataset) + former-conflict ablation.

    python experiments/phase3_global_fabric_audit/stage_b/scripts/stage_b.py prepare
    python experiments/phase3_global_fabric_audit/stage_b/scripts/stage_b.py run --variant baseline
    python experiments/phase3_global_fabric_audit/stage_b/scripts/stage_b.py run --variant ablationA
    python experiments/phase3_global_fabric_audit/stage_b/scripts/stage_b.py report

Variants (identical folds, seeds, validation and test sets; only the TRAINING images differ):
  baseline   = strict dataset, 3 bands (= ablation variant B, "expanded": includes former-conflict photos)
  ablationA  = baseline with every former-conflict photograph removed from the training split only
Training recipe = Phase-2 best (band4_strong) recipe with 3 output classes.
Writes only inside experiments/phase3_global_fabric_audit/stage_b/.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import CompositionConfig  # noqa: E402

A = PROJECT_ROOT / "experiments" / "phase3_global_fabric_audit"
B = A / "stage_b"
MODELS, CONFIGS, SPLITS, LOGS, EVAL, REPORTS = (B / d for d in ("models", "configs", "splits", "logs", "evaluation", "reports"))
CLASSES = ["medium_cotton", "high_cotton", "very_high_cotton"]
SHORT = {"medium_cotton": "Medium", "high_cotton": "High", "very_high_cotton": "Very high"}
SEEDS = [42, 43]
N_FOLDS = 5
VARIANTS = ("baseline", "ablationA")
LOW_STATEMENT = ("Low cotton composition is not evaluable because the dataset does not contain enough "
                 "independent, verified physical-fabric samples.")


# --------------------------------------------------------------------------- #
# Prepare + integrity checks
# --------------------------------------------------------------------------- #
def prepare() -> int:
    for d in (MODELS, CONFIGS, SPLITS, LOGS, EVAL, REPORTS):
        d.mkdir(parents=True, exist_ok=True)
    man = pd.read_csv(A / "datasets" / "corrected_dataset_manifest.csv")
    folds = pd.read_csv(A / "splits" / "five_fold_assignments.csv")[["photo_id", "fold"]]
    mapping = pd.read_csv(A / "audit" / "global_fabric_mapping.csv")
    inv = pd.read_csv(A / "audit" / "complete_image_inventory.csv", usecols=["image_id", "phase1_conflict_cluster",
                                                                             "phase3_provisional_status"])
    df = man.merge(folds, on="photo_id", how="left", validate="one_to_one")
    checks, notes = {}, {}
    n_low = int((df.four_class_label == "low_cotton").sum())
    notes["low_removed"] = {"photographs": n_low,
                            "fabrics": int(df.loc[df.four_class_label == "low_cotton", "global_fabric_id"].nunique())}
    df = df[df.four_class_label != "low_cotton"].reset_index(drop=True)
    df = df.rename(columns={"image_path": "path"})
    df["label"] = df.four_class_label
    df["target"] = df.label.map({c: i for i, c in enumerate(CLASSES)})
    df["label_idx"] = df.target

    # former-conflict flag: any file of the photograph belonged to a Phase-1 label-conflict cluster
    files = mapping.merge(inv, on="image_id")
    conflict_photo = files.groupby("photo_id").phase1_conflict_cluster.apply(lambda s: s.notna().any())
    prov_photo = files.groupby("photo_id").phase3_provisional_status.apply(lambda s: (s == "provisional_accept").any())
    sha_of = files.groupby("photo_id").sha256.agg(lambda s: sorted(set(s)))
    df["former_conflict"] = df.photo_id.map(conflict_photo).astype(bool)
    df["phase3_recoverable_14"] = df.photo_id.map(prov_photo).astype(bool)
    df["sha256_all_copies"] = df.photo_id.map(lambda p: ";".join(sha_of[p]))

    # ---- integrity checks (stop on failure) -------------------------------- #
    checks["every_row_has_fold"] = bool(df.fold.notna().all())
    checks["each_photograph_once"] = bool(df.photo_id.is_unique)
    checks["each_image_path_once"] = bool(df.path.is_unique)
    checks["each_photo_one_fabric"] = bool(df.groupby("photo_id").global_fabric_id.nunique().max() == 1)
    checks["each_fabric_one_fold"] = bool(df.groupby("global_fabric_id").fold.nunique().max() == 1)
    all_sha = df.assign(h=df.sha256_all_copies.str.split(";")).explode("h")
    checks["no_sha256_in_two_folds"] = bool(all_sha.groupby("h").fold.nunique().max() == 1)
    checks["no_sha256_in_two_photographs"] = bool(all_sha.groupby("h").photo_id.nunique().max() == 1)
    checks["each_fabric_one_band"] = bool(df.groupby("global_fabric_id").label.nunique().max() == 1)
    grp = pd.read_csv(A / "audit" / "physical_fabric_groups.csv").set_index("global_fabric_id")
    checks["only_consistent_confirmed_or_distinct_fabrics"] = bool(
        (grp.loc[df.global_fabric_id.unique(), "label_consistency_status"] == "CONSISTENT_SINGLE_BAND").all()
        and grp.loc[df.global_fabric_id.unique(), "identity_status"].isin(["CONFIRMED_MATCH", "DISTINCT_FABRIC"]).all())
    checks["files_exist"] = bool(all((PROJECT_ROOT / p).is_file() for p in df.path))

    split_rows = []
    for variant in VARIANTS:
        for k in range(N_FOLDS):
            m = df.copy()
            m["split"] = "train"
            m.loc[m.fold == k, "split"] = "test"
            m.loc[m.fold == (k + 1) % N_FOLDS, "split"] = "val"
            if variant == "ablationA":
                m = m[~((m.split == "train") & m.former_conflict)]
            for s in ("train", "val", "test"):
                sub = m[m.split == s]
                split_rows.append({"variant": variant, "fold": k, "split": s, "photographs": len(sub),
                                   "fabrics": sub.global_fabric_id.nunique(),
                                   **{f"{SHORT[c]}_photos": int((sub.label == c).sum()) for c in CLASSES},
                                   **{f"{SHORT[c]}_fabrics": int(sub.loc[sub.label == c, "global_fabric_id"].nunique()) for c in CLASSES}})
            m.to_csv(SPLITS / f"{variant}_fold{k}.csv", index=False)
    sr = pd.DataFrame(split_rows)
    sr.to_csv(SPLITS / "split_distribution.csv", index=False)
    checks["every_train_split_has_all_3_classes"] = bool((sr[sr.split == "train"][[f"{SHORT[c]}_photos" for c in CLASSES]] > 0).all().all())
    checks["every_val_split_has_all_3_classes"] = bool((sr[sr.split == "val"][[f"{SHORT[c]}_photos" for c in CLASSES]] > 0).all().all())
    checks["every_test_split_has_all_3_classes"] = bool((sr[sr.split == "test"][[f"{SHORT[c]}_photos" for c in CLASSES]] > 0).all().all())
    # ablation fairness: identical val/test populations between variants
    same_eval = True
    for k in range(N_FOLDS):
        a = pd.read_csv(SPLITS / f"baseline_fold{k}.csv"); b = pd.read_csv(SPLITS / f"ablationA_fold{k}.csv")
        for s in ("val", "test"):
            same_eval &= sorted(a[a.split == s].photo_id) == sorted(b[b.split == s].photo_id)
        same_eval &= set(b[b.split == "train"].photo_id) <= set(a[a.split == "train"].photo_id)
    checks["ablation_identical_val_and_test_populations"] = bool(same_eval)

    df.to_csv(SPLITS / "stage_b_dataset_3class.csv", index=False)
    notes.update({
        "photographs": int(len(df)), "fabrics": int(df.global_fabric_id.nunique()),
        "photographs_per_class": df.label.value_counts().reindex(CLASSES).to_dict(),
        "fabrics_per_class": df.groupby("label").global_fabric_id.nunique().reindex(CLASSES).to_dict(),
        "former_conflict_photographs": int(df.former_conflict.sum()),
        "former_conflict_per_class": df[df.former_conflict].label.value_counts().reindex(CLASSES, fill_value=0).to_dict(),
        "former_conflict_fabrics": int(df[df.former_conflict].global_fabric_id.nunique()),
        "fabrics_made_only_of_former_conflict_photos": int((df.groupby("global_fabric_id").former_conflict.all()).sum()),
        "phase3_recoverable_14_in_dataset": int(df.phase3_recoverable_14.sum()),
        "phase3_recoverable_14_that_are_former_conflict": int((df.phase3_recoverable_14 & df.former_conflict).sum()),
        "fabrics_per_fold": df.groupby("fold").global_fabric_id.nunique().to_dict(),
    })
    out = {"checks": checks, "all_passed": all(checks.values()), "notes": notes, "low_statement": LOW_STATEMENT}
    (SPLITS / "integrity_report.json").write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps(out, indent=2, default=str))
    print(sr.to_string())
    return 0 if out["all_passed"] else 2


# --------------------------------------------------------------------------- #
# One training run
# --------------------------------------------------------------------------- #
def train_one(variant: str, fold: int, seed: int) -> int:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader

    from training.composition.dataset import FabricDataset
    from training.composition.evaluate import compute_metrics, predict_frame
    from training.composition.model import build_model, load_model_from_checkpoint, parameter_groups
    from training.composition.train import run_epoch, save_checkpoint, set_seed
    from training.composition.transforms import build_eval_transform, build_train_transform
    from utils.device import amp_enabled, describe_device, get_device
    from utils.plotting import plot_training_curves

    run = f"{variant}/fold{fold}_seed{seed}"
    out = EVAL / variant / f"fold{fold}_seed{seed}"
    if (out / "predictions_test.csv").is_file():
        print(f"[skip] {run}")
        return 0
    out.mkdir(parents=True, exist_ok=True)
    # Phase-2 best recipe (band4_strong): CompositionConfig defaults + strong colour augmentation, 1 loader worker
    cfg = CompositionConfig(label_scheme="band3_stageB", color_aug="strong", seed=seed, num_workers=1,
                            splits_path=str(SPLITS / f"{variant}_fold{fold}.csv"), run_name=run,
                            checkpoint_path=str(MODELS / variant / f"fold{fold}_seed{seed}.pth"), output_dir=str(EVAL))
    (CONFIGS / variant).mkdir(parents=True, exist_ok=True)
    (CONFIGS / variant / f"fold{fold}_seed{seed}.json").write_text(json.dumps(cfg.to_dict(), indent=2))
    set_seed(seed)
    device = get_device()
    df = pd.read_csv(cfg.splits_path)
    tr, va, te = (df[df.split == s].reset_index(drop=True) for s in ("train", "val", "test"))
    pre = cfg.preprocessing()
    kw = dict(num_workers=cfg.num_workers, pin_memory=device.type == "cuda", persistent_workers=cfg.num_workers > 0)
    tl = DataLoader(FabricDataset(tr, build_train_transform(pre, cfg.color_aug)), batch_size=cfg.batch_size, shuffle=True,
                    drop_last=len(tr) > cfg.batch_size, **kw)
    vl = DataLoader(FabricDataset(va, build_eval_transform(pre)), batch_size=cfg.batch_size * 2, shuffle=False, **kw)
    counts = np.bincount(tr.target, minlength=len(CLASSES)).astype(float)
    w = torch.tensor(counts.sum() / (len(CLASSES) * np.maximum(counts, 1)), dtype=torch.float32, device=device)
    crit = nn.CrossEntropyLoss(weight=w, label_smoothing=cfg.label_smoothing)
    eval_crit = nn.CrossEntropyLoss(weight=w)
    model = build_model(len(CLASSES), pretrained=cfg.pretrained, dropout=cfg.dropout).to(device)
    opt = torch.optim.AdamW(parameter_groups(model, cfg.lr_backbone, cfg.lr_head), weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    use_amp = amp_enabled(device, cfg.amp)
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    print(f"{run}: {describe_device(device)} | train {len(tr)} val {len(va)} test {len(te)} | class weights {w.tolist()}", flush=True)
    hist, best, stale = [], {"val_acc": -1.0, "val_loss": float("inf"), "epoch": 0}, 0
    ck_path = Path(cfg.checkpoint_path)
    for ep in range(1, cfg.epochs + 1):
        t0 = time.time()
        trl, tra = run_epoch(model, tl, crit, device, use_amp, opt, scaler)
        val, vaa = run_epoch(model, vl, eval_crit, device, use_amp)
        sched.step()
        hist.append({"epoch": ep, "train_loss": trl, "train_acc": tra, "val_loss": val, "val_acc": vaa, "seconds": time.time() - t0})
        if vaa > best["val_acc"] or (vaa == best["val_acc"] and val < best["val_loss"]):
            best, stale = {"val_acc": vaa, "val_loss": val, "epoch": ep}, 0
            save_checkpoint(ck_path, {
                "model_state_dict": model.state_dict(), "architecture": "resnet50", "label_scheme": "band3_stageB",
                "class_to_idx": {c: i for i, c in enumerate(CLASSES)}, "idx_to_class": dict(enumerate(CLASSES)),
                "preprocessing": pre, "config": cfg.to_dict(), "best_epoch": ep, "val_acc": vaa, "val_loss": val,
                "train_acc": tra, "trained": True, "variant": variant, "fold": fold, "seed": seed})
        else:
            stale += 1
        print(f"  ep {ep:2d} train {trl:.3f}/{tra:.3f} val {val:.3f}/{vaa:.3f}", flush=True)
        if stale >= cfg.early_stopping_patience:
            break
    pd.DataFrame(hist).to_csv(out / "history.csv", index=False)
    plot_training_curves(hist, best["epoch"], out / "training_curves.png")
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    model = load_model_from_checkpoint(ck, device)
    probs = predict_frame(model, te, pre, device)
    pred = te[["photo_id", "path", "label", "global_fabric_id", "former_conflict"]].copy()
    pred["fold"], pred["seed"], pred["variant"] = fold, seed, variant
    pred["predicted"] = [CLASSES[i] for i in probs.argmax(1)]
    for j, c in enumerate(CLASSES):
        pred[f"p_{c}"] = probs[:, j]
    pred.to_csv(out / "predictions_test.csv", index=False)
    m = compute_metrics(te.target.to_numpy(), probs, CLASSES)
    info = {"variant": variant, "fold": fold, "seed": seed, "best_epoch": best["epoch"], "epochs_run": len(hist),
            "best_val_acc": best["val_acc"], "train_acc_at_best": hist[best["epoch"] - 1]["train_acc"],
            "n_train": len(tr), "n_val": len(va), "n_test": len(te), "test_metrics": m}
    (out / "run_summary.json").write_text(json.dumps(info, indent=2))
    print(f"{run}: best epoch {best['epoch']} val {best['val_acc']:.3f} | test bal-acc {m['balanced_accuracy']:.3f} acc {m['accuracy']:.3f}")
    return 0


def run_variant(variant: str) -> int:
    for k in range(N_FOLDS):
        for s in SEEDS:
            if (EVAL / variant / f"fold{k}_seed{s}" / "predictions_test.csv").is_file():
                print(f"[skip] {variant} fold{k} seed{s}", flush=True)
                continue
            rc = subprocess.call([sys.executable, __file__, "_one", "--variant", variant, "--fold", str(k), "--seed", str(s)],
                                 cwd=PROJECT_ROOT)
            if rc != 0:
                print(f"[fail] {variant} fold{k} seed{s} exit {rc}", flush=True)
                return rc
    return 0


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def metrics_of(pred: pd.DataFrame) -> dict:
    from training.composition.evaluate import compute_metrics

    y = pred.label.map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
    p = pred[[f"p_{c}" for c in CLASSES]].to_numpy()
    return compute_metrics(y, p, CLASSES)


def load_preds(variant: str) -> pd.DataFrame | None:
    parts = [pd.read_csv(p) for p in sorted((EVAL / variant).glob("fold*_seed*/predictions_test.csv"))]
    return pd.concat(parts, ignore_index=True) if parts else None


def report() -> int:
    from utils.plotting import plot_confusion_matrix

    REPORTS.mkdir(parents=True, exist_ok=True)
    out = {"low_statement": LOW_STATEMENT, "variants": {}, "paired": {}}
    rows, perclass = [], []
    preds = {v: load_preds(v) for v in VARIANTS}
    for v, P in preds.items():
        if P is None:
            continue
        for subset in ("all_test", "non_conflict_test"):
            Q = P if subset == "all_test" else P[~P.former_conflict]
            for (k, s), g in Q.groupby(["fold", "seed"]):
                m = metrics_of(g)
                rows.append({"variant": v, "test_subset": subset, "fold": k, "seed": s, "n": len(g),
                             "balanced_accuracy": m["balanced_accuracy"], "accuracy": m["accuracy"], "macro_f1": m["macro_f1"],
                             **{f"recall_{SHORT[c]}": m["per_class"][c]["recall"] for c in CLASSES},
                             **{f"precision_{SHORT[c]}": m["per_class"][c]["precision"] for c in CLASSES}})
            pooled = metrics_of(Q)  # every test photo predicted once per seed -> 2 predictions per photo
            per_seed_pooled = {int(s): {k2: metrics_of(g)[k2] for k2 in ("balanced_accuracy", "accuracy", "macro_f1")}
                               for s, g in Q.groupby("seed")}
            out["variants"].setdefault(v, {})[subset] = {
                "pooled_over_seeds": {k2: pooled[k2] for k2 in ("n_images", "balanced_accuracy", "accuracy", "macro_f1",
                                                               "per_class", "confusion_matrix")},
                "pooled_per_seed": per_seed_pooled}
            for c in CLASSES:
                perclass.append({"variant": v, "test_subset": subset, "class": SHORT[c], **pooled["per_class"][c]})
            if subset == "all_test":
                cm = np.array(pooled["confusion_matrix"])
                lab = [SHORT[c] for c in CLASSES]
                plot_confusion_matrix(cm, lab, REPORTS / f"confusion_matrix_{v}.png",
                                      f"{v}: pooled out-of-fold, 2 seeds (n={len(Q)} predictions)")
                plot_confusion_matrix(cm, lab, REPORTS / f"confusion_matrix_{v}_normalized.png",
                                      f"{v}: pooled out-of-fold (row-normalised)", normalize=True)
    R = pd.DataFrame(rows)
    R.to_csv(REPORTS / "per_run_metrics.csv", index=False)
    pd.DataFrame(perclass).to_csv(REPORTS / "per_class_pooled_metrics.csv", index=False)
    agg = R.groupby(["variant", "test_subset"])[["balanced_accuracy", "accuracy", "macro_f1"]].agg(["mean", "std", "min", "max"])
    agg.to_csv(REPORTS / "summary_mean_std.csv")
    R.groupby(["variant", "test_subset", "fold"])[["balanced_accuracy", "accuracy", "macro_f1"]].mean().to_csv(REPORTS / "per_fold_mean_over_seeds.csv")
    R.groupby(["variant", "test_subset", "seed"])[["balanced_accuracy", "accuracy", "macro_f1"]].mean().to_csv(REPORTS / "per_seed_mean_over_folds.csv")
    if preds["baseline"] is not None and preds["ablationA"] is not None:
        for subset in ("all_test", "non_conflict_test"):
            a = R[(R.variant == "baseline") & (R.test_subset == subset)].set_index(["fold", "seed"])
            b = R[(R.variant == "ablationA") & (R.test_subset == subset)].set_index(["fold", "seed"])
            d = (a[["balanced_accuracy", "accuracy", "macro_f1"]] - b[["balanced_accuracy", "accuracy", "macro_f1"]]).dropna()
            dfold = d.groupby(level="fold").mean()
            out["paired"][subset] = {
                "definition": "baseline (B, includes former-conflict photos in training) minus ablationA (A, excludes them); same test photos",
                "per_run_diff": d.round(4).reset_index().to_dict(orient="records"),
                "mean_diff": d.mean().to_dict(), "std_diff": d.std(ddof=1).to_dict(),
                "per_fold_mean_diff_balanced_accuracy": dfold.balanced_accuracy.round(4).to_dict(),
                "runs_B_better_bal_acc": int((d.balanced_accuracy > 0).sum()),
                "runs_A_better_bal_acc": int((d.balanced_accuracy < 0).sum()),
                "runs_equal_bal_acc": int((d.balanced_accuracy == 0).sum()),
                "folds_B_better_bal_acc": int((dfold.balanced_accuracy > 0).sum()), "runs": int(len(d))}
            same = sorted(preds["baseline"].photo_id) == sorted(preds["ablationA"].photo_id)
            out["paired"][subset]["identical_test_photographs"] = bool(same)
    (REPORTS / "stage_b_results.json").write_text(json.dumps(out, indent=2, default=str))
    print(agg.round(4).to_string())
    print(json.dumps(out["paired"], indent=1, default=str)[:3000])
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["prepare", "run", "report", "_one"])
    ap.add_argument("--variant", choices=VARIANTS, default="baseline")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    if a.command == "prepare":
        return prepare()
    if a.command == "run":
        return run_variant(a.variant)
    if a.command == "_one":
        return train_one(a.variant, a.fold, a.seed)
    return report()


if __name__ == "__main__":
    raise SystemExit(main())
