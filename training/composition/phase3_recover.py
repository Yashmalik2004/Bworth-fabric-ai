"""Phase 3: recover within-band label-conflict images for 4-band training.

Separate experiment; never writes outside experiments/phase3_recovered_data/
(fold checkpoints are temporary and deleted; the final checkpoint goes there too).

    python -m training.composition.phase3_recover audit      # A: labels + duplicate identity, B: swatch candidates
    python -m training.composition.phase3_recover build      # apply reviewed decisions, integrity checks, manifests, folds
    python -m training.composition.phase3_recover cv --run phase3_s42   # 5-fold CV for one run definition
    python -m training.composition.phase3_recover fixed      # one model on the fixed v2 split (+ recovered train data)
    python -m training.composition.phase3_recover report     # aggregate + compare with Phase 2

Audit A (automatic, per conflict cluster):
  * every original label maps to the same 4-band class,
  * members are the same photograph: one MD5, or re-encodings with every pairwise
    perceptual-hash distance <= 8/256 AND mean absolute pixel difference <= 6/255 at 64x64.
    Clusters held together only by chaining (some pair further apart) are rejected.
Audit B (manual, recorded in audit/swatch_decisions.json):
  each accepted representative is mapped to an existing Phase-2 swatch group, a new
  swatch, or rejected when its physical identity is uncertain or matches a fabric in
  another band.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageOps

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import BAND_DEFINITIONS, FINE_TO_BAND, ORIGINAL_DIR, PROCESSED_DIR  # noqa: E402
from training.composition.dataset import (  # noqa: E402
    check_no_leakage,
    clean_index,
    dhash_bits,
    index_images,
    resolve_path,
)
from utils.image_utils import load_rgb_image  # noqa: E402

EXP = PROJECT_ROOT / "experiments" / "phase3_recovered_data"
AUDIT = EXP / "audit"
MANIFESTS = EXP / "manifests"
CV_DIR = EXP / "cv_folds"
RUNS = EXP / "runs"
RESULTS = EXP / "results"
P2_FOLDS = PROCESSED_DIR / "cv" / "folds.csv"
P2_V2 = PROCESSED_DIR / "splits_group_v2.csv"
P2_CV_OUT = PROJECT_ROOT / "evaluation" / "composition" / "cv"
BANDS = list(BAND_DEFINITIONS)
N_FOLDS = 5
SEED = 42
DHASH_MAX = 8
MAD_MAX = 6.0

# Run definitions. Everything except data/seed equals the Phase-2 band4_strong configuration.
RUN_DEFS = {
    "phase3_s42": {"data": "expanded", "seed": 42},
    "phase3_s43": {"data": "expanded", "seed": 43},
    "phase2ctrl_s43": {"data": "clean", "seed": 43},   # Phase-2 recipe + data, second seed (noise estimate)
}
TRAIN_CFG = {"label_scheme": "band4", "color_aug": "strong", "num_workers": 1}


def md5(path) -> str:
    return hashlib.md5(resolve_path(path).read_bytes()).hexdigest()


def small_gray(path) -> np.ndarray:
    with Image.open(resolve_path(path)) as im:
        im = ImageOps.exif_transpose(im).convert("L").resize((64, 64), Image.BILINEAR)
        return np.asarray(im, dtype=np.float32)


# --------------------------------------------------------------------------- #
# Audit
# --------------------------------------------------------------------------- #
def audit_a() -> pd.DataFrame:
    raw = index_images(ORIGINAL_DIR)
    res = clean_index(raw)
    p2 = pd.read_csv(P2_V2)
    if set(res.clean["path"]) != set(p2["path"]):
        raise RuntimeError("INTEGRITY: cleaning no longer reproduces the Phase-2 clean set")
    rows = []
    for cid, g in res.removed_conflicts.groupby("dup_cluster"):
        g = g.sort_values("path")
        labels = sorted(g["label"].unique(), key=lambda c: int(c[:-6]))
        bands = sorted({FINE_TO_BAND[c] for c in labels})
        md5s = [md5(p) for p in g["path"]]
        hashes = [dhash_bits(resolve_path(p)) for p in g["path"]]
        grays = [small_gray(p) for p in g["path"]]
        pairs = list(combinations(range(len(g)), 2))
        max_dh = max(int((hashes[i] != hashes[j]).sum()) for i, j in pairs)
        max_mad = max(float(np.abs(grays[i] - grays[j]).mean()) for i, j in pairs)
        n_md5 = len(set(md5s))
        if len(bands) > 1:
            status, reason = "rejected", f"labels span bands {bands}"
        elif n_md5 == 1:
            status, reason = "accepted_A", "all copies byte-identical"
        elif max_dh <= DHASH_MAX and max_mad <= MAD_MAX:
            status, reason = "accepted_A", f"re-encoded copies of one photo (max dHash {max_dh}, max MAD {max_mad:.1f})"
        else:
            status, reason = "rejected", (f"not provably one photograph: max pairwise dHash {max_dh}/256, "
                                          f"MAD {max_mad:.1f} (cluster joined by chaining)")
        rows.append({
            "cluster_id": int(cid), "n_files": len(g), "labels": "|".join(labels),
            "band": bands[0] if len(bands) == 1 else "|".join(bands), "n_bands": len(bands),
            "n_unique_md5": n_md5, "max_pair_dhash": max_dh, "max_pair_mad": round(max_mad, 2),
            "representative": g["path"].iloc[0], "files": "|".join(g["path"]),
            "folder_labels": "|".join(g["label"]), "audit_a": status, "audit_a_reason": reason,
        })
    return pd.DataFrame(rows)


def audit_b_candidates(a: pd.DataFrame) -> pd.DataFrame:
    """Nearest clean images / groups (all bands) and recovered-recovered clusters, for manual review."""
    from sklearn.cluster import AgglomerativeClustering

    from training.composition.dataset import extract_embeddings

    acc = a[a.audit_a == "accepted_A"].reset_index(drop=True)
    clean = pd.read_csv(P2_FOLDS)
    emb_c = extract_embeddings(clean["path"].tolist())
    emb_r = extract_embeddings(acc["representative"].tolist())
    np.save(AUDIT / "emb_clean.npy", emb_c)
    np.save(AUDIT / "emb_recovered.npy", emb_r)
    clean["band"] = clean["label"].map(FINE_TO_BAND)

    # recovered-only pre-clusters within each band (same method/threshold as Phase-1 grouping)
    acc["rc_cluster"] = ""
    for band, idx in acc.groupby("band").indices.items():
        if len(idx) == 1:
            lab = np.array([0])
        else:
            lab = AgglomerativeClustering(n_clusters=None, metric="cosine", linkage="average",
                                          distance_threshold=0.35).fit_predict(emb_r[idx])
        acc.loc[idx, "rc_cluster"] = [f"R_{band}_{k:02d}" for k in lab]

    S = emb_r @ emb_c.T
    groups = clean["group"].to_numpy()
    out = []
    for i, r in acc.iterrows():
        # best similarity per clean group
        gs = pd.Series(S[i]).groupby(groups).max().sort_values(ascending=False)
        top = gs.head(3)
        out.append({
            "cluster_id": r.cluster_id, "rc_cluster": r.rc_cluster,
            "nn_group_1": top.index[0], "nn_sim_1": round(float(top.iloc[0]), 3),
            "nn_band_1": clean.loc[clean.group == top.index[0], "band"].iloc[0],
            "nn_group_2": top.index[1], "nn_sim_2": round(float(top.iloc[1]), 3),
            "nn_band_2": clean.loc[clean.group == top.index[1], "band"].iloc[0],
            "nn_group_3": top.index[2], "nn_sim_3": round(float(top.iloc[2]), 3),
            "nn_band_3": clean.loc[clean.group == top.index[2], "band"].iloc[0],
        })
    return acc.merge(pd.DataFrame(out), on=["cluster_id", "rc_cluster"])


def review_sheets(cand: pd.DataFrame, per_sheet: int = 14) -> list[Path]:
    """One row per recovered pre-cluster: its members, then samples of its 2 nearest clean groups."""
    clean = pd.read_csv(P2_FOLDS)
    T = 78
    rows = []
    for rc, g in cand.groupby("rc_cluster", sort=True):
        votes = pd.concat([g.nn_group_1, g.nn_group_2]).value_counts().index[:2].tolist()
        rows.append((rc, g, votes))
    paths = []
    for s in range(0, len(rows), per_sheet):
        chunk = rows[s:s + per_sheet]
        W = Image.new("RGB", (190 + 15 * T, len(chunk) * (T + 16)), "white")
        d = ImageDraw.Draw(W)
        for r, (rc, g, votes) in enumerate(chunk):
            y = r * (T + 16)
            d.text((2, y + 20), rc.replace("_cotton", ""), fill="black")
            d.text((2, y + 36), f"n={len(g)} ids {','.join(str(c) for c in g.cluster_id[:4])}", fill="black")
            x = 190
            for p in g.representative.tolist()[:7]:
                W.paste(ImageOps.fit(load_rgb_image(resolve_path(p)), (T - 2, T - 2)), (x, y)); x += T
            x += 10
            for grp in votes:
                gp = clean[clean.group == grp]
                d.text((x, y + T + 2), f"{grp.replace('cotton', '%')[:22]} f{gp.fold.iloc[0]}", fill="blue")
                for p in gp.path.tolist()[:3]:
                    W.paste(ImageOps.fit(load_rgb_image(resolve_path(p)), (T - 2, T - 2)), (x, y)); x += T
                x += 10
        out = AUDIT / f"review_sheet_{s // per_sheet + 1:02d}.jpg"
        W.save(out, quality=85)
        paths.append(out)
    return paths


def cmd_audit() -> int:
    AUDIT.mkdir(parents=True, exist_ok=True)
    a = audit_a()
    a.to_csv(AUDIT / "audit_a_all_conflict_clusters.csv", index=False)
    summ = a.groupby(["audit_a", "audit_a_reason"]).size() if False else a["audit_a"].value_counts().to_dict()
    print(f"Conflict clusters: {len(a)} | within one band: {(a.n_bands == 1).sum()} | audit A: {summ}")
    print(a[a.n_bands == 1].groupby("audit_a")[["max_pair_dhash", "max_pair_mad"]].describe().round(2).T.to_string())
    cand = audit_b_candidates(a)
    cand.to_csv(AUDIT / "audit_b_candidates.csv", index=False)
    print(f"Accepted by A: {len(cand)} images in {cand.rc_cluster.nunique()} recovered pre-clusters")
    for p in review_sheets(cand):
        print("sheet:", p)
    return 0


# --------------------------------------------------------------------------- #
# Build (after manual decisions)
# --------------------------------------------------------------------------- #
def assign_new_groups_to_folds(new: pd.DataFrame, base: pd.DataFrame) -> dict[str, int]:
    """Greedy, deterministic: largest new groups first, each to the fold with the fewest images of its band."""
    counts = base.assign(band=base.label.map(FINE_TO_BAND)).groupby(["band", "fold"]).size()
    load = {(b, k): int(counts.get((b, k), 0)) for b in BANDS for k in range(N_FOLDS)}
    sizes = new.groupby(["group", "band"]).size().reset_index(name="n")
    rng = np.random.default_rng(SEED)
    sizes = sizes.iloc[rng.permutation(len(sizes))].sort_values("n", ascending=False, kind="stable")
    out = {}
    for r in sizes.itertuples():
        k = min(range(N_FOLDS), key=lambda f: (load[(r.band, f)], f))
        out[r.group] = k
        load[(r.band, k)] += r.n
    return out


def cmd_build() -> int:
    dec_path = AUDIT / "swatch_decisions.json"
    if not dec_path.is_file():
        print("STOP: audit/swatch_decisions.json missing — complete the manual swatch review first.")
        return 2
    decisions = json.loads(dec_path.read_text())["decisions"]
    cand = pd.read_csv(AUDIT / "audit_b_candidates.csv")
    a = pd.read_csv(AUDIT / "audit_a_all_conflict_clusters.csv")
    base = pd.read_csv(P2_FOLDS)
    v2 = pd.read_csv(P2_V2)
    group_fold = base.groupby("group").fold.first().to_dict()
    group_split = v2.groupby("group").split.first().to_dict()
    group_band = base.assign(b=base.label.map(FINE_TO_BAND)).groupby("group").b.agg(set).to_dict()
    errors = []

    # resolve per-image decision: per-cluster override first, then per-rc_cluster decision
    rc_dec = decisions.get("rc_clusters", {})
    img_dec = {int(k): v for k, v in decisions.get("clusters", {}).items()}
    rows, audit_rows = [], []
    for r in cand.itertuples():
        d = img_dec.get(r.cluster_id) or rc_dec.get(r.rc_cluster)
        if d is None:
            errors.append(f"no decision for cluster {r.cluster_id} ({r.rc_cluster})")
            continue
        action = d["action"]
        reason = d.get("note", "")
        if action == "attach":
            g = d["group"]
            if g not in group_fold:
                errors.append(f"cluster {r.cluster_id}: unknown group {g}")
                continue
            if group_band[g] != {r.band}:
                errors.append(f"cluster {r.cluster_id}: band {r.band} attached to group of band {group_band[g]}")
                continue
            rows.append({"cluster_id": r.cluster_id, "path": r.representative, "band": r.band, "labels": r.labels,
                         "group": g, "fold": group_fold[g], "v2_split": group_split[g], "origin": "recovered_attached"})
            final = "accepted"
        elif action == "new":
            g = f"P3NEW_{d.get('swatch', r.rc_cluster)}"
            rows.append({"cluster_id": r.cluster_id, "path": r.representative, "band": r.band, "labels": r.labels,
                         "group": g, "fold": -1, "v2_split": "train", "origin": "recovered_new_swatch"})
            final = "accepted"
        elif action == "reject":
            final = "rejected"
        else:
            errors.append(f"cluster {r.cluster_id}: unknown action {action}")
            continue
        audit_rows.append({"cluster_id": r.cluster_id, "decision_b": final, "action_b": action,
                           "group_b": d.get("group", ""), "reason_b": reason})
    if errors:
        print("STOP: decision errors:\n  " + "\n  ".join(errors))
        return 2
    rec = pd.DataFrame(rows)

    # fold assignment for new swatches (clean fold assignment untouched)
    new = rec[rec.origin == "recovered_new_swatch"]
    if len(new):
        fmap = assign_new_groups_to_folds(new, base)
        rec.loc[rec.origin == "recovered_new_swatch", "fold"] = rec.loc[rec.origin == "recovered_new_swatch", "group"].map(fmap)

    # ---- integrity checks -------------------------------------------------- #
    checks = {}
    clean = base.assign(band=base.label.map(FINE_TO_BAND), origin="clean", labels=base.label, cluster_id=-1)
    clean["v2_split"] = clean.group.map(group_split)
    checks["clean_rows_identical_to_phase2"] = bool(
        clean[["path", "group", "fold"]].equals(pd.read_csv(P2_FOLDS)[["path", "group", "fold"]]))
    all_df = pd.concat([clean[["path", "band", "labels", "group", "fold", "v2_split", "origin", "cluster_id"]],
                        rec[["path", "band", "labels", "group", "fold", "v2_split", "origin", "cluster_id"]]],
                       ignore_index=True)
    checks["no_duplicate_paths"] = bool(all_df.path.is_unique)
    checks["every_group_in_one_fold"] = bool(all_df.groupby("group").fold.nunique().max() == 1)
    checks["every_group_one_band"] = bool(all_df.groupby("group").band.nunique().max() == 1)
    checks["all_folds_assigned"] = bool((all_df.fold >= 0).all())
    checks["labels_consistent_with_band"] = bool(all(
        {FINE_TO_BAND[c] for c in str(lab).split("|")} == {b} for lab, b in zip(all_df["labels"], all_df.band)))
    # recovered images must not be (near-)duplicates of any clean image or of each other across groups
    H = np.stack([dhash_bits(resolve_path(p)) for p in all_df.path])
    D = (H[:, None, :] != H[None, :, :]).sum(-1)
    np.fill_diagonal(D, 999)
    gr = all_df.group.to_numpy()
    near = np.argwhere((D <= DHASH_MAX) & (gr[:, None] != gr[None, :]))
    checks["no_near_duplicates_across_groups"] = bool(len(near) == 0)
    checks["recovered_not_in_v1_or_v2_manifests"] = bool(not set(rec.path) & set(v2.path))
    if not all(checks.values()):
        print("STOP: integrity check failed:", json.dumps(checks, indent=2))
        return 2

    # ---- write manifests --------------------------------------------------- #
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    CV_DIR.mkdir(parents=True, exist_ok=True)
    all_df["label"] = [str(lab).split("|")[0] for lab in all_df["labels"]]   # any fine label of the band (band is what is used)
    all_df["label_idx"] = -1
    all_df["dup_cluster"] = np.arange(len(all_df))
    all_df.to_csv(MANIFESTS / "image_to_swatch_mapping.csv", index=False)
    # fixed split: v2 clean split unchanged; recovered follow their group, new swatches -> train
    fixed = all_df.rename(columns={"v2_split": "split"})
    fixed.to_csv(MANIFESTS / "fixed_split_expanded.csv", index=False)
    for k in range(N_FOLDS):
        for data in ("expanded", "clean"):
            m = all_df if data == "expanded" else all_df[all_df.origin == "clean"]
            m = m.copy()
            m["split"] = "train"
            m.loc[m.fold == k, "split"] = "test"
            m.loc[m.fold == (k + 1) % N_FOLDS, "split"] = "val"
            # recovered images never used for validation (keeps model selection identical in kind to Phase 2)
            m = m[~((m.split == "val") & (m.origin != "clean"))]
            check_no_leakage(m)
            m.to_csv(CV_DIR / f"{data}_fold{k}.csv", index=False)

    # complete accepted/rejected CSV
    full = a.merge(pd.DataFrame(audit_rows), on="cluster_id", how="left").merge(
        rec[["cluster_id", "group", "fold"]], on="cluster_id", how="left")
    full["final_decision"] = np.where(full.audit_a != "accepted_A", "rejected",
                                      full.decision_b.fillna("rejected"))
    full["final_reason"] = np.where(full.audit_a != "accepted_A", full.audit_a_reason,
                                    np.where(full.final_decision == "rejected", full.reason_b,
                                             full.audit_a_reason + "; " + full.reason_b))
    full.to_csv(AUDIT / "recoverable_groups_audit.csv", index=False)

    summary = {
        "integrity_checks": checks,
        "conflict_clusters": int(len(a)),
        "within_one_band": int((a.n_bands == 1).sum()),
        "accepted_audit_a": int((a.audit_a == "accepted_A").sum()),
        "accepted_final": int((full.final_decision == "accepted").sum()),
        "rejected_final": int((full.final_decision == "rejected").sum()),
        "recovered_images": int(len(rec)),
        "recovered_attached_to_existing_swatch": int((rec.origin == "recovered_attached").sum()),
        "recovered_new_swatch_images": int((rec.origin == "recovered_new_swatch").sum()),
        "new_swatch_groups": int(rec.loc[rec.origin == "recovered_new_swatch", "group"].nunique()),
        "recovered_per_band": rec.band.value_counts().to_dict(),
        "recovered_per_fold": rec.fold.value_counts().sort_index().to_dict(),
        "images_total_expanded": int(len(all_df)),
        "images_per_band_expanded": all_df.band.value_counts().to_dict(),
        "fixed_split_counts": fixed.groupby(["split", "origin"]).size().to_dict(),
        "seed": SEED, "dhash_max": DHASH_MAX, "mad_max": MAD_MAX,
    }
    (MANIFESTS / "build_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))
    return 0


# --------------------------------------------------------------------------- #
# Training / CV
# --------------------------------------------------------------------------- #
def run_one(run: str, k: int, device_arg: str | None) -> int:
    import torch

    from configs.config import CompositionConfig
    from training.composition.dataset import apply_label_scheme
    from training.composition.evaluate import predict_frame
    from training.composition.model import load_model_from_checkpoint
    from training.composition.train import train
    from utils.device import get_device

    rd = RUN_DEFS[run]
    out = RUNS / run / f"fold{k}"
    if (out / "predictions_test.csv").is_file():
        print(f"[skip] {run} fold {k}")
        return 0
    tmp = RUNS / run / "tmp_fold_model.pth"
    cfg = CompositionConfig(**TRAIN_CFG, seed=rd["seed"], splits_path=str(CV_DIR / f"{rd['data']}_fold{k}.csv"),
                            run_name=f"{run}/fold{k}", checkpoint_path=str(tmp), output_dir=str(RUNS))
    device = get_device(device_arg)
    train(cfg, device)
    ck = torch.load(tmp, map_location="cpu", weights_only=False)
    classes = [ck["idx_to_class"][i] for i in range(len(ck["idx_to_class"]))]
    model = load_model_from_checkpoint(ck, device)
    df = apply_label_scheme(pd.read_csv(cfg.splits_path), "band4", classes)
    test = df[df.split == "test"].reset_index(drop=True)
    probs = predict_frame(model, test, ck["preprocessing"], device)
    pred = test[["path", "label", "group", "origin"]].copy()
    pred["fold"] = k
    pred["true"] = [classes[i] for i in test["target"]]
    pred["predicted"] = [classes[i] for i in probs.argmax(1)]
    for j, c in enumerate(classes):
        pred[f"p_{c}"] = probs[:, j]
    pred.to_csv(out / "predictions_test.csv", index=False)
    (out / "fold_info.json").write_text(json.dumps({"best_epoch": ck["best_epoch"], "val_acc": ck["val_acc"],
                                                    "train_acc_at_best": ck["train_acc"], "seed": rd["seed"],
                                                    "data": rd["data"], "split_file": cfg.splits_path,
                                                    "n_train": int((df.split == 'train').sum())}, indent=2))
    tmp.unlink(missing_ok=True)
    return 0


def cmd_cv(runs: list[str], device_arg: str | None) -> int:
    for k in range(N_FOLDS):
        for run in runs:
            if (RUNS / run / f"fold{k}" / "predictions_test.csv").is_file():
                print(f"[skip] {run} fold {k}", flush=True)
                continue
            cmd = [sys.executable, "-m", "training.composition.phase3_recover", "_fold", "--run", run, "--fold", str(k)]
            if device_arg:
                cmd += ["--device", device_arg]
            rc = subprocess.call(cmd, cwd=PROJECT_ROOT)
            if rc != 0:
                print(f"[fail] {run} fold {k} exit {rc}")
                return rc
    return 0


def cmd_fixed(device_arg: str | None) -> int:
    from configs.config import CompositionConfig
    from training.composition.evaluate import evaluate_checkpoint
    from training.composition.train import train
    from utils.device import get_device

    ckpt = EXP / "models" / "band4_strong_recovered.pth"
    if ckpt.exists():
        print(f"STOP: {ckpt} exists; not overwriting.")
        return 2
    cfg = CompositionConfig(**TRAIN_CFG, seed=SEED, splits_path=str(MANIFESTS / "fixed_split_expanded.csv"),
                            run_name="fixed_split/band4_strong_recovered", checkpoint_path=str(ckpt),
                            output_dir=str(RUNS))
    device = get_device(device_arg)
    train(cfg, device)
    # evaluate on the clean v2 test split only (identical to Phase 2's band4_strong_v2 evaluation)
    evaluate_checkpoint(ckpt, device, splits_path=str(P2_V2), output_name="fixed_split/eval_on_v2_clean_test")
    return 0


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def _metrics(pred: pd.DataFrame) -> dict:
    from training.composition.evaluate import compute_metrics

    y = pred["true"].map({b: i for i, b in enumerate(BANDS)}).to_numpy()
    p = pred[[f"p_{b}" for b in BANDS]].to_numpy()
    return compute_metrics(y, p, BANDS)


def load_run(run: str) -> pd.DataFrame | None:
    if run == "phase2_s42":
        base = P2_CV_OUT / "band4_strong"
        parts = [pd.read_csv(p).assign(origin="clean") for p in sorted(base.glob("fold*/predictions_test.csv"))]
    else:
        parts = [pd.read_csv(p) for p in sorted((RUNS / run).glob("fold*/predictions_test.csv"))]
    return pd.concat(parts, ignore_index=True) if parts else None


def cmd_report() -> int:
    from utils.plotting import plot_confusion_matrix
    from training.composition.evaluate import short_label

    RESULTS.mkdir(parents=True, exist_ok=True)
    runs = ["phase2_s42", "phase2ctrl_s43", "phase3_s42", "phase3_s43"]
    rows, pooled, perband = [], {}, []
    for run in runs:
        pred = load_run(run)
        if pred is None:
            continue
        pred["band_true"] = pred["true"]
        for subset in ("clean", "recovered"):
            sp = pred[pred.origin == "clean"] if subset == "clean" else pred[pred.origin != "clean"]
            if sp.empty:
                continue
            for k, fp in sp.groupby("fold"):
                m = _metrics(fp)
                rows.append({"run": run, "test_subset": subset, "fold": int(k), "n": len(fp),
                             "balanced_accuracy": m["balanced_accuracy"], "accuracy": m["accuracy"],
                             "macro_f1": m["macro_f1"]})
            m = _metrics(sp)
            pooled[f"{run}|{subset}"] = {k: m[k] for k in ("n_images", "balanced_accuracy", "accuracy", "macro_f1",
                                                           "confusion_matrix")}
            for b, v in m["per_class"].items():
                perband.append({"run": run, "test_subset": subset, "band": b, **v})
            if subset == "clean":
                plot_confusion_matrix(np.array(m["confusion_matrix"]), [short_label(b) for b in BANDS],
                                      RESULTS / f"confusion_matrix_{run}_clean_test.png",
                                      f"{run}: pooled out-of-fold, clean test images (n={len(sp)})")
                plot_confusion_matrix(np.array(m["confusion_matrix"]), [short_label(b) for b in BANDS],
                                      RESULTS / f"confusion_matrix_{run}_clean_test_normalized.png",
                                      f"{run}: clean test images (row-normalised)", normalize=True)
    pf = pd.DataFrame(rows)
    pf.to_csv(RESULTS / "per_fold_metrics.csv", index=False)
    pd.DataFrame(perband).to_csv(RESULTS / "per_band_metrics.csv", index=False)

    # same test images check + paired comparisons on clean test images
    p2 = load_run("phase2_s42")
    summary = {"pooled": pooled, "fold_summary": {}, "paired": {}}
    for run in pf.run.unique():
        f = pf[(pf.run == run) & (pf.test_subset == "clean")]
        summary["fold_summary"][run] = {m: {"mean": float(f[m].mean()), "std": float(f[m].std(ddof=1)),
                                            "min": float(f[m].min()), "max": float(f[m].max())}
                                        for m in ("balanced_accuracy", "accuracy", "macro_f1")}
    for a_run, b_run in [("phase3_s42", "phase2_s42"), ("phase3_s43", "phase2ctrl_s43"),
                         ("phase2ctrl_s43", "phase2_s42"), ("phase3_s43", "phase3_s42")]:
        A, B = load_run(a_run), load_run(b_run)
        if A is None or B is None:
            continue
        A, B = A[A.origin == "clean"], B[B.origin == "clean"]
        same = sorted(A.path) == sorted(B.path)
        fa = pf[(pf.run == a_run) & (pf.test_subset == "clean")].set_index("fold")
        fb = pf[(pf.run == b_run) & (pf.test_subset == "clean")].set_index("fold")
        d = (fa.balanced_accuracy - fb.balanced_accuracy).dropna()
        summary["paired"][f"{a_run} - {b_run}"] = {
            "identical_test_images": same, "metric": "balanced_accuracy (clean test images)",
            "mean_diff": float(d.mean()), "per_fold": d.round(4).tolist(), "folds_better": int((d > 0).sum()),
            "folds": int(len(d))}
    (RESULTS / "phase3_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(pf.round(4).to_string())
    print(json.dumps({k: v for k, v in summary.items() if k != "pooled"}, indent=2, default=str))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["audit", "build", "cv", "fixed", "report", "_fold"])
    ap.add_argument("--run", nargs="*", default=list(RUN_DEFS))
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    if args.command == "audit":
        return cmd_audit()
    if args.command == "build":
        return cmd_build()
    if args.command == "cv":
        return cmd_cv(args.run, args.device)
    if args.command == "_fold":
        return run_one(args.run[0], args.fold, args.device)
    if args.command == "fixed":
        return cmd_fixed(args.device)
    return cmd_report()


if __name__ == "__main__":
    raise SystemExit(main())
