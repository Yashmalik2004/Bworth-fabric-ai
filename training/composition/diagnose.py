"""Read-only diagnostic audit of a trained composition checkpoint.

Never trains, never writes model weights, never changes labels or splits.

    python -m training.composition.diagnose                 # full audit
    python -m training.composition.diagnose lookup IMG ...  # split membership of specific images

Full audit outputs go to evaluation/composition/diagnostics/:
    membership_original.csv      every file in Original/: split / duplicate-of / excluded + prediction
    per_class_metrics_<split>.csv  counts, correct, P/R/F1, most-confused class
    confusion_matrix_<split>.png   train / val / test, same checkpoint, eval preprocessing
    per_group_test.csv           accuracy per swatch group in the test split
    nearest_neighbour_test.csv   nearest training image (fine-tuned features) for each test image
    inference_parity.json        direct PyTorch vs CompositionPredictor (Streamlit path) vs evaluate.py
    diagnostics_summary.json     headline numbers used in the report
    sheets/*.jpg                 contact sheets of selected classes (train swatches vs test predictions)
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageOps
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import DEFAULT_CHECKPOINT, EVALUATION_DIR, ORIGINAL_DIR, cotton_label  # noqa: E402
from training.composition.dataset import (  # noqa: E402
    clean_index,
    dhash_bits,
    index_images,
    resolve_path,
    to_relative,
)
from training.composition.model import load_model_from_checkpoint  # noqa: E402
from training.composition.transforms import build_eval_transform  # noqa: E402
from utils.device import get_device  # noqa: E402
from utils.image_utils import load_rgb_image  # noqa: E402
from utils.plotting import plot_confusion_matrix  # noqa: E402

OUT_DIR = EVALUATION_DIR / "diagnostics"


def md5(path: Path) -> str:
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


# --------------------------------------------------------------------------- #
# Split membership for every Original/ file
# --------------------------------------------------------------------------- #
def build_membership(splits: pd.DataFrame) -> pd.DataFrame:
    """Re-run the (deterministic) cleaning step and map every raw file to its fate."""
    raw = index_images(ORIGINAL_DIR)
    res = clean_index(raw)
    kept = res.clean.set_index("dup_cluster")["path"].to_dict()
    split_of = splits.set_index("path")["split"].to_dict()
    group_of = splits.set_index("path")["group"].to_dict()
    missing = set(res.clean["path"]) ^ set(splits["path"])
    if missing:
        raise RuntimeError(f"Cleaning is not reproducible against the saved manifest ({len(missing)} paths differ)")

    rows = []
    for part, kind in ((res.clean, "kept"), (res.removed_duplicates, "dup"), (res.removed_conflicts, "conflict")):
        for r in part.itertuples():
            if kind == "conflict":
                labels = sorted(set(res.removed_conflicts.loc[res.removed_conflicts.dup_cluster == r.dup_cluster, "label"]))
                rows.append({"path": r.path, "folder_label": r.label, "status": "excluded_label_conflict",
                             "split": "none", "representative": "", "group": "",
                             "conflicting_labels": "|".join(labels)})
            else:
                rep = kept[r.dup_cluster]
                rows.append({"path": r.path, "folder_label": r.label,
                             "status": "in_manifest" if kind == "kept" else "duplicate_of_manifest_image",
                             "split": split_of[rep], "representative": rep if kind == "dup" else "",
                             "group": group_of[rep], "conflicting_labels": ""})
    return pd.DataFrame(rows).sort_values("path", ignore_index=True)


# --------------------------------------------------------------------------- #
# Batched fp32 inference with penultimate features
# --------------------------------------------------------------------------- #
@torch.inference_mode()
def infer(model, paths: list[str], preprocessing: dict, device, batch_size: int = 16):
    tf = build_eval_transform(preprocessing)
    feats_store = {}
    hook = model.avgpool.register_forward_hook(lambda m, i, o: feats_store.__setitem__("f", o.flatten(1)))
    probs, feats = [], []
    try:
        for i in range(0, len(paths), batch_size):
            x = torch.stack([tf(load_rgb_image(resolve_path(p))) for p in paths[i:i + batch_size]]).to(device)
            probs.append(torch.softmax(model(x).float(), 1).cpu().numpy())
            feats.append(torch.nn.functional.normalize(feats_store["f"].float(), dim=1).cpu().numpy())
    finally:
        hook.remove()
    return np.concatenate(probs), np.concatenate(feats)


def per_class_table(y_true, y_pred, classes) -> pd.DataFrame:
    labels = list(range(len(classes)))
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    rows = []
    for i, c in enumerate(classes):
        off = cm[i].copy()
        off[i] = 0
        j = int(off.argmax())
        rows.append({
            "class": cotton_label(c), "n": int(s[i]), "correct": int(cm[i, i]), "incorrect": int(s[i] - cm[i, i]),
            "accuracy": float(cm[i, i] / s[i]) if s[i] else np.nan,
            "precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]),
            "times_predicted": int(cm[:, i].sum()),
            "most_confused_with": f"{cotton_label(classes[j])} ({int(off[j])})" if off[j] > 0 else "-",
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Inference-pipeline parity
# --------------------------------------------------------------------------- #
def direct_pytorch_probs(ckpt_path: Path, image_paths: list[Path], device) -> np.ndarray:
    """Independent re-implementation: no project helpers, only torch/torchvision/PIL."""
    from torch import nn
    from torchvision import models
    from torchvision import transforms as T

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    net = models.resnet50()
    net.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(2048, len(ck["class_to_idx"])))
    net.load_state_dict(ck["model_state_dict"])
    net = net.to(device).eval()
    pp = ck["preprocessing"]
    tf = T.Compose([T.Resize(pp["resize_size"], antialias=True), T.CenterCrop(pp["image_size"]),
                    T.ToTensor(), T.Normalize(pp["mean"], pp["std"])])
    out = []
    with torch.inference_mode():
        for p in image_paths:
            img = ImageOps.exif_transpose(Image.open(p)).convert("RGB")
            out.append(torch.softmax(net(tf(img).unsqueeze(0).to(device)).float(), 1)[0].cpu().numpy())
    return np.stack(out)


def predictor_probs(ckpt_path: Path, image_paths: list[Path], device) -> np.ndarray:
    """Exactly what the Streamlit app does: raw bytes -> load_rgb_image -> CompositionPredictor.predict."""
    from inference.composition.predict import CompositionPredictor

    pred = CompositionPredictor(ckpt_path, device=device)
    out = []
    for p in image_paths:
        img = load_rgb_image(Path(p).read_bytes())          # app: image = load_rgb_image(source bytes)
        r = pred.predict(img)                                 # app: predictor.predict(image)
        out.append(np.array([r.probabilities[c] for c in pred.classes]))
    return np.stack(out)


def streamlit_app_probs(sample_indices: list[int]) -> list[dict]:
    """Drive the real app script headlessly and read the probability table it renders."""
    from streamlit.testing.v1 import AppTest

    results = []
    for idx in sample_indices:
        at = AppTest.from_file(str(PROJECT_ROOT / "app" / "streamlit_app.py"), default_timeout=180).run()
        at.selectbox[0].set_value(idx).run()
        at.button[0].click().run()
        if at.exception:
            raise RuntimeError(f"App raised: {at.exception}")
        table = at.dataframe[0].value
        results.append({"index": idx, "table": dict(zip(table["Category"], table["Probability (%)"]))})
    return results


# --------------------------------------------------------------------------- #
# Contact sheets
# --------------------------------------------------------------------------- #
def contact_sheet(rows: list[tuple[str, list[tuple[str, str, str]]]], out: Path, tile: int = 120) -> None:
    """rows: [(row_title, [(path, caption, colour), ...]), ...]"""
    ncol = max(len(items) for _, items in rows)
    W = Image.new("RGB", (tile * ncol + 170, (tile + 18) * len(rows)), "white")
    d = ImageDraw.Draw(W)
    for r, (title, items) in enumerate(rows):
        y = r * (tile + 18)
        d.text((4, y + tile // 2), title, fill="black")
        for c, (p, cap, col) in enumerate(items):
            im = load_rgb_image(resolve_path(p))
            im = ImageOps.fit(im, (tile - 4, tile - 4))
            W.paste(im, (170 + c * tile, y))
            d.text((170 + c * tile, y + tile - 2), cap, fill=col)
    W.save(out, quality=90)


# --------------------------------------------------------------------------- #
# Full audit
# --------------------------------------------------------------------------- #
def run_audit(ckpt_path: Path, device, app_samples: int) -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "sheets").mkdir(exist_ok=True)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    classes = [ck["idx_to_class"][i] for i in range(len(ck["idx_to_class"]))]
    cidx = {c: i for i, c in enumerate(classes)}
    splits = pd.read_csv(resolve_path(ck["config"]["splits_path"]))
    model = load_model_from_checkpoint(ck, device)
    summary: dict = {"checkpoint": to_relative(ckpt_path), "checkpoint_md5": md5(ckpt_path),
                     "splits_md5": md5(resolve_path(ck["config"]["splits_path"])), "device": str(device),
                     "precision": "fp32 (no autocast)"}

    # 1. membership + predictions for every Original/ file
    mem = build_membership(splits)
    print(f"Running inference on {len(mem)} Original/ images ...", flush=True)
    probs_all, feats_all = infer(model, mem["path"].tolist(), ck["preprocessing"], device)
    mem["predicted"] = [classes[i] for i in probs_all.argmax(1)]
    mem["score"] = probs_all.max(1)
    mem["p_folder_label"] = [probs_all[k, cidx[c]] for k, c in enumerate(mem["folder_label"])]
    mem["correct_vs_folder_label"] = mem["predicted"] == mem["folder_label"]
    mem.to_csv(OUT_DIR / "membership_original.csv", index=False)
    row_of = {p: k for k, p in enumerate(mem["path"])}

    status_key = mem["status"].where(mem["status"] == "excluded_label_conflict", mem["status"] + ":" + mem["split"])
    acc_by_status = mem.groupby(status_key)["correct_vs_folder_label"].agg(["size", "mean"])
    summary["original_folder_accuracy_by_membership"] = {
        k: {"n": int(v["size"]), "accuracy": round(float(v["mean"]), 4)} for k, v in acc_by_status.iterrows()}
    summary["original_membership_per_class"] = (
        pd.crosstab(mem["folder_label"], status_key).reindex(classes).to_dict(orient="index"))

    # 2/5. per-class metrics + confusion matrices on train / val / test
    short = [c.replace("cotton", "%") for c in classes]
    split_res = {}
    for split in ("train", "val", "test"):
        f = splits[splits.split == split].reset_index(drop=True)
        k = [row_of[p] for p in f["path"]]
        y = f["label_idx"].to_numpy()
        yp = probs_all[k].argmax(1)
        tab = per_class_table(y, yp, classes)
        tab.to_csv(OUT_DIR / f"per_class_metrics_{split}.csv", index=False)
        cm = confusion_matrix(y, yp, labels=list(range(len(classes))))
        np.savetxt(OUT_DIR / f"confusion_matrix_{split}.csv", cm, fmt="%d", delimiter=",")
        plot_confusion_matrix(cm, short, OUT_DIR / f"confusion_matrix_{split}.png",
                              f"{split.capitalize()} confusion matrix (counts, n={len(f)})")
        plot_confusion_matrix(cm, short, OUT_DIR / f"confusion_matrix_{split}_normalized.png",
                              f"{split.capitalize()} confusion matrix (row-normalised)", normalize=True)
        split_res[split] = {"frame": f, "rows": k, "y": y, "yp": yp, "table": tab}
        summary[f"{split}_accuracy"] = round(float((yp == y).mean()), 4)
        summary[f"{split}_n"] = int(len(f))
    summary["per_class_train_accuracy"] = dict(zip(split_res["train"]["table"]["class"],
                                                   split_res["train"]["table"]["accuracy"].round(4)))

    # class weights exactly as train.py computed them
    counts = np.bincount(split_res["train"]["y"], minlength=len(classes)).astype(float)
    w = counts.sum() / (len(classes) * counts)
    summary["class_weights_used_in_training"] = {cotton_label(c): round(float(x), 3) for c, x in zip(classes, w)}
    summary["train_counts"] = {cotton_label(c): int(x) for c, x in zip(classes, counts)}

    # per-group (swatch) accuracy on test
    te = split_res["test"]
    tf_ = te["frame"].assign(predicted=[classes[i] for i in te["yp"]], correct=te["yp"] == te["y"],
                             score=probs_all[te["rows"]].max(1))
    grp = (tf_.groupby(["label", "group"]).agg(n=("correct", "size"), correct=("correct", "sum"),
                                                predictions=("predicted", lambda s: dict(Counter(s))))
           .reset_index())
    grp["accuracy"] = grp["correct"] / grp["n"]
    grp.to_csv(OUT_DIR / "per_group_test.csv", index=False)

    # nearest training neighbour in fine-tuned feature space
    tr = split_res["train"]
    Ftr, Fte = feats_all[tr["rows"]], feats_all[te["rows"]]
    sim = Fte @ Ftr.T
    nn_i = sim.argmax(1)
    nn = tf_[["path", "label", "group", "predicted", "correct"]].copy()
    nn["nn_train_path"] = tr["frame"]["path"].to_numpy()[nn_i]
    nn["nn_train_label"] = tr["frame"]["label"].to_numpy()[nn_i]
    nn["nn_similarity"] = sim.max(1)
    same = np.array([[tr["frame"]["label"].iloc[j] == lab for j in range(len(tr["frame"]))] for lab in nn["label"]])
    nn["best_same_class_similarity"] = np.where(same, sim, -1).max(1)
    nn.to_csv(OUT_DIR / "nearest_neighbour_test.csv", index=False)
    # same thing for val→train as reference, and train→train (leave-one-out) as a ceiling
    Str = Ftr @ Ftr.T
    np.fill_diagonal(Str, -1)
    summary["nn_similarity_mean"] = {
        "test_to_train": round(float(nn["nn_similarity"].mean()), 4),
        "train_to_other_train": round(float(Str.max(1).mean()), 4),
    }
    summary["nn1_label_agreement_test"] = round(float((nn["nn_train_label"] == nn["label"]).mean()), 4)
    per_cls_nn = nn.groupby("label").agg(
        nn_sim=("nn_similarity", "mean"), same_class_sim=("best_same_class_similarity", "mean"),
        nn_label_match=("nn_train_label", lambda s: 0.0), acc=("correct", "mean"))
    per_cls_nn["nn_label_match"] = nn.assign(m=nn.nn_train_label == nn.label).groupby("label")["m"].mean()
    summary["per_class_nn"] = per_cls_nn.round(4).reindex(classes).to_dict(orient="index")

    # prediction distribution on test (sink classes)
    summary["test_prediction_counts"] = {cotton_label(c): int((te["yp"] == i).sum()) for i, c in enumerate(classes)}
    summary["test_true_counts"] = {cotton_label(c): int((te["y"] == i).sum()) for i, c in enumerate(classes)}
    # probability the model assigns to the true class, test
    p_true = probs_all[te["rows"], te["y"]]
    summary["test_mean_p_true_by_class"] = (pd.Series(p_true).groupby(te["frame"]["label"]).mean()
                                            .round(4).reindex(classes).to_dict())
    summary["test_mean_score_correct_vs_wrong"] = {
        "correct": round(float(tf_.loc[tf_.correct, "score"].mean()), 4),
        "wrong": round(float(tf_.loc[~tf_.correct, "score"].mean()), 4)}

    # colour statistics per class/split (mean HSV)
    def hsv_stats(paths):
        out = []
        for p in paths:
            im = load_rgb_image(resolve_path(p)).resize((64, 64)).convert("HSV")
            a = np.asarray(im, dtype=float) / 255
            out.append(a.reshape(-1, 3).mean(0))
        return np.array(out)
    colour = {}
    for c in ("30cotton", "80cotton", "99cotton", "58cotton", "63cotton", "40cotton", "53cotton"):
        colour[c] = {}
        for s in ("train", "test"):
            f = splits[(splits.split == s) & (splits.label == c)]
            h = hsv_stats(f["path"])
            colour[c][s] = {"saturation_mean": round(float(h[:, 1].mean()), 3),
                            "value_mean": round(float(h[:, 2].mean()), 3), "n": len(f)}
    summary["colour_stats"] = colour

    # contact sheets
    for c in ("30cotton", "80cotton", "99cotton", "63cotton"):
        rows = []
        for s in ("train", "val"):
            f = splits[(splits.split == s) & (splits.label == c)]
            for g, gf in f.groupby("group"):
                rows.append((f"{s} {g[-3:]} n={len(gf)}", [(p, "", "black") for p in gf["path"][:10]]))
        ft = tf_[tf_.label == c]
        for g, gf in ft.groupby("group"):
            rows.append((f"TEST {g[-3:]} n={len(gf)}",
                         [(r.path, f"-> {r.predicted.replace('cotton', '%')}", "green" if r.correct else "red")
                          for r in gf.itertuples()][:10]))
        contact_sheet(rows, OUT_DIR / "sheets" / f"{c}.jpg")

    # 6. inference parity
    print("Checking inference parity ...", flush=True)
    parity: dict = {}
    sel = te["frame"].groupby("label").head(2)["path"].tolist()  # 26 test images, 2 per class
    sel_paths = [resolve_path(p) for p in sel]
    a = direct_pytorch_probs(ckpt_path, sel_paths, device)
    b = predictor_probs(ckpt_path, sel_paths, device)
    b_cpu = predictor_probs(ckpt_path, sel_paths, torch.device("cpu"))
    c_batch = probs_all[[row_of[p] for p in sel]]
    parity["n_images"] = len(sel)
    parity["direct_vs_predictor_max_abs_diff"] = float(np.abs(a - b).max())
    parity["direct_vs_predictor_same_argmax"] = bool((a.argmax(1) == b.argmax(1)).all())
    parity["predictor_gpu_vs_cpu_max_abs_diff"] = float(np.abs(b - b_cpu).max())
    parity["predictor_gpu_vs_cpu_same_argmax"] = bool((b.argmax(1) == b_cpu.argmax(1)).all())
    parity["batched_fp32_vs_predictor_max_abs_diff"] = float(np.abs(c_batch - b).max())

    # evaluate.py path (batched, fp16 autocast on CUDA) on the full test split
    from training.composition.evaluate import predict_frame
    amp_probs = predict_frame(model, te["frame"], ck["preprocessing"], device)
    fp32 = probs_all[te["rows"]]
    parity["evaluate_amp_vs_fp32_test_max_abs_diff"] = float(np.abs(amp_probs - fp32).max())
    parity["evaluate_amp_vs_fp32_test_argmax_disagreements"] = int((amp_probs.argmax(1) != fp32.argmax(1)).sum())
    parity["test_accuracy_fp32"] = float((fp32.argmax(1) == te["y"]).mean())
    parity["test_accuracy_evaluate_py_amp"] = float((amp_probs.argmax(1) == te["y"]).mean())

    if app_samples > 0:
        samples = splits[splits.split == "test"].reset_index(drop=True)  # same order the app lists
        idxs = list(np.linspace(0, len(samples) - 1, app_samples).round().astype(int))
        app = streamlit_app_probs(idxs)
        bp = predictor_probs(ckpt_path, [resolve_path(samples.path[i]) for i in idxs], device)
        diffs, same_top = [], []
        pred = None
        for k, res in enumerate(app):
            app_vec = np.array([res["table"][cotton_label(c)] for c in classes]) / 100
            diffs.append(float(np.abs(app_vec - bp[k]).max()))
            same_top.append(int(app_vec.argmax()) == int(bp[k].argmax()))
        parity["streamlit_app_samples"] = [samples.path[i] for i in idxs]
        parity["streamlit_vs_predictor_max_abs_diff"] = max(diffs)
        parity["streamlit_vs_predictor_note"] = "app table rounds to 0.01 %, so diffs <= 5e-5 mean identical"
        parity["streamlit_vs_predictor_same_argmax"] = all(same_top)
    (OUT_DIR / "inference_parity.json").write_text(json.dumps(parity, indent=2))
    summary["inference_parity"] = parity

    summary["checkpoint_md5_after"] = md5(ckpt_path)
    (OUT_DIR / "diagnostics_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"Diagnostics written to {OUT_DIR}")
    return summary


# --------------------------------------------------------------------------- #
# Lookup: split membership of arbitrary images
# --------------------------------------------------------------------------- #
def lookup(paths: list[str]) -> None:
    mem_path = OUT_DIR / "membership_original.csv"
    if not mem_path.is_file():
        raise SystemExit("Run the full audit first: python -m training.composition.diagnose")
    mem = pd.read_csv(mem_path).fillna("")
    mem["md5"] = [md5(resolve_path(p)) for p in mem["path"]]
    hashes = np.stack([dhash_bits(resolve_path(p)) for p in mem["path"]])
    for q in paths:
        qp = Path(q)
        if not qp.is_file():
            print(f"{q}: file not found")
            continue
        hit = mem[mem.md5 == md5(qp)]
        how = "exact bytes"
        if hit.empty:
            try:
                bits = dhash_bits(qp)
            except Exception:  # noqa: BLE001
                print(f"{q}: not a readable image")
                continue
            d = (hashes != bits).sum(1)
            if d.min() <= 8:
                hit, how = mem.iloc[[int(d.argmin())]], f"perceptual match (distance {int(d.min())}/256)"
        if hit.empty:
            print(f"{q}: NOT in the dataset (unseen image)")
            continue
        for r in hit.itertuples():
            extra = f", conflicting labels {r.conflicting_labels}" if r.conflicting_labels else ""
            print(f"{q}: [{how}] {r.path} | folder {r.folder_label} | {r.status} | split={r.split}{extra} "
                  f"| model predicts {r.predicted} ({r.score:.2f})")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", nargs="?", default="audit", choices=["audit", "lookup"])
    ap.add_argument("images", nargs="*")
    ap.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    ap.add_argument("--device", default=None)
    ap.add_argument("--app-samples", type=int, default=4, help="test samples to run through the real app (0=skip)")
    args = ap.parse_args()
    if args.command == "lookup":
        lookup(args.images)
        return 0
    run_audit(resolve_path(args.checkpoint), get_device(args.device), args.app_samples)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
