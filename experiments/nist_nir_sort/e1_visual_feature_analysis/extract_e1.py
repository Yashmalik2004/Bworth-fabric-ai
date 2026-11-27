"""NIST E1, step 1: validate the image -> swatch mapping, then extract frozen ImageNet embeddings
(ResNet-50, EfficientNet-B0) and colour-only features. EXPLORATORY ONLY: no training, no classifier.

Reads the NIST colour TIFFs read-only. Pretrained weights are loaded from the local torch hub cache only.
Writes only into this E1 directory.

    python experiments/nist_nir_sort/e1_visual_feature_analysis/extract_e1.py [--device cuda|cpu] [--batch 8]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
NIST = PROJECT / "datasets" / "external" / "NIST_NIR_SORT" / "raw_download" / "NIST_SORT_DATA"
INVENTORY = HERE.parent / "nist_composition_inventory.csv"
NAME = re.compile(r"^([A-D]\d{2}_(?:[a-ix]\d{3}){3}_\d{2}_\d{2}_)T_(\d{2})C\.tiff$")
VIEWS = {"01": ("front", "low", 34.3), "02": ("front", "medium", 47.4), "03": ("front", "high", 76.2),
         "04": ("back", "low", 34.3), "05": ("back", "medium", 47.4), "06": ("back", "high", 76.2)}
META_COLS = ["source", "cotton_percentage", "other_fibre_composition", "evidence_category", "bworth_band",
             "group_strict"]


# ---------------------------------------------------------------- phase 1: validation
def validate() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    inv = pd.read_csv(INVENTORY, dtype=str, keep_default_na=False)
    files = sorted(p for p in NIST.rglob("*") if p.is_file() and p.suffix.lower() in (".tif", ".tiff"))
    rows, problems = [], []
    for p in files:
        m = NAME.match(p.name)
        if not m:
            problems.append(f"unparsed filename: {p.name}"); continue
        sid, rep = m.group(1), m.group(2)
        folder = next((q for q in p.parts if q.startswith("Source_")), "?")
        if folder[-1] != sid[0]:
            problems.append(f"{p.name}: in {folder}")
        side, mag, scale = VIEWS[rep]
        rows.append({"image_id": p.stem, "swatch_id": sid, "replicate": rep, "side": side, "magnification": mag,
                     "scale_px_per_0p1mm": scale, "rel_path": p.relative_to(NIST).as_posix()})
    img = pd.DataFrame(rows)
    per = img.groupby("swatch_id").replicate.apply(lambda s: tuple(sorted(s)))
    not_in_inv = sorted(set(img.swatch_id) - set(inv.sample_id))
    no_images = sorted(set(inv.sample_id) - set(img.swatch_id))
    incomplete = sorted(k for k, v in per.items() if v != tuple(VIEWS))
    count_mismatch = [r.sample_id for r in inv.itertuples() if int(r.n_images) != int((img.swatch_id == r.sample_id).sum())]
    problems += [f"swatch not in inventory: {s}" for s in not_in_inv] + [f"inventory swatch without images: {s}" for s in no_images]
    problems += [f"incomplete views: {s}" for s in incomplete] + [f"n_images mismatch: {s}" for s in count_mismatch]
    img = img.merge(inv[["sample_id"] + META_COLS], left_on="swatch_id", right_on="sample_id", how="left").drop(columns="sample_id")
    report = {"tiff_files": len(files), "parsed_images": len(img), "swatches_with_images": int(img.swatch_id.nunique()),
              "inventory_rows": len(inv), "images_per_swatch": dict(Counter(img.groupby("swatch_id").size().tolist())),
              "strict_groups": int(inv.group_strict.nunique()), "problems": problems,
              "missing_metadata_fields": {c: int((inv[c].str.strip() == "").sum()) for c in META_COLS}}
    return img, inv, report


# ---------------------------------------------------------------- phase 3: colour features
def colour_features(rgb: np.ndarray) -> dict:
    """rgb: HxWx3 uint8 at 1/4 resolution. Returns a flat dict of colour-only statistics."""
    x = rgb.astype(np.float32)
    f = {}
    for i, c in enumerate("rgb"):
        f[f"mean_{c}"] = float(x[..., i].mean()); f[f"std_{c}"] = float(x[..., i].std())
        h, _ = np.histogram(rgb[..., i], bins=16, range=(0, 256))
        for b, v in enumerate(h / h.sum()):
            f[f"hist_{c}_{b:02d}"] = float(v)
    hsv = np.asarray(Image.fromarray(rgb).convert("HSV"), dtype=np.uint8)
    for name, ch, bins in (("h", 0, 18), ("s", 1, 8), ("v", 2, 8)):
        h, _ = np.histogram(hsv[..., ch], bins=bins, range=(0, 256))
        for b, v in enumerate(h / h.sum()):
            f[f"hsv_{name}_{b:02d}"] = float(v)
    s, v = hsv[..., 1].astype(np.float32) / 255, hsv[..., 2].astype(np.float32) / 255
    lum = (x @ np.array([0.299, 0.587, 0.114], dtype=np.float32)) / 255
    f.update(brightness_mean=float(lum.mean()), brightness_std=float(lum.std()),
             value_mean=float(v.mean()), saturation_mean=float(s.mean()), saturation_std=float(s.std()),
             frac_highlight_clipped=float((rgb.max(-1) >= 250).mean()), frac_shadow_clipped=float((rgb.min(-1) <= 5).mean()))
    return f


# ---------------------------------------------------------------- phase 2: models
def load_models(device):
    import torch
    from torchvision import models

    hub = Path(torch.hub.get_dir()) / "checkpoints"
    specs = {"resnet50": (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V2),
             "efficientnet_b0": (models.efficientnet_b0, models.EfficientNet_B0_Weights.IMAGENET1K_V1)}
    out, meta = {}, {}
    for name, (ctor, w) in specs.items():
        f = hub / os.path.basename(w.url)
        if not f.is_file():
            raise FileNotFoundError(f"{name}: cached weights {f} not found (no download is attempted)")
        net = ctor(weights=None)
        net.load_state_dict(torch.load(f, map_location="cpu", weights_only=True))
        if name == "resnet50":
            net.fc = torch.nn.Identity()            # 2048-d global-average-pooled features
        else:
            net.classifier = torch.nn.Identity()    # 1280-d pooled features (dropout+linear removed)
        net = net.to(device).eval()
        tf = w.transforms()
        out[name] = (net, tf)
        meta[name] = {"torchvision_builder": ctor.__name__, "weights_enum": str(w), "weights_url": w.url,
                      "weights_file": f.name, "weights_location": "torch hub cache (checkpoints/)",
                      "weights_file_sha256": hashlib.sha256(f.read_bytes()).hexdigest(),
                      "weights_file_bytes": f.stat().st_size, "imagenet_acc@1": w.meta["_metrics"]["ImageNet-1K"]["acc@1"],
                      "head_removed": "fc -> Identity" if name == "resnet50" else "classifier (Dropout+Linear) -> Identity",
                      "preprocessing": repr(tf), "fine_tuned": False}
    return out, meta


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch", type=int, default=8)
    a = ap.parse_args(argv)
    import torch, torchvision, sklearn, matplotlib

    t0 = time.time()
    img, inv, vreport = validate()
    print(json.dumps({k: v for k, v in vreport.items() if k != "problems"}, indent=1), "problems:", vreport["problems"])
    if vreport["problems"]:
        print("VALIDATION FAILED; stopping"); return 1

    device_note = ""
    if a.device:
        device = a.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        if device == "cuda":
            torch.zeros(1, device="cuda")
    except Exception as e:  # noqa: BLE001
        device_note = f"CUDA failed ({e}); fell back to CPU"; device = "cpu"
    nets, model_meta = load_models(device)

    feats = {k: [] for k in nets}; colour = []
    paths = [NIST / p for p in img.rel_path]
    torch.manual_seed(0)
    with torch.inference_mode():
        for i in range(0, len(paths), a.batch):
            pil_batch = []
            for p in paths[i:i + a.batch]:
                with Image.open(p) as im:
                    assert im.mode == "RGBA" and im.size == (2448, 2048), (p.name, im.mode, im.size)
                    alpha = np.asarray(im.getchannel("A"))
                    assert alpha.min() == 255, f"{p.name}: non-opaque alpha"
                    rgb = im.convert("RGB")       # alpha is constant 255; dropping it loses nothing
                rgb.load()
                pil_batch.append(rgb)
                colour.append(colour_features(np.asarray(rgb.reduce(4))))
            for name, (net, tf) in nets.items():
                x = torch.stack([tf(im) for im in pil_batch]).to(device)
                feats[name].append(net(x).float().cpu().numpy())
            if (i // a.batch) % 10 == 0:
                print(f"{i + len(pil_batch)}/{len(paths)}  {time.time() - t0:.0f}s", flush=True)

    keys = ["image_id", "swatch_id", "replicate", "side", "magnification", "scale_px_per_0p1mm"] + META_COLS
    for name in nets:
        E = np.concatenate(feats[name]).astype(np.float32)
        cols = [f"f{j:04d}" for j in range(E.shape[1])]
        df = pd.concat([img[keys].reset_index(drop=True), pd.DataFrame(E, columns=cols)], axis=1)
        df.to_csv(HERE / f"{name}_image_embeddings.csv", index=False, float_format="%.6g")
        sw = df.groupby("swatch_id", sort=True)
        S = sw[cols].mean()
        first = sw[META_COLS].first()
        out = pd.concat([first, sw.size().rename("n_images"), S], axis=1).reset_index()
        out.to_csv(HERE / f"{name}_swatch_embeddings.csv", index=False, float_format="%.6g")
        model_meta[name]["embedding_dim"] = int(E.shape[1])
    C = pd.DataFrame(colour)
    cimg = pd.concat([img[keys].reset_index(drop=True), C], axis=1)
    cimg.insert(0, "level", "image")
    sw = cimg.groupby("swatch_id", sort=True)
    csw = pd.concat([sw[META_COLS].first(), sw.size().rename("n_images"), sw[list(C.columns)].mean()], axis=1).reset_index()
    csw.insert(0, "level", "swatch")
    pd.concat([cimg, csw], ignore_index=True).to_csv(HERE / "colour_features.csv", index=False, float_format="%.6g")

    meta = {
        "experiment": "NIST E1 visual feature analysis (exploratory; no classifier, no fine-tuning)",
        "run_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "runtime_s": round(time.time() - t0, 1),
        "device": device, "device_name": torch.cuda.get_device_name(0) if device == "cuda" else platform.processor(),
        "device_note": device_note or None, "batch_size": a.batch,
        "cuda_visible_devices_env": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "software": {"python": sys.version.split()[0], "torch": torch.__version__, "torchvision": torchvision.__version__,
                     "cuda_runtime": torch.version.cuda, "numpy": np.__version__, "pandas": pd.__version__,
                     "pillow": Image.__version__, "scikit-learn": sklearn.__version__, "matplotlib": matplotlib.__version__,
                     "platform": platform.platform()},
        "input": {"images": len(img), "format": "TIFF RGBA 8-bit, 2448x2048, alpha constant 255 (asserted per image)",
                  "conversion": "PIL Image.convert('RGB') (alpha dropped); RGB colour preserved"},
        "models": model_meta,
        "colour_features": {"computed_at": "1/4 resolution (PIL reduce(4) -> 612x512)", "n_features": int(C.shape[1]),
                            "contents": "mean/std RGB; 16-bin RGB histograms; HSV histograms (H18, S8, V8); brightness, value, "
                                        "saturation stats; highlight/shadow clipping fractions"},
        "swatch_aggregation": "arithmetic mean of the 6 per-image vectors (raw, un-normalised)",
        "validation": vreport,
        "dataset_root": str(NIST.relative_to(PROJECT)), "inventory": str(INVENTORY.relative_to(PROJECT)),
    }
    (HERE / "experiment_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("done", time.time() - t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
