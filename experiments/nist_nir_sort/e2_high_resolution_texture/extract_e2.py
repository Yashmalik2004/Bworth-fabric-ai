"""NIST E2, step 1: native-resolution tile extraction, tile QC and frozen texture features for undyed white Source B.
EXPLORATORY ONLY: no classifier, no fine-tuning. Reads NIST TIFFs read-only; writes only into this E2 directory.

Tiles: non-overlapping 224x224 px tiles cut from the ORIGINAL 2448x2048 images (no whole-image resizing).
Representations per tile:
  A rgb    - original RGB
  B grey   - luma (0.299R+0.587G+0.114B), replicated to 3 channels for CNNs
  C cnorm  - per-parent-image, per-channel z-scored RGB (removes colour cast, brightness and global contrast)
Features: ResNet-50 / EfficientNet-B0 (A, B, C); Gabor and GLCM on raw grey and on per-image-normalised grey;
colour-only baseline on raw RGB. Image-level features = mean over included tiles (cached as .npy, git-ignored).

    CUDA_VISIBLE_DEVICES=0 python experiments/nist_nir_sort/e2_high_resolution_texture/extract_e2.py [--batch 16]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
NIST = PROJECT / "datasets" / "external" / "NIST_NIR_SORT" / "raw_download" / "NIST_SORT_DATA"
INVENTORY = HERE.parent / "nist_composition_inventory.csv"
SAMPLEINFO = next(NIST.glob("*/mds2-3325/SampleInformation.csv"))
CACHE = HERE / "cache"
TILE = 224
VIEWS = {"01": ("front", "low", 34.3), "02": ("front", "medium", 47.4), "03": ("front", "high", 76.2),
         "04": ("back", "low", 34.3), "05": ("back", "medium", 47.4), "06": ("back", "high", 76.2)}
QC = {"max_highlight_clip_frac": 0.20, "max_shadow_clip_frac": 0.20, "min_grey_std": 3.0,
      "min_focus_ratio_to_image_median": 0.25}
GABOR_LAMBDAS = [4, 8, 16, 32]           # px; ~5-42 um at high magnification (1.31 um/px)
GABOR_THETAS = [0, 45, 90, 135]
GLCM_LEVELS, GLCM_DISTS = 16, [1, 2, 4, 8, 16]
IMNET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMNET_STD = np.array([0.229, 0.224, 0.225], np.float32)


# ------------------------------------------------------------------ selection + validation
def select() -> tuple[pd.DataFrame, pd.DataFrame]:
    inv = pd.read_csv(INVENTORY, dtype=str, keep_default_na=False)
    si = pd.read_csv(SAMPLEINFO, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    si["Sample ID"] = si["Sample ID"].str.strip()
    si = si.set_index("Sample ID")
    inv["dye_type"] = [si.loc[s, "Dye Type"].strip() for s in inv.sample_id]
    inv["visual_colour"] = [si.loc[s, "Visual Inspected Color"].strip() for s in inv.sample_id]
    inv["construction"] = [si.loc[s, "Form Factor"].strip().lower() for s in inv.sample_id]
    inv["finish"] = [si.loc[s, "Finish"].strip() for s in inv.sample_id]
    inv["nist_notes"] = [si.loc[s, "Notes: "].strip() for s in inv.sample_id]
    sel = inv[(inv.source == "B") & (inv.dye_type.str.lower() == "none") & (inv.visual_colour.str.lower() == "white")].copy()
    sel["cotton"] = np.where(sel.cotton_percentage.astype(float) > 0, "cotton", "no cotton")
    rows = []
    for s in sel.sample_id:
        files = sorted(NIST.rglob(f"{s}T_*C.tiff"))
        for f in files:
            rep = re.search(r"T_(\d{2})C\.tiff$", f.name).group(1)
            side, mag, scale = VIEWS[rep]
            rows.append({"image_id": f.stem, "swatch_id": s, "replicate": rep, "side": side, "magnification": mag,
                         "scale_px_per_0p1mm": scale, "rel_path": f.relative_to(NIST).as_posix()})
    img = pd.DataFrame(rows)
    counts = img.groupby("swatch_id").replicate.apply(lambda r: tuple(sorted(r)))
    assert len(sel) == 18 and sel.group_strict.nunique() == 16, (len(sel), sel.group_strict.nunique())
    assert (counts == tuple(VIEWS)).all() and len(counts) == 18, counts
    return sel, img


# ------------------------------------------------------------------ features
def gabor_bank():
    bank = []
    for lam in GABOR_LAMBDAS:
        sigma = 0.56 * lam
        ks = int(2 * round(3 * sigma) + 1)
        for th in GABOR_THETAS:
            ev = cv2.getGaborKernel((ks, ks), sigma, np.deg2rad(th), lam, 0.5, 0, ktype=cv2.CV_32F)
            od = cv2.getGaborKernel((ks, ks), sigma, np.deg2rad(th), lam, 0.5, np.pi / 2, ktype=cv2.CV_32F)
            ev -= ev.mean(); od -= od.mean()                     # zero-mean: response is exposure-offset invariant
            ev /= np.linalg.norm(ev); od /= np.linalg.norm(od)
            bank.append((lam, th, ev, od))
    return bank


def glcm_feats(q: np.ndarray) -> np.ndarray:
    L = GLCM_LEVELS
    i, j = np.meshgrid(np.arange(L), np.arange(L), indexing="ij")
    out = []
    for d in GLCM_DISTS:
        acc = np.zeros(5)
        for dy, dx in ((0, d), (-d, d), (-d, 0), (-d, -d)):
            H, W = q.shape
            ya, yb = max(0, -dy), H - max(0, dy)
            xa, xb = max(0, -dx), W - max(0, dx)
            a = q[ya:yb, xa:xb]; b = q[ya + dy:yb + dy, xa + dx:xb + dx]
            P = np.bincount((a * L + b).ravel(), minlength=L * L).reshape(L, L).astype(np.float64)
            P = P + P.T; P /= P.sum()
            mi = (i * P).sum(); mj = (j * P).sum()
            si_ = np.sqrt(((i - mi) ** 2 * P).sum()); sj = np.sqrt(((j - mj) ** 2 * P).sum())
            acc += [(P * (i - j) ** 2).sum(), (P * np.abs(i - j)).sum(), (P / (1 + (i - j) ** 2)).sum(),
                    np.sqrt((P ** 2).sum()), ((i - mi) * (j - mj) * P).sum() / max(si_ * sj, 1e-12)]
        out.append(acc / 4)
    return np.concatenate(out)


GLCM_NAMES = [f"glcm_d{d}_{p}" for d in GLCM_DISTS for p in ("contrast", "dissimilarity", "homogeneity", "energy", "correlation")]
GABOR_NAMES = [f"gabor_l{l}_t{t}_{s}" for l in GABOR_LAMBDAS for t in GABOR_THETAS for s in ("mean", "std")]
COLOUR_NAMES = ["mean_r", "mean_g", "mean_b", "std_r", "std_g", "std_b", "brightness_mean", "brightness_std",
                "saturation_mean", "blue_yellow_mean", "frac_highlight_clipped", "frac_shadow_clipped"]


def colour_feats(t: np.ndarray, grey: np.ndarray) -> np.ndarray:
    x = t.astype(np.float32)
    hsv = cv2.cvtColor(t, cv2.COLOR_RGB2HSV)
    return np.array([*x.reshape(-1, 3).mean(0), *x.reshape(-1, 3).std(0), grey.mean() / 255, grey.std() / 255,
                     hsv[..., 1].mean() / 255, (x[..., 2] - (x[..., 0] + x[..., 1]) / 2).mean(),
                     (t.max(-1) >= 250).mean(), (t.min(-1) <= 5).mean()], np.float64)


def load_cnns(device):
    import torch
    from torchvision import models

    hub = Path(torch.hub.get_dir()) / "checkpoints"
    nets, meta = {}, {}
    for name, ctor, w in (("resnet50", models.resnet50, models.ResNet50_Weights.IMAGENET1K_V2),
                          ("efficientnet_b0", models.efficientnet_b0, models.EfficientNet_B0_Weights.IMAGENET1K_V1)):
        f = hub / os.path.basename(w.url)
        if not f.is_file():
            raise FileNotFoundError(f"{name}: cached weights not found ({f.name}); no download attempted")
        net = ctor(weights=None)
        net.load_state_dict(torch.load(f, map_location="cpu", weights_only=True))
        if name == "resnet50":
            net.fc = torch.nn.Identity()
        else:
            net.classifier = torch.nn.Identity()
        nets[name] = net.to(device).eval()
        meta[name] = {"weights_enum": str(w), "weights_file": f.name, "weights_location": "torch hub cache (checkpoints/)",
                      "weights_file_sha256": hashlib.sha256(f.read_bytes()).hexdigest(), "fine_tuned": False}
    return nets, meta


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=16)
    a = ap.parse_args(argv)
    import torch, torchvision

    t0 = time.time()
    sel, img = select()
    print("selected swatches:", list(sel.sample_id.str[:3]), "groups:", sel.group_strict.nunique(), "images:", len(img))
    device, note = ("cuda" if torch.cuda.is_available() else "cpu"), None
    if device == "cuda":
        try:
            torch.zeros(1, device="cuda")
        except Exception as e:  # noqa: BLE001
            device, note = "cpu", f"CUDA failed ({e}); CPU fallback"
    nets, cnn_meta = load_cnns(device)
    gmem0 = torch.cuda.mem_get_info()[0] // 2**20 if device == "cuda" else None
    bank = gabor_bank()
    CACHE.mkdir(exist_ok=True)
    tiles_rows = []
    keys = [f"{m}_{r}" for m in nets for r in ("rgb", "grey", "cnorm")] + ["gabor_grey", "gabor_greynorm", "glcm_grey",
                                                                            "glcm_greynorm", "colour_rgb"]
    feats = {k: [] for k in keys}
    mean_t = torch.tensor(IMNET_MEAN, device=device).view(1, 3, 1, 1)
    std_t = torch.tensor(IMNET_STD, device=device).view(1, 3, 1, 1)
    peak = 0
    for n_img, r in enumerate(img.itertuples()):
        with Image.open(NIST / r.rel_path) as im:
            assert im.mode == "RGBA" and im.size == (2448, 2048)
            assert np.asarray(im.getchannel("A")).min() == 255
            rgb = np.asarray(im.convert("RGB"))
        H, W = rgb.shape[:2]
        grey = (rgb.astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32))
        g_mu, g_sd = float(grey.mean()), float(grey.std())
        c_mu = rgb.reshape(-1, 3).astype(np.float32).mean(0); c_sd = rgb.reshape(-1, 3).astype(np.float32).std(0)
        nx, ny = W // TILE, H // TILE
        ox, oy = (W - nx * TILE) // 2, (H - ny * TILE) // 2
        # Gabor magnitude on the full-resolution image (no tile-edge artefacts); normalised variant = raw / image grey sd
        mags = []
        for lam, th, ev, od in bank:
            e = cv2.filter2D(grey, cv2.CV_32F, ev, borderType=cv2.BORDER_REFLECT)
            o = cv2.filter2D(grey, cv2.CV_32F, od, borderType=cv2.BORDER_REFLECT)
            mags.append(np.sqrt(e * e + o * o))
        coords = [(ox + ix * TILE, oy + iy * TILE) for iy in range(ny) for ix in range(nx)]
        lap = []
        for (x0, y0) in coords:
            gt = grey[y0:y0 + TILE, x0:x0 + TILE]
            lap.append(float(cv2.Laplacian(gt, cv2.CV_32F).var()))
        lap_med = float(np.median(lap))
        inc_tiles = []
        for k, (x0, y0) in enumerate(coords):
            t = rgb[y0:y0 + TILE, x0:x0 + TILE]; gt = grey[y0:y0 + TILE, x0:x0 + TILE]
            hi, lo, gsd = float((t.max(-1) >= 250).mean()), float((t.min(-1) <= 5).mean()), float(gt.std())
            reasons = []
            if hi > QC["max_highlight_clip_frac"]: reasons.append("highlight_clipping")
            if lo > QC["max_shadow_clip_frac"]: reasons.append("shadow_clipping")
            if gsd < QC["min_grey_std"]: reasons.append("insufficient_texture_or_fabric")
            if lap[k] < QC["min_focus_ratio_to_image_median"] * lap_med: reasons.append("poor_focus")
            tiles_rows.append({"tile_id": f"{r.image_id}_x{x0}_y{y0}", "image_id": r.image_id, "swatch_id": r.swatch_id,
                               "replicate": r.replicate, "side": r.side, "magnification": r.magnification,
                               "x0": x0, "y0": y0, "x1": x0 + TILE, "y1": y0 + TILE, "frac_highlight_clipped": round(hi, 5),
                               "frac_shadow_clipped": round(lo, 5), "grey_std": round(gsd, 3), "laplacian_var": round(lap[k], 3),
                               "image_median_laplacian_var": round(lap_med, 3), "included": not reasons,
                               "exclusion_reason": ";".join(reasons)})
            if not reasons:
                inc_tiles.append((x0, y0))
        if not inc_tiles:
            raise RuntimeError(f"{r.image_id}: all tiles excluded")
        # handcrafted per included tile
        gab_raw, glr, gln, col = [], [], [], []
        for (x0, y0) in inc_tiles:
            sl = (slice(y0, y0 + TILE), slice(x0, x0 + TILE))
            gab_raw.append(np.concatenate([[m[sl].mean(), m[sl].std()] for m in mags]))
            gt = grey[sl]
            glr.append(glcm_feats(np.clip((gt / 256 * GLCM_LEVELS).astype(np.int64), 0, GLCM_LEVELS - 1)))
            z = np.clip((gt - g_mu) / g_sd, -2.5, 2.5)
            gln.append(glcm_feats(np.clip(((z + 2.5) / 5 * GLCM_LEVELS).astype(np.int64), 0, GLCM_LEVELS - 1)))
            col.append(colour_feats(rgb[sl], gt))
        gab_raw = np.array(gab_raw)
        feats["gabor_grey"].append(gab_raw.mean(0)); feats["gabor_greynorm"].append(gab_raw.mean(0) / g_sd)
        feats["glcm_grey"].append(np.mean(glr, 0)); feats["glcm_greynorm"].append(np.mean(gln, 0))
        feats["colour_rgb"].append(np.mean(col, 0))
        # CNN features
        tiles = np.stack([rgb[y0:y0 + TILE, x0:x0 + TILE] for (x0, y0) in inc_tiles]).astype(np.float32)   # N,224,224,3
        greys = tiles @ np.array([0.299, 0.587, 0.114], np.float32)
        reps = {"rgb": tiles / 255.0, "grey": np.repeat(greys[..., None] / 255.0, 3, -1),
                "cnorm": (tiles - c_mu) / c_sd}
        with torch.inference_mode():
            for rname, arr in reps.items():
                x_all = torch.from_numpy(np.ascontiguousarray(arr.transpose(0, 3, 1, 2)))
                for mname, net in nets.items():
                    out = []
                    for b in range(0, len(x_all), a.batch):
                        x = x_all[b:b + a.batch].to(device)
                        if rname != "cnorm":
                            x = (x - mean_t) / std_t
                        out.append(net(x).float().cpu().numpy())
                    feats[f"{mname}_{rname}"].append(np.concatenate(out).mean(0))
        if device == "cuda":
            peak = max(peak, torch.cuda.max_memory_allocated() // 2**20)
        print(f"{n_img + 1}/{len(img)} {r.image_id[:3]} rep{r.replicate}: {len(inc_tiles)}/{len(coords)} tiles  {time.time() - t0:.0f}s", flush=True)

    img.to_csv(CACHE / "image_index.csv", index=False)
    for k, v in feats.items():
        np.save(CACHE / f"img_{k}.npy", np.vstack(v).astype(np.float32))
    names = {"gabor_grey": GABOR_NAMES, "gabor_greynorm": GABOR_NAMES, "glcm_grey": GLCM_NAMES, "glcm_greynorm": GLCM_NAMES,
             "colour_rgb": COLOUR_NAMES}
    (CACHE / "feature_names.json").write_text(json.dumps(names, indent=1), encoding="utf-8")
    tinv = pd.DataFrame(tiles_rows).merge(sel[["sample_id", "group_strict", "cotton", "cotton_percentage", "construction"]],
                                          left_on="swatch_id", right_on="sample_id").drop(columns="sample_id")
    tinv.to_csv(HERE / "tile_inventory.csv", index=False)
    sel.to_csv(CACHE / "selection.csv", index=False)
    meta = {
        "experiment": "NIST E2 high-resolution texture analysis (exploratory; no classifier, no fine-tuning)",
        "run_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "runtime_s": round(time.time() - t0, 1),
        "device": device, "device_name": torch.cuda.get_device_name(0) if device == "cuda" else platform.processor(),
        "device_note": note, "gpu_free_mib_after_model_load": gmem0, "gpu_peak_allocated_mib": peak if device == "cuda" else None,
        "batch_size": a.batch, "cuda_visible_devices_env": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "software": {"python": sys.version.split()[0], "torch": torch.__version__, "torchvision": torchvision.__version__,
                     "opencv": cv2.__version__, "numpy": np.__version__, "pandas": pd.__version__, "pillow": Image.__version__,
                     "platform": platform.platform(), "scikit_image": "not installed (GLCM implemented in NumPy)"},
        "selection": {"rule": "inventory source == B AND SampleInformation 'Dye Type' == none AND 'Visual Inspected Color' == White",
                      "swatches": sel.sample_id.tolist(), "n_swatches": int(len(sel)), "n_strict_groups": int(sel.group_strict.nunique()),
                      "images": int(len(img))},
        "tiling": {"tile_px": TILE, "grid": "non-overlapping, centred on the 2448x2048 image (10 x 9 = 90 tiles/image)",
                   "no_resizing": True, "qc_thresholds": QC,
                   "qc_definitions": {"highlight": "fraction of pixels with max(R,G,B) >= 250", "shadow": "fraction with min(R,G,B) <= 5",
                                      "texture": "std of luma (0-255)", "focus": "variance of Laplacian of luma, relative to the image's median tile"}},
        "representations": {"rgb": "original RGB /255, ImageNet mean/std normalisation",
                            "grey": "luma 0.299R+0.587G+0.114B, /255, replicated to 3 channels, ImageNet normalisation",
                            "cnorm": "per-parent-image per-channel z-score (whole-image mean/std), fed directly to the CNN"},
        "cnn": {**cnn_meta, "input": "native 224x224 tiles (no resize, no crop)", "pooling": "global average pool (penultimate)",
                "image_feature": "mean over included tiles"},
        "handcrafted": {"gabor": {"lambdas_px": GABOR_LAMBDAS, "thetas_deg": GABOR_THETAS, "sigma": "0.56*lambda", "gamma": 0.5,
                                  "kernels": "zero-mean, L2-normalised even/odd pair; magnitude; tile mean and std; computed on full-res luma",
                                  "greynorm": "raw Gabor features / image luma std (exact, kernels are zero-mean and linear)"},
                        "glcm": {"levels": GLCM_LEVELS, "distances_px": GLCM_DISTS, "angles_deg": [0, 45, 90, 135],
                                 "properties": ["contrast", "dissimilarity", "homogeneity", "energy", "correlation"],
                                 "grey": "fixed 0-255 quantisation", "greynorm": "per-image z-score clipped to +/-2.5 then quantised"},
                        "colour": COLOUR_NAMES},
        "dataset_root": str(NIST.relative_to(PROJECT)),
    }
    (HERE / "experiment_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("done", round(time.time() - t0), "s; tiles", len(tinv), "included", int(tinv.included.sum()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
