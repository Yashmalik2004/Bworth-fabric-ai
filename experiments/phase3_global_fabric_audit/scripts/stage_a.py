"""Phase 3B Stage A: dataset-wide physical-fabric identity audit (no training).

Run from the project root:
    python experiments/phase3_global_fabric_audit/scripts/stage_a.py inventory
    python experiments/phase3_global_fabric_audit/scripts/stage_a.py photos
    python experiments/phase3_global_fabric_audit/scripts/stage_a.py families
    python experiments/phase3_global_fabric_audit/scripts/stage_a.py apply      # after manual review
Reads the original images only; never writes outside experiments/phase3_global_fabric_audit/.
"""

from __future__ import annotations

import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import AUGMENTED_DIR, BAND_DEFINITIONS, FINE_TO_BAND, ORIGINAL_DIR, PROCESSED_DIR  # noqa: E402

EXP = PROJECT_ROOT / "experiments" / "phase3_global_fabric_audit"
AUDIT, SHEETS, MAPS, EXCL, DATA, SPLITS, REPORTS, LOGS = (EXP / d for d in (
    "audit", "audit/contact_sheets", "mappings", "excluded", "datasets", "splits", "reports", "logs"))
BANDS = list(BAND_DEFINITIONS)
SEED = 42
DHASH_MAX, MAD_MAX = 8, 6.0          # same-photograph criterion (pairwise, complete linkage)


def rel(p: Path) -> str:
    return Path(p).resolve().relative_to(PROJECT_ROOT).as_posix()


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dhash_hex(im: Image.Image, size: int = 16) -> str:
    g = np.asarray(im.convert("L").resize((size + 1, size), Image.BILINEAR), dtype=np.int16)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return np.packbits(bits).tobytes().hex()


def hex_bits(h: str) -> np.ndarray:
    return np.unpackbits(np.frombuffer(bytes.fromhex(h), dtype=np.uint8)).astype(bool)


# --------------------------------------------------------------------------- #
# Inventory
# --------------------------------------------------------------------------- #
def inspect(p: Path, decode: bool) -> dict:
    d = {"original_path": rel(p), "filename": p.name, "file_size_bytes": p.stat().st_size, "sha256": sha256(p)}
    try:
        with Image.open(p) as im:
            d["width"], d["height"], d["mode"] = im.size[0], im.size[1], im.mode
            if decode:
                im = ImageOps.exif_transpose(im)
                d["dhash16"] = dhash_hex(im)
        d["readable"] = True
    except Exception as exc:  # noqa: BLE001
        d["readable"], d["error"] = False, str(exc)
    return d


def cmd_inventory() -> int:
    rows = []
    for subset, root, decode in (("Original", ORIGINAL_DIR, True), ("Augmented", AUGMENTED_DIR, False)):
        for cls_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            files = sorted(f for f in cls_dir.iterdir() if f.is_file())
            with ThreadPoolExecutor(8) as ex:
                recs = list(ex.map(lambda f: inspect(f, decode), files))
            for r in recs:
                r.update({"subset": subset, "original_class": cls_dir.name,
                          "original_cotton_percentage": int(cls_dir.name[:-6]),
                          "cotton_band": FINE_TO_BAND.get(cls_dir.name, "")})
            rows += recs
            print(f"  {subset}/{cls_dir.name}: {len(files)}", flush=True)
    inv = pd.DataFrame(rows)

    # existing membership (Phase 1 / Phase 2 / Phase 3 provisional)
    v1 = pd.read_csv(PROCESSED_DIR / "splits_group.csv").set_index("path")
    v2 = pd.read_csv(PROCESSED_DIR / "splits_group_v2.csv").set_index("path")
    folds = pd.read_csv(PROCESSED_DIR / "cv" / "folds.csv").set_index("path")
    conf = pd.read_csv(PROCESSED_DIR / "removed_label_conflicts.csv")
    memb = pd.read_csv(PROJECT_ROOT / "evaluation/composition/diagnostics/membership_original.csv").set_index("path")
    prov = pd.read_csv(PROJECT_ROOT / "experiments/phase3_recovered_data/audit/recoverable_groups_audit_PROVISIONAL.csv")
    prov_rep = prov.set_index("representative")["final_status"].to_dict()
    p = inv["original_path"]
    inv["phase1_split_v1"] = p.map(v1["split"])
    inv["phase1_group_v1"] = p.map(v1["group"])
    inv["phase2_split_v2"] = p.map(v2["split"])
    inv["phase2_group_v2"] = p.map(v2["group"])
    inv["phase2_cv_fold"] = p.map(folds["fold"])
    inv["phase1_membership_status"] = p.map(memb["status"])
    inv["phase1_conflict_cluster"] = p.map(conf.set_index("path")["dup_cluster"])
    inv["phase3_provisional_status"] = p.map(prov_rep)
    inv.loc[inv.subset == "Augmented", "phase1_membership_status"] = "not_used_no_source_mapping"
    inv.insert(0, "image_id", [f"IMG_{i:05d}" for i in range(len(inv))])
    AUDIT.mkdir(parents=True, exist_ok=True)
    inv.to_csv(AUDIT / "complete_image_inventory.csv", index=False)
    o = inv[inv.subset == "Original"]
    print(f"Inventory: {len(inv)} files ({len(o)} Original, {len(inv) - len(o)} Augmented); "
          f"unreadable {(~inv.readable).sum()}; unique sha256 in Original {o.sha256.nunique()}")
    return 0


# --------------------------------------------------------------------------- #
# Photo-level identity (exact + near-duplicate photographs)
# --------------------------------------------------------------------------- #
def small_gray(path: str) -> np.ndarray:
    with Image.open(PROJECT_ROOT / path) as im:
        return np.asarray(ImageOps.exif_transpose(im).convert("L").resize((64, 64), Image.BILINEAR), dtype=np.float32)


def cmd_photos() -> int:
    inv = pd.read_csv(AUDIT / "complete_image_inventory.csv")
    o = inv[inv.subset == "Original"].reset_index(drop=True)
    # 1) byte-identical groups
    sha_rep = o.groupby("sha256").image_id.first()
    uniq = o.drop_duplicates("sha256").reset_index(drop=True)
    H = np.stack([hex_bits(h) for h in uniq.dhash16])
    D = (H[:, None, :] != H[None, :, :]).sum(-1)
    # 2) near-identical candidates by dHash, confirmed by pixel MAD; complete linkage (no chaining)
    cand = np.argwhere(np.triu(D <= DHASH_MAX, 1))
    grays = {}
    def g(i):
        if i not in grays:
            grays[i] = small_gray(uniq.original_path[i])
        return grays[i]
    pairs = []
    for i, j in cand:
        mad = float(np.abs(g(i) - g(j)).mean())
        pairs.append({"a": uniq.image_id[i], "b": uniq.image_id[j], "dhash_dist": int(D[i, j]), "mad": round(mad, 2),
                      "same_photograph": mad <= MAD_MAX})
    P = pd.DataFrame(pairs)
    # complete-linkage merge of confirmed same-photograph pairs
    same = {(r.a, r.b) for r in P.itertuples() if r.same_photograph} | {(r.b, r.a) for r in P.itertuples() if r.same_photograph}
    clusters: list[set] = []
    for a, b in sorted(same):
        ca = next((c for c in clusters if a in c), None)
        cb = next((c for c in clusters if b in c), None)
        if ca is cb and ca is not None:
            continue
        merged = (ca or {a}) | (cb or {b})
        if all((x, y) in same for x, y in combinations(merged, 2)):
            clusters = [c for c in clusters if c is not ca and c is not cb] + [merged]
    photo_of = {}
    for k, c in enumerate(sorted(clusters, key=lambda c: sorted(c)[0])):
        for x in c:
            photo_of[x] = k
    nxt = len(clusters)
    for x in uniq.image_id:
        if x not in photo_of:
            photo_of[x] = nxt; nxt += 1
    uniq["photo_id"] = [f"PH{photo_of[x]:04d}" for x in uniq.image_id]
    o["photo_id"] = o.sha256.map(uniq.set_index("sha256").photo_id)
    o["sha_group_size"] = o.groupby("sha256").image_id.transform("size")
    o["photo_group_size"] = o.groupby("photo_id").image_id.transform("size")
    o["photo_classes"] = o.groupby("photo_id").original_class.transform(lambda s: "|".join(sorted(set(s), key=lambda c: int(c[:-6]))))
    o["photo_bands"] = o.groupby("photo_id").cotton_band.transform(lambda s: "|".join(sorted(set(s))))
    MAPS.mkdir(parents=True, exist_ok=True)
    o[["image_id", "original_path", "original_class", "cotton_band", "sha256", "dhash16", "photo_id", "sha_group_size",
       "photo_group_size", "photo_classes", "photo_bands"]].to_csv(MAPS / "image_to_photograph.csv", index=False)
    P.to_csv(AUDIT / "near_duplicate_photo_pairs.csv", index=False)
    ph = o.groupby("photo_id").agg(n=("image_id", "size"), classes=("photo_classes", "first"), bands=("photo_bands", "first"))
    print(f"Original images {len(o)} -> unique sha256 {len(uniq)} -> unique photographs {ph.shape[0]}")
    print(f"near-dup candidate pairs {len(P)}, confirmed same-photo {int(P.same_photograph.sum()) if len(P) else 0}")
    print(f"photographs under >1 class: {(ph.classes.str.contains('|', regex=False)).sum()} | under >1 band: {(ph.bands.str.contains('|', regex=False)).sum()}")
    return 0


# --------------------------------------------------------------------------- #
# Families (shortlist for physical-identity review) + contact sheets
# --------------------------------------------------------------------------- #
def photo_table() -> pd.DataFrame:
    m = pd.read_csv(MAPS / "image_to_photograph.csv")
    t = m.groupby("photo_id").agg(rep=("original_path", "first"), n_files=("image_id", "size"),
                                  classes=("photo_classes", "first"), bands=("photo_bands", "first")).reset_index()
    return t


def color_feat(path: str) -> np.ndarray:
    with Image.open(PROJECT_ROOT / path) as im:
        a = np.asarray(ImageOps.exif_transpose(im).convert("HSV").resize((64, 64)), dtype=np.float32) / 255
    h, _ = np.histogramdd(a.reshape(-1, 3), bins=(12, 4, 4), range=((0, 1), (0, 1), (0, 1)))
    h = h.flatten()
    return h / np.linalg.norm(h)


def cmd_families(threshold: float = 0.30) -> int:
    from sklearn.cluster import AgglomerativeClustering

    from training.composition.dataset import extract_embeddings

    t = photo_table()
    emb = extract_embeddings([str(PROJECT_ROOT / p) for p in t.rep])
    col = np.stack([color_feat(p) for p in t.rep])
    np.save(AUDIT / "photo_embeddings_imagenet.npy", emb)
    np.save(AUDIT / "photo_color_hist.npy", col)
    # combined cosine distance: texture/appearance embedding + colour histogram
    S = 0.7 * (emb @ emb.T) + 0.3 * (col @ col.T)
    Dm = np.clip(1 - S, 0, 2)
    lab = AgglomerativeClustering(n_clusters=None, metric="precomputed", linkage="average",
                                  distance_threshold=threshold).fit_predict(Dm)
    t["family"] = [f"FAM{k:03d}" for k in lab]
    # nearest out-of-family neighbours (to catch under-merging during review)
    np.fill_diagonal(S, -1)
    nn = []
    for i in range(len(t)):
        order = np.argsort(-S[i])
        out = [j for j in order if t.family[j] != t.family[i]][:2]
        nn.append(";".join(f"{t.photo_id[j]}({t.family[j]},{S[i, j]:.2f})" for j in out))
    t["nearest_other_family"] = nn
    t.to_csv(AUDIT / "photo_families.csv", index=False)
    fam = t.groupby("family").agg(n=("photo_id", "size"), classes=("classes", lambda s: "|".join(sorted(set("|".join(s).split("|")), key=lambda c: int(c[:-6])))))
    fam["n_classes"] = fam.classes.str.count(r"\|") + 1
    fam["bands"] = fam.classes.map(lambda s: "|".join(sorted({FINE_TO_BAND[c] for c in s.split("|")})))
    fam["n_bands"] = fam.bands.str.count(r"\|") + 1
    fam.to_csv(AUDIT / "family_summary.csv")
    print(f"{len(t)} photographs -> {len(fam)} families (threshold {threshold})")
    print("family size distribution:", fam.n.describe().round(1).to_dict())
    print("families with >1 class:", (fam.n_classes > 1).sum(), "| >1 band:", (fam.n_bands > 1).sum(),
          "| photos in multi-band families:", fam.loc[fam.n_bands > 1, "n"].sum())
    family_sheets(t, fam)
    return 0


def _font(size=13):
    for f in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            pass
    return ImageFont.load_default()


BAND_COL = {"low_cotton": (214, 39, 40), "medium_cotton": (31, 119, 180), "high_cotton": (44, 160, 44),
            "very_high_cotton": (148, 103, 189)}


def draw_tiles(items, cols, tile, title, out):
    """items: list of (path, caption, band_or_bands)."""
    rows = (len(items) + cols - 1) // cols
    W = Image.new("RGB", (cols * tile, 28 + rows * (tile + 18)), "white")
    d = ImageDraw.Draw(W)
    d.text((4, 4), title, fill="black", font=_font(15))
    f = _font(12)
    for k, (p, cap, bands) in enumerate(items):
        x, y = (k % cols) * tile, 28 + (k // cols) * (tile + 18)
        W.paste(ImageOps.fit(Image.open(PROJECT_ROOT / p).convert("RGB"), (tile - 4, tile - 4)), (x + 2, y))
        b = bands.split("|")
        col = BAND_COL[b[0]] if len(b) == 1 else (0, 0, 0)
        d.rectangle([x + 2, y + tile - 4, x + tile - 2, y + tile + 14], fill=(255, 255, 255))
        d.text((x + 3, y + tile - 3), cap, fill=col, font=f)
    W.save(out, quality=88)


def family_sheets(t: pd.DataFrame, fam: pd.DataFrame) -> None:
    out = SHEETS / "review_families"
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.jpg"):
        old.unlink()
    order = fam.sort_values(["n_bands", "n_classes", "n"], ascending=False).index
    for fid in order:
        g = t[t.family == fid].sort_values("classes")
        items = [(r.rep, f"{r.photo_id[2:]} {r.classes.replace('cotton', '')}", r.bands) for r in g.itertuples()]
        cols = min(10, max(4, len(items)))
        f = fam.loc[fid]
        draw_tiles(items, cols, 150, f"{fid}: {f.n} photos | classes {f.classes.replace('cotton', '%')} | bands {f.bands}",
                   out / f"{'X' if f.n_bands > 1 else ('C' if f.n_classes > 1 else 'S')}_{fid}.jpg")


if __name__ == "__main__" and sys.argv[1] in ("inventory", "photos", "families"):
    cmd = sys.argv[1]
    raise SystemExit({"inventory": cmd_inventory, "photos": cmd_photos,
                      "families": lambda: cmd_families(float(sys.argv[2]) if len(sys.argv) > 2 else 0.30)}[cmd]())


# --------------------------------------------------------------------------- #
# Packed review pages + cross-family candidate pairs
# --------------------------------------------------------------------------- #
def cmd_pages(tile: int = 150, cols: int = 10, max_rows: int = 6) -> int:
    t = pd.read_csv(AUDIT / "photo_families.csv")
    fam = pd.read_csv(AUDIT / "family_summary.csv", index_col=0)
    out = SHEETS / "review_pages"
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.jpg"):
        old.unlink()
    f = _font(12); fh = _font(14)
    for kind, sel in (("X_multiband", fam.n_bands > 1), ("C_multiclass_oneband", (fam.n_bands == 1) & (fam.n_classes > 1)),
                      ("S_singleclass", fam.n_classes == 1)):
        fids = fam[sel].sort_values("n", ascending=False).index.tolist()
        page, rows_used, blocks, pno = [], 0, [], 1
        def flush():
            nonlocal page, rows_used, pno
            if not page:
                return
            H = sum(28 + r * (tile + 18) for _, _, r in page)
            W = Image.new("RGB", (cols * tile, H), "white"); d = ImageDraw.Draw(W); y = 0
            for fid, g, r in page:
                fr = fam.loc[fid]
                d.text((4, y + 6), f"{fid}  ({fr.n} photos; {fr.classes.replace('cotton', '%')})", fill="black", font=fh)
                y += 28
                for k, row in enumerate(g.itertuples()):
                    x, yy = (k % cols) * tile, y + (k // cols) * (tile + 18)
                    W.paste(ImageOps.fit(Image.open(PROJECT_ROOT / row.rep).convert("RGB"), (tile - 4, tile - 4)), (x + 2, yy))
                    b = row.bands.split("|")
                    d.text((x + 3, yy + tile - 3), f"{row.photo_id[2:]} {row.classes.replace('cotton', '')}",
                           fill=BAND_COL[b[0]] if len(b) == 1 else (0, 0, 0), font=f)
                y += r * (tile + 18)
            W.save(out / f"{kind}_p{pno:02d}.jpg", quality=86)
            pno += 1; page = []; rows_used = 0
        for fid in fids:
            g = t[t.family == fid].sort_values(["bands", "classes"])
            r = (len(g) + cols - 1) // cols
            if rows_used + r > max_rows:
                flush()
            page.append((fid, g, r)); rows_used += r
        flush()
    print(sorted(p.name for p in out.glob("*.jpg")))
    return 0


def cmd_crosspairs(min_sim: float = 0.80, limit: int = 40) -> int:
    t = pd.read_csv(AUDIT / "photo_families.csv")
    emb = np.load(AUDIT / "photo_embeddings_imagenet.npy"); col = np.load(AUDIT / "photo_color_hist.npy")
    S = 0.7 * (emb @ emb.T) + 0.3 * (col @ col.T)
    fam = t.family.to_numpy(); bands = t.bands.to_numpy()
    rows = []
    for i, j in zip(*np.triu_indices(len(t), 1)):
        if fam[i] != fam[j] and set(bands[i].split("|")) != set(bands[j].split("|")) and S[i, j] >= min_sim:
            rows.append((S[i, j], i, j))
    rows.sort(reverse=True)
    P = pd.DataFrame([{"photo_a": t.photo_id[i], "family_a": fam[i], "classes_a": t.classes[i],
                       "photo_b": t.photo_id[j], "family_b": fam[j], "classes_b": t.classes[j], "similarity": round(s, 3)}
                      for s, i, j in rows])
    P.to_csv(AUDIT / "cross_family_cross_band_candidates.csv", index=False)
    print(f"cross-family, cross-band pairs with similarity >= {min_sim}: {len(P)}")
    items = []
    for s, i, j in rows[:limit]:
        items += [(t.rep[i], f"{t.photo_id[i][2:]} {t.classes[i].replace('cotton', '')}", t.bands[i]),
                  (t.rep[j], f"{t.photo_id[j][2:]} {t.classes[j].replace('cotton', '')} s{s:.2f}", t.bands[j])]
    if items:
        draw_tiles(items, 8, 150, f"Cross-family, cross-band look-alike pairs (top {min(limit, len(rows))}, sim >= {min_sim}); pairs side by side",
                   SHEETS / "review_cross_family_pairs.jpg")
    return 0


if __name__ == "__main__" and sys.argv[1] in ("pages", "crosspairs"):
    raise SystemExit(cmd_pages() if sys.argv[1] == "pages" else cmd_crosspairs())
