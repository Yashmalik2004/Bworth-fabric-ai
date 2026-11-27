"""Read-only audit of the Beglarian Fabrics evaluation sample (CC BY-NC 4.0, (c) Beglarian Fabrics).

Opens dataset files for reading only and executes nothing from the dataset. Writes three CSVs to <out_dir> and,
optionally, a JSON summary (near-duplicate pairs, resolutions, relationship candidates) to [summary.json].

    python -I experiments/beglarian_dataset_audit/audit_beglarian.py \
        datasets/external/beglarian_fabrics experiments/beglarian_dataset_audit [summary.json]

Licence note: fabric_inventory.csv and image_inventory.csv reproduce per-fabric catalogue labels and per-file hashes of
the gated CC BY-NC sample. They are kept out of the public repository; regenerate them locally with this script.
"""
import csv, hashlib, json, sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

root, out = Path(sys.argv[1]), Path(sys.argv[2])
train = root / "data" / "train"
R = [json.loads(l) for l in open(train / "metadata.jsonl", encoding="utf-8") if l.strip()]
LABEL_FIELDS = ["captured", "fabric_type", "color", "composition", "weave", "stretch", "features", "intended_use",
                "country_of_origin", "width_cm", "weight_gsm", "title_en", "description_en", "description_fr"]
BANDS = [("low", 30, 40), ("medium", 50, 66), ("high", 80, 95), ("very_high", 98, 100)]
FIBRE_NORMALISE = {"Polyamid": "Polyamide/Nylon", "Nylon": "Polyamide/Nylon", "Virgin Wool": "Wool"}


def band(c):
    if c == 0: return "no_cotton"
    for b, lo, hi in BANDS:
        if lo <= c <= hi: return b
    if c < 30: return "out_of_band_below_30"
    for (b1, _, h1), (b2, l2, _) in zip(BANDS, BANDS[1:]):
        if h1 < c < l2: return f"out_of_band_gap_{h1}_{l2}"
    return "unclassified"


def dhash(im, n=8):
    g = np.asarray(im.convert("L").resize((n + 1, n), Image.BILINEAR), dtype=np.int16)
    return (g[:, 1:] > g[:, :-1]).flatten()


# ---------------- images
files = sorted(p for p in train.iterdir() if p.is_file() and p.name != "metadata.jsonl")
meta_by_file = {r["file_name"]: r for r in R}
img_rows, hashes, dh = [], defaultdict(list), {}
for p in files:
    row = {"file_name": p.name, "bytes": p.stat().st_size, "extension": p.suffix.lower()}
    try:
        with Image.open(p) as im:
            im.verify()
        with Image.open(p) as im:
            im.load()
            row.update(format=im.format, mode=im.mode, width=im.size[0], height=im.size[1], readable=True)
            rgb = np.asarray(im.convert("RGB").reduce(4), dtype=np.float32)
            row["mean_r"], row["mean_g"], row["mean_b"] = [round(float(v), 2) for v in rgb.reshape(-1, 3).mean(0)]
            row["mean_brightness"] = round(float((rgb @ np.array([0.299, 0.587, 0.114])).mean()), 2)
            dh[p.name] = dhash(im)
    except Exception as e:  # noqa: BLE001
        row.update(readable=False, error=str(e)[:200])
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    row["sha256"] = h
    hashes[h].append(p.name)
    m = meta_by_file.get(p.name)
    row["in_metadata"] = m is not None
    row["sample_id"] = m["sample_id"] if m else ""
    row["view"] = m["view"] if m else ""
    img_rows.append(row)

dup_groups = [v for v in hashes.values() if len(v) > 1]
names = list(dh)
near = []
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        a, b = names[i], names[j]
        sa, sb = meta_by_file.get(a, {}).get("sample_id"), meta_by_file.get(b, {}).get("sample_id")
        if sa != sb:
            d = int((dh[a] != dh[b]).sum())
            if d <= 6:
                near.append((a, b, d))
meta_without_image = sorted(set(meta_by_file) - {p.name for p in files})
image_without_meta = sorted({p.name for p in files} - set(meta_by_file))
dup_meta_files = [k for k, v in Counter(r["file_name"] for r in R).items() if v > 1]

# ---------------- fabrics
by_fab = defaultdict(list)
for r in R:
    by_fab[r["sample_id"]].append(r)
fab_rows, inconsistent = [], {}
img_by_name = {r["file_name"]: r for r in img_rows}
for sid, rs in sorted(by_fab.items()):
    bad = [f for f in LABEL_FIELDS if len({json.dumps(r[f], sort_keys=True) for r in rs}) > 1]
    if bad:
        inconsistent[sid] = bad
    r0 = sorted(rs, key=lambda r: r["view"])[0]
    comp = r0["composition"]
    total = round(sum(c["pct"] for c in comp), 3)
    cotton = sum(c["pct"] for c in comp if c["fiber"].strip().lower() == "cotton")
    fibres = sorted({FIBRE_NORMALISE.get(c["fiber"], c["fiber"]) for c in comp})
    nonc = [FIBRE_NORMALISE.get(c["fiber"], c["fiber"]) for c in comp if c["fiber"].strip().lower() != "cotton"]
    if not comp:
        cat = "missing"
    elif cotton == 100:
        cat = "100% cotton"
    elif set(fibres) == {"Polyester"}:
        cat = "100% polyester"
    elif cotton > 0 and set(nonc) == {"Polyester"}:
        cat = "cotton-polyester blend"
    elif cotton > 0 and set(nonc) <= {"Elastane", "Polyurethane"}:
        cat = "cotton + elastane/polyurethane only"
    elif cotton > 0:
        cat = "cotton + other fibres"
    else:
        cat = "no cotton (other fibres)"
    ambiguous = [c["fiber"] for c in comp if c["fiber"] in ("Other Fibres",)]
    views = sorted(r["view"] for r in rs)
    bright = [img_by_name[r["file_name"]].get("mean_brightness") for r in rs if r["file_name"] in img_by_name]
    v0 = img_by_name.get(r0["file_name"], {})
    fab_rows.append({
        "sample_id": sid, "n_images": len(rs), "views": " ".join(map(str, views)),
        "views_contiguous_from_0": views == list(range(len(views))), "captured": r0["captured"],
        "composition": "; ".join(f"{c['fiber']} {c['pct']:g}" for c in comp), "composition_sum": total,
        "cotton_pct": cotton, "bworth_band": band(cotton), "composition_category": cat,
        "fibres_normalised": "|".join(fibres), "n_fibres": len(comp),
        "ambiguous_fibre_labels": "|".join(ambiguous),
        "fabric_type": "|".join(r0["fabric_type"]), "weave": "|".join(r0["weave"]), "stretch": "|".join(r0["stretch"]),
        "color": "|".join(r0["color"]), "features": "|".join(r0["features"]), "intended_use": "|".join(r0["intended_use"]),
        "country_of_origin": "|".join(r0["country_of_origin"]), "width_cm": r0["width_cm"], "weight_gsm": r0["weight_gsm"],
        "title_en": r0["title_en"], "labels_consistent_across_images": not bad,
        "view0_mean_brightness": v0.get("mean_brightness"),
        "mean_brightness_all_views": round(float(np.mean([b for b in bright if b is not None])), 2) if bright else None,
        "view0_mean_rgb": f"{v0.get('mean_r')},{v0.get('mean_g')},{v0.get('mean_b')}",
    })

# possible same-underlying-fabric candidates: identical composition + gsm + width + country (+ type), different sample_id
sig = defaultdict(list)
for f in fab_rows:
    sig[(f["composition"], f["weight_gsm"], f["width_cm"], f["country_of_origin"])].append(f["sample_id"])
same_spec = {" / ".join(map(str, k)): v for k, v in sig.items() if len(v) > 1}
sig2 = defaultdict(list)
for f in fab_rows:
    sig2[(f["composition"], f["weight_gsm"])].append(f["sample_id"])
same_comp_gsm = {" / ".join(map(str, k)): v for k, v in sig2.items() if len(v) > 1}
titles = Counter(f["title_en"] for f in fab_rows)
# conservative grouping: fabrics sharing identical composition AND weight are flagged as a possible same-base-cloth group
for i, (k, ids) in enumerate(sorted(same_comp_gsm.items())):
    for f in fab_rows:
        if f["sample_id"] in ids:
            f["possible_related_group"] = f"REL-{i + 1:02d}"
            f["related_basis"] = f"identical composition and weight_gsm ({k}); " + (
                "same width and country" if k in {" / ".join(map(str, kk[:2])) for kk in [kk for kk, v in sig.items() if len(v) > 1]} else
                "width or country differs")
for f in fab_rows:
    f.setdefault("possible_related_group", "")
    f.setdefault("related_basis", "")
    f["independent_group_strict"] = f["possible_related_group"] or f["sample_id"]

# ---------------- write CSVs
out.mkdir(parents=True, exist_ok=True)
fab_by = {f["sample_id"]: f for f in fab_rows}
img_cols = ["file_name", "sample_id", "view", "in_metadata", "readable", "format", "mode", "width", "height", "extension",
            "bytes", "sha256", "mean_r", "mean_g", "mean_b", "mean_brightness"]
with open(out / "image_inventory.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=img_cols + ["cotton_pct", "bworth_band"], extrasaction="ignore")
    w.writeheader()
    for r in img_rows:
        f = fab_by.get(r["sample_id"], {})
        w.writerow({**r, "cotton_pct": f.get("cotton_pct", ""), "bworth_band": f.get("bworth_band", "")})
with open(out / "fabric_inventory.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(fab_rows[0]))
    w.writeheader(); w.writerows(fab_rows)
order = ["very_high", "high", "medium", "low", "out_of_band_below_30", "out_of_band_gap_40_50", "out_of_band_gap_66_80",
         "out_of_band_gap_95_98", "no_cotton"]
with open(out / "composition_distribution.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(["table", "key", "n_fabrics", "n_images", "sample_ids"])
    for b in order:
        fs = [f for f in fab_rows if f["bworth_band"] == b]
        w.writerow(["bworth_band", b, len(fs), sum(f["n_images"] for f in fs), " ".join(f["sample_id"] for f in fs)])
    for c, n in Counter(f["composition_category"] for f in fab_rows).most_common():
        fs = [f for f in fab_rows if f["composition_category"] == c]
        w.writerow(["composition_category", c, n, sum(f["n_images"] for f in fs), " ".join(f["sample_id"] for f in fs)])
    for v, n in sorted(Counter(f["cotton_pct"] for f in fab_rows).items()):
        fs = [f for f in fab_rows if f["cotton_pct"] == v]
        w.writerow(["cotton_pct_value", f"{v:g}", n, sum(f["n_images"] for f in fs), " ".join(f["sample_id"] for f in fs)])
    for v, n in Counter(fib for f in fab_rows for fib in f["fibres_normalised"].split("|")).most_common():
        w.writerow(["fibre_present_in_fabric", v, n, "", ""])

summary = {
    "metadata_records": len(R), "image_files": len(files), "fabrics": len(by_fab),
    "image_formats": Counter(f"{r.get('format')}/{r.get('mode')}" for r in img_rows),
    "extensions": Counter(r["extension"] for r in img_rows),
    "unreadable": [r["file_name"] for r in img_rows if not r.get("readable")],
    "resolutions": Counter(f"{r.get('width')}x{r.get('height')}" for r in img_rows).most_common(15),
    "n_distinct_resolutions": len({(r.get("width"), r.get("height")) for r in img_rows}),
    "long_side": Counter(max(r.get("width") or 0, r.get("height") or 0) for r in img_rows),
    "orientation": Counter("landscape" if (r.get("width") or 0) > (r.get("height") or 0) else ("portrait" if (r.get("width") or 0) < (r.get("height") or 0) else "square") for r in img_rows),
    "bytes_total": sum(r["bytes"] for r in img_rows),
    "sha256_duplicate_groups": dup_groups, "near_duplicate_cross_fabric_dhash_le6": near,
    "meta_without_image": meta_without_image, "image_without_meta": image_without_meta, "duplicate_file_names_in_metadata": dup_meta_files,
    "images_per_fabric": Counter(len(v) for v in by_fab.values()),
    "views_noncontiguous": [f["sample_id"] for f in fab_rows if not f["views_contiguous_from_0"]],
    "label_inconsistent_fabrics": inconsistent,
    "same_spec_candidates(comp,gsm,width,country)": same_spec, "same_comp_and_gsm": same_comp_gsm,
    "duplicate_titles": {k: v for k, v in titles.items() if v > 1},
    "composition_sum_not_100": [(f["sample_id"], f["composition_sum"]) for f in fab_rows if abs(f["composition_sum"] - 100) > 0.5],
    "captured_range": [min(f["captured"] for f in fab_rows), max(f["captured"] for f in fab_rows)],
}
if len(sys.argv) > 3:
    Path(sys.argv[3]).write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
print(json.dumps({k: v for k, v in summary.items() if k not in ("same_comp_and_gsm",)}, indent=1, default=str)[:6000])
