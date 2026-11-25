"""Readiness gates R1-R9 (experiments/phase5/PHASE5_PROTOCOL.md section 6) on a frozen snapshot.

Overall: READY (all pass), PILOT_ONLY (all pass except R1 at pilot level), NOT_READY otherwise.
Writes <snapshot>/readiness/readiness.json, READINESS_REPORT.md and lookalike_candidates.csv.
R8 review decisions are read from <snapshot>/readiness/lookalike_review.csv with columns
decision in {distinct, same}, reviewer, date and either image_a, image_b (one image pair) or
fabric_a, fabric_b (all image pairs of two fabrics). Flags never merge fabrics or declare them identical.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from training.phase5 import embeddings
from training.phase5.baselines import colour_family, colour_histogram, grouped_cv_metadata_balanced_accuracy
from training.phase5.protocol import CLASSES, bands, gates
from training.phase5.provenance import build_groups, norm
from training.phase5.snapshot import image_root, load_meta, load_tables, verify_snapshot
from training.phase5.splits import verify_lockbox


def cramers_v(a: pd.Series, b: pd.Series) -> float:
    from scipy.stats import chi2_contingency

    t = pd.crosstab(a, b)
    if t.shape[0] < 2 or t.shape[1] < 2:
        return 0.0
    chi2 = chi2_contingency(t, correction=False)[0]
    n = t.to_numpy().sum()
    return float(np.sqrt(chi2 / (n * (min(t.shape) - 1))))


def distance_to_band(c: float, lo: float, hi: float) -> float:
    """dist(c, [lo, hi]) = max(lo - c, 0, c - hi)."""
    return max(lo - c, 0.0, c - hi)


def adjacent_band_distance(c: float, own: str) -> float:
    """Distance from a verified cotton % to the nearest OTHER band (protocol v1.1, R7)."""
    return min(distance_to_band(c, lo, hi) for k, (lo, hi) in bands().items() if k != own)


def boundary_flags(cotton: pd.Series, band: pd.Series, distance: float) -> pd.Series:
    """R7 (protocol v1.1): at risk iff the nearest other band is within ``distance`` points. Edges facing an empty
    gap or the physical limits 0/100 do not count."""
    return pd.Series([adjacent_band_distance(float(c), k) <= distance + 1e-9 for c, k in zip(cotton, band)], index=cotton.index)


def gap_edge_proximity(cotton: pd.Series, band: pd.Series, distance: float) -> pd.Series:
    """Informational only (not gated): within ``distance`` of an own-band edge that faces a gap (not 0/100)."""
    b = bands()
    out = []
    for c, k in zip(cotton.astype(float), band):
        lo, hi = b[k]
        out.append(bool((lo > 0 and c - lo <= distance) or (hi < 100 and hi - c <= distance)))
    return pd.Series(out, index=cotton.index)


def similarity_features(path: Path) -> np.ndarray:
    from PIL import Image, ImageOps

    col = colour_histogram(path)
    with Image.open(path) as im:
        g = np.asarray(ImageOps.exif_transpose(im).convert("L").resize((16, 16)), dtype=np.float32).flatten()
    g = (g - g.mean()) / (g.std() + 1e-6)
    v = np.concatenate([col / (np.linalg.norm(col) + 1e-9), g / (np.linalg.norm(g) + 1e-9)])
    return v / np.linalg.norm(v)


def lookalike_candidates(paths, provenance_ids, image_ids, fabric_ids, colour_threshold: float,
                         embedding_threshold: float | None) -> dict:
    """Cross-group image pairs flagged by colour/structure similarity and/or cached ResNet-50 embedding similarity.
    Each pair carries both scores (embedding score None if unavailable) and the reason(s) it was flagged."""
    n = len(paths)
    out = {"pairs": [], "n_colour": 0, "n_embedding": 0}
    if n < 2:
        return out
    col = np.stack([similarity_features(p) for p in paths])
    Sc = col @ col.T
    emb = embeddings.embed(list(paths)) if embedding_threshold is not None else None
    Se = emb @ emb.T if emb is not None else None
    pid = np.asarray(provenance_ids)
    for a, b in zip(*np.triu_indices(n, 1)):
        if pid[a] == pid[b]:
            continue
        reasons = []
        if Sc[a, b] >= colour_threshold:
            reasons.append(f"colour/structure similarity {Sc[a, b]:.3f} >= {colour_threshold}")
        if Se is not None and Se[a, b] >= embedding_threshold:
            reasons.append(f"ImageNet ResNet-50 embedding similarity {Se[a, b]:.3f} >= {embedding_threshold}")
        if reasons:
            out["n_colour"] += int(Sc[a, b] >= colour_threshold)
            out["n_embedding"] += int(Se is not None and Se[a, b] >= embedding_threshold)
            out["pairs"].append({"image_a": image_ids[a], "image_b": image_ids[b], "fabric_a": fabric_ids[a],
                                 "fabric_b": fabric_ids[b], "colour_structure_similarity": round(float(Sc[a, b]), 4),
                                 "embedding_similarity": None if Se is None else round(float(Se[a, b]), 4),
                                 "reasons": "; ".join(reasons)})
    return out


def review_decisions(rev: pd.DataFrame) -> dict:
    """Map (image pair) and (fabric pair) keys, in both orders, to the reviewer's decision."""
    d = {}
    for r in rev.to_dict("records"):
        dec = str(r.get("decision", "")).strip().lower()
        if dec not in ("distinct", "same"):
            continue
        for x, y, kind in (("image_a", "image_b", "img"), ("fabric_a", "fabric_b", "fab")):
            if str(r.get(x, "")).strip() and str(r.get(y, "")).strip():
                d[(kind, r[x], r[y])] = d[(kind, r[y], r[x])] = dec
    return d


def decision_for(pair: dict, decided: dict) -> str | None:
    return decided.get(("img", pair["image_a"], pair["image_b"])) or decided.get(("fab", pair["fabric_a"], pair["fabric_b"]))


def evaluate(snap: Path) -> dict:
    snap = Path(snap)
    G = gates()
    t = load_tables(snap)
    man, fabrics, ver, issues = t["manifest"], t["fabrics"], t["verification"], t["issues"]
    meta = load_meta(snap)
    out_dir = snap / "readiness"
    out_dir.mkdir(exist_ok=True)
    res: dict[str, dict] = {}
    fab = man.groupby("fabric_id").agg(band=("band", "first"), cotton=("cotton_percentage", "first")).reset_index()
    groups = build_groups(fabrics, ver, fab.fabric_id).set_index("fabric_id")
    fab["provenance_id"] = fab.fabric_id.map(groups.provenance_id)
    fmeta = fabrics.drop_duplicates("fabric_id").set_index("fabric_id")

    # R1
    per_band = fab.band.value_counts().reindex(CLASSES, fill_value=0).astype(int).to_dict()
    lvl = ("PASS" if min(per_band.values()) >= G["R1_min_verified_fabrics_per_band"] else
           "PILOT" if min(per_band.values()) >= G["R1_pilot_min_verified_fabrics_per_band"] else "FAIL")
    res["R1"] = {"status": lvl, "verified_fabrics_per_band": per_band,
                 "required": G["R1_min_verified_fabrics_per_band"], "pilot_minimum": G["R1_pilot_min_verified_fabrics_per_band"]}
    # R2
    n_err = int((issues.severity == "ERROR").sum())
    pol = int(meta["verification_policy_version"])
    res["R2"] = {"status": "PASS" if n_err <= G["R2_max_validator_errors"] and pol >= G["R2_required_verification_policy_version"] else "FAIL",
                 "validator_errors": n_err, "verification_policy_version": pol}
    # R3
    by_fab = man.groupby("fabric_id")
    short = [f for f, g in by_fab if len(g) < G["R3_min_photos_per_fabric"] or not set(G["R3_required_image_types"]) <= set(g.image_type)]
    res["R3"] = {"status": "PASS" if not short and len(fab) else "FAIL", "fabrics_failing": short[:50], "n_failing": len(short)}
    # R4
    spans = fab.groupby("provenance_id").band.nunique()
    keys = {}
    for f in fmeta.reset_index().itertuples():
        if f.fabric_id in groups.index and norm(f.source_product_id):
            keys.setdefault(norm(f.source_product_id), set()).add(groups.provenance_id[f.fabric_id])
    shared = {k: sorted(v) for k, v in keys.items() if len(v) > 1}
    src = fab.fabric_id.map(fmeta.source).map(norm)
    srcs = {b: int(src[fab.band == b].replace("", np.nan).nunique()) for b in CLASSES}
    share = {b: float(src[fab.band == b].value_counts(normalize=True).max()) if (fab.band == b).any() else 1.0 for b in CLASSES}
    r4 = (not (spans > 1).any() and not shared and min(srcs.values()) >= G["R4_min_distinct_sources_per_band"]
          and max(share.values()) <= G["R4_max_single_source_share_per_band"])
    res["R4"] = {"status": "PASS" if r4 else "FAIL", "groups_spanning_bands": sorted(spans[spans > 1].index),
                 "identifiers_shared_across_groups": shared, "distinct_sources_per_band": srcs,
                 "max_single_source_share_per_band": share, "provenance_groups": int(fab.provenance_id.nunique())}
    # R5
    fam = fab.fabric_id.map(fmeta.colour).map(colour_family)
    nfam = {b: int(fam[(fab.band == b) & (fam != "unknown")].nunique()) for b in CLASSES}
    weave = pd.crosstab(fab.band, fab.fabric_id.map(fmeta.weave_or_knit).replace("", "unrecorded")).to_dict(orient="index")
    res["R5"] = {"status": "PASS" if min(nfam.values()) >= G["R5_min_colour_families_per_band"] else "FAIL",
                 "colour_families_per_band": nfam, "weave_or_knit_by_band": weave}
    # R6
    vs, missing = {}, []
    for field in G["R6_confound_fields"]:
        col = man[field].fillna("") if field in man else pd.Series("", index=man.index)
        if (col == "").all():
            missing.append(field)
            continue
        vs[field] = cramers_v(man.band, col)
    meta_ba = grouped_cv_metadata_balanced_accuracy(man, fabrics, fab.set_index("fabric_id").provenance_id)
    chance = 1 / len(CLASSES)
    r6 = (not missing and all(v < G["R6_max_cramers_v"] for v in vs.values()) and meta_ba is not None
          and meta_ba <= chance + G["R6_metadata_baseline_max_margin_over_chance"])
    res["R6"] = {"status": "PASS" if r6 else "FAIL", "cramers_v": vs, "fields_not_recorded": missing,
                 "metadata_baseline_balanced_accuracy": meta_ba, "chance": chance}
    # R7
    d7 = G["R7_boundary_distance_pct_points"]
    flags = boundary_flags(fab.cotton, fab.band, d7)
    gapf = gap_edge_proximity(fab.cotton, fab.band, d7)
    bshare = {b: float(flags[fab.band == b].mean()) if (fab.band == b).any() else 0.0 for b in CLASSES}
    gshare = {b: float(gapf[fab.band == b].mean()) if (fab.band == b).any() else 0.0 for b in CLASSES}
    res["R7"] = {"status": "PASS" if max(bshare.values()) <= G["R7_max_boundary_share_per_band"] else "FAIL",
                 "definition": "adjacent_band_distance (protocol v1.1)", "distance_pct_points": d7,
                 "boundary_share_per_band": bshare, "fabrics_at_adjacent_band_risk": sorted(fab.fabric_id[flags]),
                 "gap_edge_proximity_share_per_band (informational, not gated)": gshare}
    # R8: flag cross-group look-alike pairs for HUMAN review (never merged, never declared identical)
    root = image_root(snap)
    paths = [root / pth for pth in man.image_path]
    pid = man.fabric_id.map(fab.set_index("fabric_id").provenance_id).to_numpy()
    emb_info = embeddings.describe()
    cand = lookalike_candidates(paths, pid, man.image_id.tolist(), man.fabric_id.tolist(),
                                G["R8_lookalike_similarity_threshold"], G.get("R8_embedding_similarity_threshold"))
    cols = ["image_a", "image_b", "fabric_a", "fabric_b", "colour_structure_similarity", "embedding_similarity", "reasons"]
    cdf = pd.DataFrame(cand["pairs"], columns=cols)
    cdf.to_csv(out_dir / "lookalike_candidates.csv", index=False)
    rev_file = out_dir / "lookalike_review.csv"
    rev = pd.read_csv(rev_file, dtype=str, keep_default_na=False) if rev_file.is_file() else pd.DataFrame()
    decided = review_decisions(rev)
    undecided = [c for c in cand["pairs"] if decision_for(c, decided) is None]
    same = [c for c in cand["pairs"] if decision_for(c, decided) == "same"]
    res["R8"] = {"status": "PASS" if not undecided and not same else "FAIL", "candidate_pairs": len(cand["pairs"]),
                 "flagged_by_colour_structure": cand["n_colour"], "flagged_by_embedding": cand["n_embedding"],
                 "undecided": len(undecided), "decided_same_fabric": len(same),
                 "colour_structure_threshold": G["R8_lookalike_similarity_threshold"],
                 "embedding_threshold": G.get("R8_embedding_similarity_threshold"),
                 "embedding_check": emb_info,
                 "policy": "flags only trigger human review; no fabric IDs are merged and no pair is declared identical "
                           "automatically. A 'same' decision requires assigning one provenance_group in Phase 4."}
    # R9
    vs_ = verify_snapshot(snap)
    lb = verify_lockbox(snap)
    res["R9"] = {"status": "PASS" if vs_["ok"] and lb["ok"] else "FAIL", "snapshot_verified": vs_["ok"],
                 "snapshot_problems": vs_["problems"][:5], "lockbox_sealed": lb["ok"], "lockbox_problem": lb.get("problem", "")}

    statuses = {k: v["status"] for k, v in res.items()}
    others_pass = all(v == "PASS" for k, v in statuses.items() if k != "R1")
    overall = ("READY" if others_pass and statuses["R1"] == "PASS" else
               "PILOT_ONLY" if others_pass and statuses["R1"] == "PILOT" else "NOT_READY")
    report = {"snapshot_version": meta["snapshot_version"], "synthetic": meta["synthetic"], "overall": overall,
              "gates": res, "gate_status": statuses}
    (out_dir / "readiness.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    lines = [f"# Readiness report - snapshot {meta['snapshot_version']}", ""]
    if meta["synthetic"]:
        lines += ["> **SYNTHETIC SNAPSHOT - pipeline test only. Not a real dataset.**", ""]
    lines += [f"**Overall: {overall}**", "", "| Gate | Status | Key values |", "|---|---|---|"]
    keyvals = {"R1": res["R1"]["verified_fabrics_per_band"], "R2": {"errors": n_err, "policy": pol},
               "R3": {"fabrics_failing": res["R3"]["n_failing"]}, "R4": {"sources": srcs, "shared_ids": len(shared)},
               "R5": nfam, "R6": {"cramers_v": {k: round(v, 3) for k, v in vs.items()}, "metadata_bal_acc": meta_ba,
                                   "missing": missing},
               "R7": {k: round(v, 2) for k, v in bshare.items()},
               "R8": {"candidates": len(cand["pairs"]), "undecided": len(undecided),
                      "embedding_check": "available" if emb_info["available"] else "unavailable"},
               "R9": {"snapshot": vs_["ok"], "lockbox": lb["ok"]}}
    for k in res:
        lines.append(f"| {k} | {statuses[k]} | `{json.dumps(keyvals[k], default=str)}` |")
    (out_dir / "READINESS_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report
