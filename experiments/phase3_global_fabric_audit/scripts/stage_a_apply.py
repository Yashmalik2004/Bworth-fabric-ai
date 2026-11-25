"""Phase 3B Stage A: apply the manual identity review -> mapping, groups, reports, dataset, folds.

    python experiments/phase3_global_fabric_audit/scripts/stage_a_apply.py

Reads audit/identity_review_decisions.json (manual review) plus the outputs of stage_a.py.
Writes only inside experiments/phase3_global_fabric_audit/. No training.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage_a import AUDIT, DATA, EXCL, MAPS, REPORTS, SEED, SPLITS  # noqa: E402

CONF = {"CONFIRMED_MATCH": "high", "DISTINCT_FABRIC": "medium", "POSSIBLE_MATCH": "medium", "UNCERTAIN": "low"}


def sort_classes(classes) -> str:
    return "|".join(sorted(set(classes), key=lambda c: int(c[:-6])))


def build_units() -> tuple[pd.DataFrame, list[str]]:
    t = pd.read_csv(AUDIT / "photo_families.csv")
    dec = json.loads((AUDIT / "identity_review_decisions.json").read_text())["families"]
    errors = []
    missing = sorted(set(t.family) - set(dec))
    extra = sorted(set(dec) - set(t.family))
    if missing:
        errors.append(f"families without a decision: {missing}")
    if extra:
        errors.append(f"decisions for unknown families: {extra}")
    rows = []
    for fam, g in t.groupby("family"):
        d = dec.get(fam, {"units": [], "rest": "no decision"})
        ids = {p[2:]: p for p in g.photo_id}
        assigned = {}
        for u in d["units"]:
            members = list(ids) if u["photos"] == ["*"] else u["photos"]
            for p in members:
                if p not in ids:
                    errors.append(f"{fam}: photo {p} not in family")
                elif p in assigned:
                    errors.append(f"{fam}: photo {p} in two units")
                else:
                    assigned[p] = u
        for p, pid in ids.items():
            u = assigned.get(p)
            if u:
                rows.append({"photo_id": pid, "family": fam, "unit_label": u["label"], "identity_status": u["identity"],
                             "identity_evidence": u["evidence"], "reviewer_notes": ""})
            else:
                rows.append({"photo_id": pid, "family": fam, "unit_label": f"UNRESOLVED_{fam}",
                             "identity_status": "UNCERTAIN", "identity_evidence": "",
                             "reviewer_notes": d.get("rest", "")})
    return pd.DataFrame(rows).merge(t, on=["photo_id", "family"]), errors


def fabric_identity(statuses) -> str:
    s = set(statuses)
    if "UNCERTAIN" in s:
        return "UNCERTAIN"
    if "CONFIRMED_MATCH" in s:
        return "CONFIRMED_MATCH"
    if s == {"DISTINCT_FABRIC"}:
        return "DISTINCT_FABRIC"
    return "POSSIBLE_MATCH"


def main() -> int:
    from sklearn.model_selection import StratifiedGroupKFold

    units, errors = build_units()
    if errors:
        print("STOP: decision errors:\n  " + "\n  ".join(errors))
        return 2
    imgs = pd.read_csv(MAPS / "image_to_photograph.csv")
    inv = pd.read_csv(AUDIT / "complete_image_inventory.csv")

    # ---- global fabric ids (units with the same label are one fabric) ------ #
    labels = sorted(l for l in units.unit_label.unique() if not l.startswith("UNRESOLVED_"))
    fid = {l: f"FABRIC_{k + 1:04d}" for k, l in enumerate(labels)}
    units["global_fabric_id"] = units.unit_label.map(lambda l: fid.get(l, l))
    units["band_list"] = units.bands.str.split("|")      # every band any copy of the photo is filed under

    img_fab = imgs.merge(units[["photo_id", "global_fabric_id"]], on="photo_id")
    g = units.groupby("global_fabric_id")
    groups = pd.DataFrame({
        "image_count": img_fab.groupby("global_fabric_id").size(),
        "photo_count": g.size(),
        "original_labels": g.classes.agg(lambda s: sort_classes("|".join(s).split("|"))),
        "original_bands": g.band_list.agg(lambda s: "|".join(sorted({b for x in s for b in x}))),
        "families": g.family.agg(lambda s: "|".join(sorted(set(s)))),
        "unit_label": g.unit_label.first(),
        "identity_status": g.identity_status.agg(fabric_identity),
    })
    groups["n_bands"] = groups.original_bands.str.count(r"\|") + 1
    groups["label_consistency_status"] = np.select(
        [groups.identity_status == "UNCERTAIN", groups.n_bands > 1],
        ["INSUFFICIENT_EVIDENCE", "CONTRADICTORY_BANDS"], "CONSISTENT_SINGLE_BAND")
    consistent = groups.label_consistency_status == "CONSISTENT_SINGLE_BAND"
    strict_ok = consistent & groups.identity_status.isin(["CONFIRMED_MATCH", "DISTINCT_FABRIC"])
    safe_ok = consistent & groups.identity_status.isin(["CONFIRMED_MATCH", "DISTINCT_FABRIC", "POSSIBLE_MATCH"])
    groups["eligibility_for_training"] = np.where(
        strict_ok, "ELIGIBLE_STRICT", np.where(safe_ok, "ELIGIBLE_ONLY_UNDER_FAMILY_SAFE_POLICY", "EXCLUDED"))
    groups["exclusion_reason"] = np.select(
        [groups.label_consistency_status == "CONTRADICTORY_BANDS", groups.identity_status == "UNCERTAIN",
         groups.identity_status == "POSSIBLE_MATCH"],
        ["same physical fabric carries labels from more than one cotton band",
         "physical identity unresolved: visually inseparable from fabrics filed under other bands, or generic appearance",
         "strict policy only: identity POSSIBLE (same colour/structure, no specific marks); band-consistent"], "")
    groups = groups.reset_index()
    groups[["global_fabric_id", "image_count", "photo_count", "original_labels", "original_bands", "identity_status",
            "label_consistency_status", "eligibility_for_training", "exclusion_reason", "families", "unit_label"]] \
        .to_csv(AUDIT / "physical_fabric_groups.csv", index=False)
    gi = groups.set_index("global_fabric_id")

    # ---- image-level mapping (every original image) ------------------------ #
    m = imgs.merge(units[["photo_id", "global_fabric_id", "identity_status", "identity_evidence", "reviewer_notes",
                          "family"]], on="photo_id")
    m = m.merge(inv[["image_id", "filename", "original_cotton_percentage"]], on="image_id")
    m["identity_evidence"] = [
        (f"one of {n} files of the same photograph (identical SHA-256 or near-identical re-encoding); " if n > 1 else "") + (e or "")
        for n, e in zip(m.photo_group_size, m.identity_evidence.fillna(""))]
    m["confidence"] = m.identity_status.map(CONF)
    m["label_consistency_status"] = m.global_fabric_id.map(gi.label_consistency_status)
    m["eligibility_for_training"] = m.global_fabric_id.map(gi.eligibility_for_training)
    mapping = m.rename(columns={"dhash16": "perceptual_hash"})[
        ["image_id", "filename", "original_path", "original_class", "original_cotton_percentage", "cotton_band", "sha256",
         "perceptual_hash", "global_fabric_id", "identity_status", "identity_evidence", "confidence", "reviewer_notes",
         "photo_id", "photo_group_size", "photo_classes", "family", "label_consistency_status",
         "eligibility_for_training"]].sort_values("image_id")
    mapping.to_csv(AUDIT / "global_fabric_mapping.csv", index=False)

    # ---- contradictory / uncertain reports -------------------------------- #
    con = mapping[mapping.label_consistency_status == "CONTRADICTORY_BANDS"]
    con_rep = con.groupby("global_fabric_id").agg(
        n_images=("image_id", "size"), original_labels=("original_class", sort_classes),
        bands=("cotton_band", lambda s: "|".join(sorted(set(s)))),
        image_paths=("original_path", lambda s: ";".join(s))).reset_index()
    con_rep.insert(1, "fabric_description", con_rep.global_fabric_id.map(gi.unit_label))
    con_rep["evidence"] = con_rep.global_fabric_id.map(
        units.drop_duplicates("global_fabric_id").set_index("global_fabric_id").identity_evidence)
    con_rep.to_csv(AUDIT / "contradictory_fabrics.csv", index=False)

    unc = groups[groups.identity_status.isin(["UNCERTAIN", "POSSIBLE_MATCH"])].copy()
    first = units.drop_duplicates("global_fabric_id").set_index("global_fabric_id")
    unc["photos"] = unc.global_fabric_id.map(units.groupby("global_fabric_id").photo_id.agg(";".join))
    unc["why_not_confirmed"] = [
        first.reviewer_notes[f] if s == "UNCERTAIN" else
        "same colour and structure but no specific mark (stain, motif, defect, pattern) proving one physical fabric: "
        + first.identity_evidence[f]
        for f, s in zip(unc.global_fabric_id, unc.identity_status)]
    unc[["global_fabric_id", "identity_status", "photo_count", "image_count", "original_labels", "original_bands",
         "label_consistency_status", "eligibility_for_training", "why_not_confirmed", "photos"]] \
        .to_csv(AUDIT / "uncertain_fabric_matches.csv", index=False)

    # ---- datasets: one row per photograph (identical files not duplicated) - #
    photo_rows = m.sort_values("original_path").groupby("photo_id").agg(
        image_path=("original_path", "first"), original_label=("photo_classes", "first"),
        four_class_label=("cotton_band", "first"), global_fabric_id=("global_fabric_id", "first"),
        identity_status=("identity_status", "first"), eligibility=("eligibility_for_training", "first"),
        n_files=("image_id", "size")).reset_index()
    photo_rows["source_dataset"] = "CottonFabricImageBD/Original"
    photo_rows["inclusion_reason"] = [
        f"{s} physical identity; every label of {f} maps to {b}" + (f"; {n} identical files collapsed to one" if n > 1 else "")
        for s, f, b, n in zip(photo_rows.identity_status, photo_rows.global_fabric_id, photo_rows.four_class_label,
                              photo_rows.n_files)]
    cols = ["image_path", "original_label", "four_class_label", "global_fabric_id", "source_dataset", "inclusion_reason",
            "identity_status", "photo_id"]
    strict = photo_rows[photo_rows.eligibility == "ELIGIBLE_STRICT"][cols].reset_index(drop=True)
    safe = photo_rows[photo_rows.eligibility != "EXCLUDED"][cols].reset_index(drop=True)
    strict.to_csv(DATA / "corrected_dataset_manifest.csv", index=False)
    safe.to_csv(DATA / "corrected_dataset_manifest_ALT_family_safe.csv", index=False)
    excl = photo_rows[photo_rows.eligibility != "ELIGIBLE_STRICT"].copy()
    excl["exclusion_reason"] = excl.global_fabric_id.map(gi.exclusion_reason)
    excl.to_csv(EXCL / "excluded_photographs_strict.csv", index=False)
    mapping[mapping.eligibility_for_training == "EXCLUDED"].to_csv(EXCL / "excluded_images_all_files.csv", index=False)

    # ---- five folds grouped by fabric, stratified by band ------------------- #
    emb = np.load(AUDIT / "photo_embeddings_imagenet.npy")
    col = np.load(AUDIT / "photo_color_hist.npy")
    fam_t = pd.read_csv(AUDIT / "photo_families.csv")
    S = 0.7 * (emb @ emb.T) + 0.3 * (col @ col.T)
    pidx = {p: i for i, p in enumerate(fam_t.photo_id)}
    sha_of = imgs.groupby("photo_id").sha256.agg(set)
    results = {}
    for name, ds in (("strict", strict), ("ALT_family_safe", safe)):
        nfab = ds.groupby("four_class_label").global_fabric_id.nunique()
        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
        fold = np.full(len(ds), -1)
        for k, (_, te) in enumerate(sgkf.split(ds, y=ds.four_class_label, groups=ds.global_fabric_id)):
            fold[te] = k
        ds = ds.assign(fold=fold)
        sfx = "" if name == "strict" else "_ALT_family_safe"
        ds.to_csv(SPLITS / f"five_fold_assignments{sfx}.csv", index=False)
        dist = ds.groupby(["fold", "four_class_label"]).agg(
            images=("photo_id", "size"), fabrics=("global_fabric_id", "nunique")).unstack(fill_value=0)
        dist.columns = [f"{a}_{b}" for a, b in dist.columns]
        dist["images_total"] = ds.groupby("fold").size()
        dist["fabrics_total"] = ds.groupby("fold").global_fabric_id.nunique()
        dist.to_csv(SPLITS / f"fold_distribution_report{sfx}.csv")

        sha_fold: dict = {}
        for p, f in zip(ds.photo_id, ds.fold):
            for h in sha_of[p]:
                sha_fold.setdefault(h, set()).add(f)
        ii = [pidx[p] for p in ds.photo_id]
        Ssub = S[np.ix_(ii, ii)]
        fa = ds.fold.to_numpy()
        hi = [{"photo_a": ds.photo_id[a], "fabric_a": ds.global_fabric_id[a], "fold_a": int(fa[a]),
               "photo_b": ds.photo_id[b], "fabric_b": ds.global_fabric_id[b], "fold_b": int(fa[b]),
               "similarity": round(float(Ssub[a, b]), 3)}
              for a, b in zip(*np.triu_indices(len(ds), 1)) if fa[a] != fa[b] and Ssub[a, b] >= 0.90]
        pd.DataFrame(hi).to_csv(SPLITS / f"cross_fold_high_similarity_pairs{sfx}.csv", index=False)
        results[name] = {
            "ds": ds, "nfab": nfab, "dist": dist,
            "checks": {
                "fabric_ids_shared_across_folds": int((ds.groupby("global_fabric_id").fold.nunique() > 1).sum()),
                "photographs_in_more_than_one_fold": int(ds.photo_id.duplicated().sum()),
                "sha256_in_more_than_one_fold": int(sum(len(v) > 1 for v in sha_fold.values())),
                "all_rows_assigned_a_fold": bool((ds.fold >= 0).all()),
                "every_fabric_single_band": bool(ds.groupby("global_fabric_id").four_class_label.nunique().max() == 1),
                "contradictory_or_uncertain_fabrics_included": int(ds.global_fabric_id.isin(
                    groups.loc[groups.label_consistency_status != "CONSISTENT_SINGLE_BAND", "global_fabric_id"]).sum()),
                "cross_fold_pairs_similarity_ge_0_90": len(hi),
                "min_fabrics_per_band": int(nfab.min()),
                "five_fold_feasible_(>=5_fabrics_per_band)": bool((nfab >= 5).all()),
            }}
    (SPLITS / "integrity_checks.json").write_text(json.dumps({n: r["checks"] for n, r in results.items()}, indent=2))

    # ---- Phase 3 provisional recovered images ------------------------------ #
    prov = inv[inv.phase3_provisional_status == "provisional_accept"][["image_id", "original_path"]].merge(
        mapping[["image_id", "photo_id", "global_fabric_id", "identity_status", "label_consistency_status",
                 "eligibility_for_training"]], on="image_id")
    prov.to_csv(AUDIT / "phase3_provisional_14_status.csv", index=False)

    resolved = ~groups.global_fabric_id.str.startswith("UNRESOLVED")
    summary = {
        "images_audited_original": int(len(imgs)),
        "augmented_files_inventoried_only": int((inv.subset == "Augmented").sum()),
        "unique_sha256": int(imgs.sha256.nunique()), "unique_photographs": int(imgs.photo_id.nunique()),
        "photographs_filed_under_more_than_one_class": int((imgs.drop_duplicates("photo_id").photo_classes.str.contains("|", regex=False)).sum()),
        "photographs_filed_under_more_than_one_band": int((imgs.drop_duplicates("photo_id").photo_bands.str.contains("|", regex=False)).sum()),
        "visual_families_reviewed": int(units.family.nunique()),
        "global_fabric_ids_assigned": int(resolved.sum()),
        "unresolved_remainders_(not_fabric_ids)": int((~resolved).sum()),
        "photographs_by_identity_status": units.identity_status.value_counts().to_dict(),
        "images_by_identity_status": mapping.identity_status.value_counts().to_dict(),
        "fabric_ids_by_identity_status": groups[resolved].identity_status.value_counts().to_dict(),
        "fabric_ids_by_label_status": groups[resolved].label_consistency_status.value_counts().to_dict(),
        "contradictory_band_fabrics": int((groups.label_consistency_status == "CONTRADICTORY_BANDS").sum()),
        "contradictory_band_images": int((mapping.label_consistency_status == "CONTRADICTORY_BANDS").sum()),
        "eligible": {n: {"photographs": int(len(r["ds"])), "fabrics": int(r["ds"].global_fabric_id.nunique()),
                         "photographs_per_band": r["ds"].four_class_label.value_counts().to_dict(),
                         "fabrics_per_band": r["nfab"].to_dict(),
                         "fabrics_per_fold": r["ds"].groupby("fold").global_fabric_id.nunique().to_dict(),
                         "photographs_per_fold": r["ds"].groupby("fold").size().to_dict()}
                     for n, r in results.items()},
        "excluded_strict": {"original_image_files": int(len(imgs) - imgs.photo_id.isin(strict.photo_id).sum()),
                            "photographs": int(len(photo_rows) - len(strict)),
                            "fabric_ids_and_remainders": int(len(groups) - strict.global_fabric_id.nunique())},
        "integrity": {n: r["checks"] for n, r in results.items()},
        "phase3_provisional_14": prov.eligibility_for_training.value_counts().to_dict(),
    }
    (REPORTS / "stage_a_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
