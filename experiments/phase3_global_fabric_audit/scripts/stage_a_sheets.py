"""Phase 3B Stage A: contact sheets for review of the applied identity audit.

    python experiments/phase3_global_fabric_audit/scripts/stage_a_sheets.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage_a import AUDIT, DATA, SHEETS, draw_tiles  # noqa: E402

OUT = SHEETS / "results"


def photo_reps() -> pd.DataFrame:
    m = pd.read_csv(AUDIT / "global_fabric_mapping.csv")
    reps = m.sort_values("original_path").groupby("photo_id").agg(
        path=("original_path", "first"), classes=("photo_classes", "first"), band=("cotton_band", "first"),
        fabric=("global_fabric_id", "first"), identity=("identity_status", "first"),
        label_status=("label_consistency_status", "first"), elig=("eligibility_for_training", "first")).reset_index()
    bands = m.groupby("photo_id").cotton_band.agg(lambda s: "|".join(sorted(set(s))))
    reps["bands"] = reps.photo_id.map(bands)
    return reps


def fabric_sheet(reps, groups, title, fname, max_fabrics=14, per=7):
    items = []
    for f in groups[:max_fabrics]:
        g = reps[reps.fabric == f].sort_values("classes")
        sel = pd.concat([g.drop_duplicates("classes"), g]).drop_duplicates("photo_id").head(per)
        for r in sel.itertuples():
            items.append((r.path, f"{f[-4:]} {r.photo_id[2:]} {r.classes.replace('cotton', '')}", r.bands))
        items += [None] * ((-len(items)) % per)
    items = [i for i in items if i is not None] if not items else items
    clean = [i if i is not None else (reps.path.iloc[0], "", "low_cotton") for i in items]
    # blank padding tiles are drawn as white
    draw_tiles([i for i in clean], per, 150, title, OUT / fname)
    return len(groups)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    reps = photo_reps()
    groups = pd.read_csv(AUDIT / "physical_fabric_groups.csv")
    multi_class = groups[(groups.original_labels.str.contains("|", regex=False)) &
                         (~groups.global_fabric_id.str.startswith("UNRESOLVED"))]
    cross_band = groups[groups.label_consistency_status == "CONTRADICTORY_BANDS"].sort_values("image_count", ascending=False)
    cross_class_same_band = multi_class[multi_class.label_consistency_status == "CONSISTENT_SINGLE_BAND"].sort_values("image_count", ascending=False)
    possible = groups[groups.identity_status == "POSSIBLE_MATCH"].sort_values("image_count", ascending=False)
    uncertain = groups[groups.identity_status == "UNCERTAIN"].sort_values("image_count", ascending=False)

    def sheets(gs, stem, title):
        ids = gs.global_fabric_id.tolist()
        for k in range(0, len(ids), 12):
            items = []
            for f in ids[k:k + 12]:
                g = reps[reps.fabric == f]
                sel = pd.concat([g.drop_duplicates("classes"), g]).drop_duplicates("photo_id").head(8)
                items += [(r.path, f"{f.replace('FABRIC_', 'F').replace('UNRESOLVED_', 'U-')} {r.photo_id[2:]} {r.classes.replace('cotton', '')}",
                           r.bands) for r in sel.itertuples()]
            draw_tiles(items, 8, 140, f"{title} ({k + 1}-{min(k + 12, len(ids))} of {len(ids)}); caption: fabric id, photo, classes",
                       OUT / f"{stem}_{k // 12 + 1:02d}.jpg")

    sheets(cross_band, "confirmed_cross_band_contradictions", "Contradictory fabrics: one physical fabric under several bands")
    sheets(cross_class_same_band, "confirmed_cross_class_same_band", "Same fabric under several classes, one band")
    sheets(possible, "possible_matches", "POSSIBLE_MATCH groups (colour/structure only)")
    sheets(uncertain.head(24), "uncertain_unresolved", "UNRESOLVED remainders (largest 24)")

    # previously identified problematic fabrics
    m = pd.read_csv(AUDIT / "global_fabric_mapping.csv")
    named = {"flower_pink": "pink flower-watermark fabric", "red_slub_plainweave": "red fabric with white slub lines",
             "chevron_knit_garment": "chevron knit garment"}
    items = []
    for lab, desc in named.items():
        f = groups.loc[groups.unit_label == lab, "global_fabric_id"].iloc[0]
        g = reps[reps.fabric == f]
        sel = pd.concat([g.drop_duplicates("classes"), g]).drop_duplicates("photo_id").head(8)
        items += [(r.path, f"{desc[:14]} {r.photo_id[2:]} {r.classes.replace('cotton', '')}", r.bands) for r in sel.itertuples()]
    draw_tiles(items, 8, 150, "Previously identified problematic fabrics: flower pink (58%+80%), red slub (30%+65%+80%), chevron knit (65%+66%)",
               OUT / "previously_identified_problem_fabrics.jpg")

    # safety check: every kept (family-safe) photograph vs its most similar photograph in ANOTHER band
    emb = np.load(AUDIT / "photo_embeddings_imagenet.npy"); col = np.load(AUDIT / "photo_color_hist.npy")
    fam_t = pd.read_csv(AUDIT / "photo_families.csv")
    S = 0.7 * (emb @ emb.T) + 0.3 * (col @ col.T)
    idx = {p: i for i, p in enumerate(fam_t.photo_id)}
    rb = reps.set_index("photo_id")
    kept = pd.read_csv(DATA / "corrected_dataset_manifest_ALT_family_safe.csv")
    rows = []
    for p in kept.photo_id:
        i = idx[p]
        own = set(rb.bands[p].split("|"))
        best, bj = -1, None
        for q, j in idx.items():
            if not (set(rb.bands[q].split("|")) & own) and S[i, j] > best:
                best, bj = S[i, j], q
        rows.append({"photo_id": p, "fabric": rb.fabric[p], "band": rb.band[p], "nearest_other_band_photo": bj,
                     "nearest_other_band": rb.bands[bj], "similarity": round(float(best), 3)})
    chk = pd.DataFrame(rows).sort_values("similarity", ascending=False)
    chk.to_csv(AUDIT / "kept_vs_nearest_other_band.csv", index=False)
    items = []
    for r in chk.head(24).itertuples():
        items += [(rb.path[r.photo_id], f"KEPT {r.photo_id[2:]} {rb.classes[r.photo_id].replace('cotton', '')}", rb.bands[r.photo_id]),
                  (rb.path[r.nearest_other_band_photo], f"{r.nearest_other_band_photo[2:]} {rb.classes[r.nearest_other_band_photo].replace('cotton', '')} s{r.similarity}",
                   rb.bands[r.nearest_other_band_photo])]
    draw_tiles(items, 8, 140, "Safety check: kept photographs (left) vs most similar photograph in another band (right), top 24",
               OUT / "safety_check_kept_vs_other_band.jpg")
    print(chk.head(24).to_string())
    print(sorted(p.name for p in OUT.glob("*.jpg")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
