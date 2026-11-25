# Corrected four-class dataset: cleaning report (Phase 3B, Stage A)

Manifest-based: no image was copied or modified. Every row points to an original file in
`datasets/compositions/CottonFabricImageBD/Original/`. One row per **photograph**: byte-identical or
near-identical copies of the same photograph are collapsed to one file, and the other file paths are
recorded in `../audit/global_fabric_mapping.csv`.

| Quantity | Value |
|---|---|
| Original image files (`Original/`) | 1,300 |
| Augmented files (inventoried only; no source mapping, never used) | 27,305 |
| Unique SHA-256 | 889 |
| Unique photographs (after near-identical re-encodings, MAD ≤ 6/255 and dHash ≤ 8) | 883 |
| Photographs filed under more than one class / more than one band | 242 / 24 |
| Visual families reviewed (all classes together) | 150 |
| Global physical-fabric IDs assigned | 100 |
| — CONFIRMED_MATCH / POSSIBLE_MATCH / DISTINCT_FABRIC | 60 / 21 / 19 |
| Unresolved remainders (photographs with no confirmed identity; no fabric ID) | 77 groups, 447 photographs, 601 files |
| Fabrics with consistent band labels | 73 |
| Fabrics with contradictory bands | 27 (339 image files) |

## Two eligibility policies

| | **Strict (primary: `corrected_dataset_manifest.csv`)** | Family-safe (alternative: `corrected_dataset_manifest_ALT_family_safe.csv`) |
|---|---|---|
| Rule | identity CONFIRMED_MATCH or DISTINCT_FABRIC, and single band | additionally POSSIBLE_MATCH groups, if single band |
| Photographs | **133** | 236 |
| Physical fabrics | **54** | 73 |
| Low (30–40 %) | 4 photos / **2 fabrics** | 13 photos / **3 fabrics** |
| Medium (50–66 %) | 59 / 24 | 126 / 38 |
| High (80–95 %) | 56 / 20 | 70 / 23 |
| Very high (98–99 %) | 14 / 8 | 27 / 9 |
| Excluded original files | 1,116 of 1,300 | 1,004 of 1,300 |
| Previously "label-conflict" photographs now included (band-consistent copies) | 24 | 79 |

Why the family-safe alternative exists: a POSSIBLE_MATCH group is photographs of the same colour and
structure, all filed under one band, where no specific mark proves a single fabric. For training these are
label-safe at band level and are kept together in one fold, so they cannot leak. The brief asks to exclude
unconfirmed identities, so strict is the primary manifest. The alternative is provided for your decision.

## Exclusion reasons (strict)
* **Contradictory bands (27 fabrics, 339 files):** one physical fabric under several bands. 14 of these are
  proven by byte-identical photographs filed under different bands. The others rest on distinctive matching
  structure (flower watermark, slub lines, identical meshes and weaves). See `../audit/contradictory_fabrics.csv`.
* **Unresolved identity (447 photographs):** visually inseparable from fabrics filed under other bands (warm
  plain weaves, grey denims/twills, plain greys, teal/olive/mustard meshes) or generic in appearance. Each
  group's reason: `../audit/uncertain_fabric_matches.csv`.
* **POSSIBLE_MATCH only (21 fabrics):** strict policy only.
* Six groups were moved to unresolved after the final safety check (kept photographs vs their most similar
  photograph in another band): `../audit/contact_sheets/results/safety_check_kept_vs_other_band.jpg`.

Full per-photograph list with reasons: `../excluded/excluded_photographs_strict.csv`; per file:
`../excluded/excluded_images_all_files.csv`.

## The 14 Phase-3 "recoverable" images
The global audit covers every photograph, so these are already classified:
* 6 are eligible under the strict policy (star-motif grey FABRIC_0050, and three 80 % fabrics).
* 8 are eligible only under the family-safe policy (98 % flecked tweed FABRIC_0028 and two 60 % groups).

A separate "expanded" dataset is therefore not meaningful any more; see the Stage A report.
