# Phase 3B — Stage A: dataset-wide physical-fabric identity audit

Audit of CottonFabricImageBD for physical fabrics that appear under several cotton classes or bands, and
construction of a corrected four-class dataset with leakage-free five-fold assignments. **No model was trained
in Stage A.** Findings and the open decisions are in [`reports/STAGE_A_AUDIT_REPORT.md`](reports/STAGE_A_AUDIT_REPORT.md).

## Contents

| Path | What |
|---|---|
| `reports/STAGE_A_AUDIT_REPORT.md` | Method, findings, decisions needed before Stage B |
| `audit/complete_image_inventory.csv` | Every dataset file (1,300 Original + 27,305 Augmented): SHA-256, size, dimensions, perceptual hash, prior Phase 1/2/3 membership |
| `audit/global_fabric_mapping.csv` | Every original image → photograph → global physical-fabric ID, identity status, evidence, confidence |
| `audit/physical_fabric_groups.csv` | One row per fabric ID / unresolved group: labels, bands, identity status, label consistency, eligibility |
| `audit/contradictory_fabrics.csv` | Fabrics filed under more than one cotton band, with all image paths |
| `audit/uncertain_fabric_matches.csv` | POSSIBLE and UNCERTAIN groups and why they were not confirmed |
| `audit/identity_review_decisions.json` | The manual review decisions (the human input to the audit) |
| `audit/photo_families.csv`, `family_summary.csv`, `cross_family_cross_band_candidates.csv` | Automated shortlisting used for the review |
| `audit/kept_vs_nearest_other_band.csv` | Safety check: every kept photo vs its most similar photo in another band |
| `audit/phase3_provisional_14_status.csv` | Status of the 14 images provisionally recovered in the earlier Phase 3 audit |
| `audit/preservation/` | File list and SHA-256 of all pre-existing artifacts at the start of Stage A |
| `audit/contact_sheets/` | **Not committed** — evidence and review contact sheets (dataset thumbnails); regenerate locally, see "Data and licensing" |
| `mappings/image_to_photograph.csv` | Byte-identical / near-identical file → photograph mapping |
| `datasets/corrected_dataset_manifest.csv` | **Strict** corrected dataset (primary): 133 photographs, 54 fabrics |
| `datasets/corrected_dataset_manifest_ALT_family_safe.csv` | Family-safe alternative: 236 photographs, 73 fabrics |
| `datasets/dataset_cleaning_report.md` | Counts and exclusion reasons |
| `excluded/` | Every excluded photograph / file with its reason |
| `splits/five_fold_assignments*.csv` | Proposed fold per photograph (grouped by fabric, stratified by band, seed 42) |
| `splits/fold_distribution_report*.csv`, `fold_integrity_report.md`, `integrity_checks.json` | Fold balance and leakage checks |
| `scripts/` | Code that produced everything above |

## Data and licensing

The dataset images are **not** in this repository. Place a copy at:

```
datasets/compositions/CottonFabricImageBD/
├── Original/{30,40,50,53,58,60,63,65,66,80,95,98,99}cotton/original_1001.jpg … original_1100.jpg
├── Augmented/<same 13 folders>/augmented_<n>.jpg
└── read_me.pdf
```

The dataset is the public *CottonFabricImageBD* collection (13 cotton-percentage classes; see its
`read_me.pdf`). Its download location and licence are not recorded in this repository: add them here
before sharing outside the team.

Verify a local copy against the SHA-256 recorded for every file:

```
python experiments/phase3_global_fabric_audit/scripts/verify_dataset.py            # all 28,605 files
python experiments/phase3_global_fabric_audit/scripts/verify_dataset.py --original-only
```

All manifests use project-relative paths (`datasets**Contact sheets** (`audit/contact_sheets/`, including the evidence sheets cited by the reports) contain
thumbnails of dataset photographs and are **not committed**, because the dataset licence for redistributing
derivatives is unconfirmed. Regenerate them locally with `stage_a.py pages`, `stage_a.py crosspairs` and
`stage_a_sheets.py`. Also not committed (regenerable): embedding arrays (`*.npy`) and logs.

 the embedding arrays (`*.npy`);
* logs.

## Reproduce Stage A

From the project root, with the dataset in place and the project venv:

```
python experiments/phase3_global_fabric_audit/scripts/stage_a.py inventory
python experiments/phase3_global_fabric_audit/scripts/stage_a.py photos
python experiments/phase3_global_fabric_audit/scripts/stage_a.py families 0.30   # needs torchvision weights; GPU optional
python experiments/phase3_global_fabric_audit/scripts/stage_a.py pages            # review sheets
python experiments/phase3_global_fabric_audit/scripts/stage_a.py crosspairs
# manual step: review the sheets and edit audit/identity_review_decisions.json
python experiments/phase3_global_fabric_audit/scripts/stage_a_apply.py
python experiments/phase3_global_fabric_audit/scripts/stage_a_sheets.py
```

* `inventory` reads the Phase 1/2 split files in `processed/composition/` and the Phase 3 provisional CSV
  in `experiments/phase3_recovered_data/audit/` to record prior membership.
* Family clustering uses neural embeddings. Re-running `families` on different hardware can change family
  boundaries slightly and would require re-reviewing.
* The committed `audit/photo_families.csv` and `identity_review_decisions.json` are the authoritative record
  of what was reviewed. `stage_a_apply.py` regenerates every downstream file from them deterministically.
