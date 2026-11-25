# Data dictionary: Phase 4 verified fabric dataset

All registries are UTF-8 CSV files in `metadata/`, created empty with these headers by
`data_collection/phase4/registry.py:init_dataset`. Paths are relative to `datasets/phase4_verified/`.
Dates use ISO format (`YYYY-MM-DD`). Percentages are numbers 0–100 without `%`.

## `metadata/fabric_registry.csv`: one row per physical fabric

| Column | Required | Description |
|---|---|---|
| `fabric_id` | yes | `BWF-0001` format, unique, assigned by the app; written on the physical sample tag |
| `cotton_percentage` | for verification | cotton share of the whole fabric |
| `other_fibre_percentage` | for verification | 100 − cotton (validated: must sum to 100 ± 0.5) |
| `fabric_type` | recommended | e.g. poplin, twill, denim, jersey, rib |
| `weave_or_knit` | recommended | `woven` / `knit` / `non_woven` / `unknown` |
| `colour` | recommended | dominant colour(s) |
| `source` | recommended | supplier/donor and lot, for provenance |
| `verification_method` | informational | claimed evidence type (one of the `evidence_type` values); the status is computed from `verification_records.csv` |
| `verification_document` | optional | path to a key document; must exist if given |
| `collection_date` | yes (auto) | date the sample was collected |
| `notes` | optional | free text |
| `other_fibre_composition` | recommended | extension: `fibre:percent` list, e.g. `polyester:30\|elastane:5`; must sum to `other_fibre_percentage`; must not list cotton |
| `physical_sample_location` | recommended | extension: where the physical swatch is stored, so it can be re-tested |

Example: `BWF-0001,65,35,poplin,woven,navy,Supplier A lot 7,laboratory_test,,2026-11-25,,polyester:35,Box 1 / 12`

## `metadata/image_registry.csv`: one row per photograph

| Column | Required | Description |
|---|---|---|
| `image_id` | yes | `BWI-000001`, unique, assigned by the app |
| `fabric_id` | yes | must exist in the fabric registry |
| `image_path` | yes | `images/<fabric_id>/<image_id>.<ext>` |
| `capture_device` | recommended | phone/camera model and lens |
| `image_type` | yes | `full_fabric`, `macro_texture`, `natural_light`, `alternate_angle`, `controlled` |
| `lighting` | yes | `natural_daylight`, `controlled_led_5000k`, `controlled_other`, `indoor_mixed`, `unknown` |
| `camera_angle` | yes | `overhead_90`, `angled_45`, `angled_other`, `unknown` |
| `resolution` | auto | `WIDTHxHEIGHT` of the stored file |
| `capture_date` | yes | date taken |
| `sha256` | auto | SHA-256 of the stored file; duplicates are rejected |
| `notes` | optional | extension |

## `metadata/verification_records.csv`: evidence for composition labels

| Column | Required | Description |
|---|---|---|
| `record_id` | yes | `BWV-00001`, assigned by the app |
| `fabric_id` | yes | fabric the evidence refers to |
| `evidence_type` | yes | accepted: `laboratory_test`, `manufacturer_documentation`, `supplier_documentation`; recorded only: `garment_care_label`, `seller_listing`, `visual_estimate`, `none` |
| `evidence_reference` | for acceptance | report or document number |
| `issuing_party` | recommended | laboratory, manufacturer or supplier |
| `product_identifier` | for documentation | article/style/lot number identifying the exact product |
| `document_path` | for acceptance | uploaded file under `metadata/verification_documents/<fabric_id>/` |
| `reported_cotton_percentage` | for acceptance | cotton percentage stated by the evidence |
| `reported_other_fibres` | recommended | e.g. `polyester:35` |
| `test_method_standard` | for lab tests | e.g. ISO 1833-11, AATCC 20A |
| `reviewed_by`, `review_date` | for acceptance | who checked the evidence against the physical sample |
| `review_decision` | yes | `pending` (default), `accepted`, `rejected` |
| `notes` | optional | |

## Generated files (do not edit)

| File | Content |
|---|---|
| `metadata/manifests/verified_training_manifest.csv` | the **only** file training may use: images of VERIFIED, in-band, error-free fabrics, with `fabric_id` (grouping key), `cotton_percentage`, `band`, `verification_records` |

`band` values (inclusive, from the verified cotton percentage): `low` 30–40 %, `medium` 50–66 %, `high` 80–95 %,
`very_high` 98–100 % (100 % cotton included). Verified values in the gaps (e.g. 45 %, 70 %, 97 %) are reported as
`out_of_band` and are not in the manifest. Target: 30 independent physical fabrics per band.
| `metadata/manifests/fabric_status.csv` | computed status and reasons for every fabric |
| `documentation/audit/dataset_audit_report.md`, `audit_results.json`, `issues.csv` | validator output |
