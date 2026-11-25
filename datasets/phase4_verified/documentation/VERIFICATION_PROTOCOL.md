# Verification protocol: composition labels

A composition label is only as good as the evidence behind it. CottonFabricImageBD showed what happens
without this: the same physical fabric was filed under several cotton percentages, and 27 fabrics carried
labels from different composition bands. This protocol makes every label traceable to evidence, and keeps
unverified labels out of training.

## 1. One physical fabric = one record = one label
* Every physical piece of fabric gets one `fabric_id` (`BWF-0001`, …), written on a tag attached to the
  physical sample.
* Two pieces cut from the **same bolt or roll** are the **same fabric**: register them once.
* Pieces of the same product in a different colourway or lot are different fabrics only if they come from
  different bolts *and* have their own evidence.
* Keep the physical sample (`physical_sample_location`) so it can be re-tested.

## 2. Accepted evidence (can verify a label)

| `evidence_type` | Requirements (all must be met) |
|---|---|
| `laboratory_test` | Fibre-composition report from an independent or accredited laboratory. Record the report number (`evidence_reference`), the laboratory (`issuing_party`) and the method (`test_method_standard`, e.g. ISO 1833-x, AATCC 20A). The report file must be uploaded. |
| `manufacturer_documentation` | Technical data sheet, certificate or test certificate from the fabric manufacturer that **names this exact product** (`product_identifier`: article/style/lot number) and states the composition. File uploaded. |
| `supplier_documentation` | Invoice, specification or delivery note from the supplier that **identifies this exact product or lot** (`product_identifier`) and states the composition. File uploaded. |

For every accepted record:
* a reviewer (`reviewed_by`, `review_date`) has checked that the document belongs to this physical fabric
  (product number, lot, photo of the bolt tag);
* the reviewer sets `review_decision = accepted`.

### Verification policy v2: independent second review
Under policy v2 (required for Phase 5 training), an accepted record also needs a **second review** by a different person
(`second_reviewed_by`, `second_review_decision = accepted`). Same-person second reviews are errors. A second-review
rejection makes the record unusable. Legacy datasets stay on policy v1 until migrated explicitly; the dry-run migration
lists every fabric that would move from VERIFIED to PENDING_VERIFICATION.

### Related fabrics and product identifiers
Fabrics cut from the same bolt, product or garment must share a `provenance_group`. The validator warns when a
product identifier (`source_product_id` or evidence `product_identifier`) appears on several fabrics without a common
group, and reports an ERROR if such fabrics have different verified compositions.

## 3. Recorded but never sufficient

| `evidence_type` | Why not |
|---|---|
| `garment_care_label` | Garment labels are frequently wrong, refer to the main body only, or are missing after alteration. Record them as a hint; **never** as verification. |
| `seller_listing` | Online listings and verbal claims cannot be traced to the physical item. |
| `visual_estimate` | That is exactly what the model is supposed to learn; it cannot be the label. |
| `none` | Placeholder. |

## 4. How the status is computed
Status is computed by `data_collection/phase4/registry.py` and **cannot be typed in**.

| Status | Rule | In training manifest? |
|---|---|---|
| `VERIFIED` | at least one accepted record meeting §2; all accepted records agree within ±3 percentage points; the registry cotton value agrees within ±3 points | **yes** (if in a task band and error-free) |
| `PENDING_VERIFICATION` | no accepted evidence yet (including care-label-only fabrics) | no |
| `CONFLICTING_EVIDENCE` | accepted records disagree with each other or with the registry value by more than 3 points | no; resolve by re-testing |
| `REJECTED` | all evidence was reviewed and rejected | no |
| `INCOMPLETE_RECORD` | cotton percentage missing or invalid | no |

The ±3-point tolerance reflects typical laboratory reproducibility for blend analysis. Re-test the sample if
evidence disagrees by more.

## 5. Bands
* Low 30–40 %, Medium 50–66 %, High 80–95 %, Very high 98–100 % (inclusive). **100 % cotton is a valid Very high
  sample**, but like every other value it must be verified by accepted evidence. A claimed 100 % is never verified
  just because it is 100 %, and "100 % cotton" garment labels are a common source of error.
* Verified fabrics outside these ranges (e.g. 45 %, 70 %, 97 %) are kept and reported as `out_of_band`, but are not
  in the band manifest. Decide later whether to extend the bands.
* Never move a fabric into a band by rounding or by majority vote.

## 6. Reviewer checklist
1. Does the document name the product/lot of *this* sample? Compare tags and photos.
2. Is the composition stated for the whole fabric (not a trim or lining)?
3. Lab tests: report number, laboratory, method and date present?
4. Does the reported cotton percentage match the registry entry (±3 points)?
5. Is the same physical fabric already registered under another ID? (Check the validator's
   near-duplicate warnings and the physical sample store.)
6. Record `reviewed_by`, `review_date` and the decision.

## 7. Audit
* Run `python -m data_collection.phase4.validate` after every collection session, and before any training.
* Training must use only `metadata/manifests/verified_training_manifest.csv`, with `fabric_id` as the grouping
  key for every split.
