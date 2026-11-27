# NIST NIR-SORT: composition feasibility audit for BWorth

- **Date:** 27 November 2026
- **Status:** read-only analysis. No model was trained.
- **Companion file:** `nist_composition_inventory.csv`, with one row per physical swatch (94 rows).
- **Earlier reports:** `NIST_DATASET_AUDIT.md` and `SOURCE_A_COUNT_INVESTIGATION.md`.
- **Data source:** K.E. Goodge, C.J. Vederman, R.V. Patel, C.M. Wentz, Z. Tsinas, A.K. Landauer, A.L. Forster, "Near Infrared Spectra of Origin-defined and Real-world Textiles (NIR-SORT)", National Institute of Standards and Technology, Version 2.0.0, https://doi.org/10.18434/mds2-3325. The inventory CSV is derived from NIST's `SampleInformation.csv`; the NIST data itself is not redistributed in this repository.

## Verdict: SUITABLE ONLY FOR EXPLORATORY ANALYSIS

For BWorth's four-band cotton task, the dataset can't support training or a credible evaluation, with four classes or with fewer:

- Outside the very high band, the supplier-documented swatches give **1 high, 3 medium and 1 to 2 low independent materials**.
- All of those are undyed white Source B fabrics, so a model could separate them from 100% cotton by colour and source alone.
- The dataset is still useful for limited, clearly labelled exploratory work (§7).

---

## 0. Scope, sources and rules followed

**Inputs**, all read-only:
- `SampleInformation.csv`;
- `ColorParameters.csv`;
- `DataDictionary.csv`;
- `README_main.txt`, `README_stereo.txt`, `ChangeLog.txt`, `DataFramework.txt`;
- the 564 colour TIFF images.

The image statistics in §4 were computed at quarter resolution.

**Scope:**
- **94 fabric swatches**, the only entries with colour images, are analysed.
- The other 61 rows of `SampleInformation.csv` are form factor `04`, shredded fibres: the A55 to A73 cotton/polyester fibre blends and their A07 and B09 fibre references. They have no colour images and are excluded.
- NIST's polarised images of those fibres are not in the local extract.

**Rules followed:**
- No NIST file was modified.
- No Phase 4 data, registry, checkpoint, experiment or branch was touched.
- Nothing was committed.
- No composition was inferred where the metadata is blank.

**Facts and inferences.** The report keeps documented facts separate from relationships it infers:

| Tag | Meaning |
|---|---|
| **[DOC]** | Stated in a NIST file |
| **[INF]** | Inferred by this audit |
| **[MEAS]** | Measured from the images by this audit |

---

## 1. Composition distribution

**[DOC] Percentages:** the component percentages add up to exactly 100 for all 155 rows, and so for all 94 imaged swatches. For every swatch, the fibre code in the filename agrees with the CSV.

**Cotton percentage of each imaged swatch, by BWorth band:**

| BWorth band | Swatches | Sample IDs (cotton %) |
|---|---|---|
| Very high (98 to 100) | **40** | A01 to A05, A07 to A29, A42 to A44 (31 × 100%); B11, B12, B14, B15 (100%); C02, C03, C04, C08, C09 (100%) |
| High (80 to 95) | **3** | B01 (92), D05 (85), D09 (85) |
| Medium (50 to 66) | **8** | B04, B06, B08 (50); C01, C06 (60); C11 (52); D02, D04 (50) |
| Low (30 to 40) | **5** | B05, B07 (35); C05 (35); D03 (38); D06 (30) |
| Outside bands, below 30 | 1 | D01 (14) |
| Outside bands, gap 40 to 50 | 1 | D07 (44) |
| Outside bands, gap 66 to 80 | 1 | D08 (75) |
| Outside bands, gap 95 to 98 | 0 | none |
| No cotton | **35** | A30 to A33, A35 to A41, A45 to A53; B02, B03, B09, B10, B13, B16 to B20; C07, C10; D10, D11, D12 |
| **Total** | **94** | 59 contain cotton; 56 fall in a band |

**[DOC] Cotton mixed with other fibres:** 18 swatches. Every one is either undyed Source B or uncorroborated Source C/D.

| Sample | Cotton | Other fibres |
|---|---|---|
| B01 | 92 | spandex 8 |
| B04, B06, B08 | 50 | polyester 50 |
| B05, B07 | 35 | polyester 65 |
| C01 | 60 | recycled polyester 40 |
| C06 | 60 | polyester 40 |
| C11 | 52 | polyester 48 |
| C05 | 35 | polyester 53, regenerated cellulosic 12 |
| D01 | 14 | polyester 80, spandex 6 |
| D02 | 50 | modal rayon 50 |
| D03 | 38 | polyester 50, regenerated cellulosic 12 |
| D04 | 50 | polyester 50 |
| D05 | 85 | polyester 12, regenerated cellulosic 3 |
| D06 | 30 | recycled polyester 70 |
| D07 | 44 | polyester 50, regenerated cellulosic 6 |
| D08 | 75 | polyester 25 |
| D09 | 85 | polyester 15 |

**[DOC] The 35 swatches with no cotton:**
- 100% polyester, nylon 6 or 6,6, acrylic, viscose rayon or wool;
- D10 (regenerated cellulosic / nylon / spandex);
- D11 (polyester / modal / spandex);
- D12 (rayon / wool / nylon).

**[DOC] Construction:**
- Every Source A swatch is woven except A39, a knit.
- Sources C and D are all knit.
- Source B is mixed.

## 2. Label reliability: three evidence categories

| Category | Definition | Swatches |
|---|---|---|
| `supplier_documented` | Sources A and B, from Testfabrics, Inc. Composition is the supplier's specification **[DOC]**. It is **not** a laboratory result for these specimens. | 71 |
| `uncorroborated_care_label` | Sources C and D. Composition "taken from garment care label or similar and currently uncorroborated" **[DOC, DataDictionary]**. | 22 |
| `lab_confirmed` | C02 only. "Confirmed to be 100 % cotton by fully dissolving it in 70 % sulfuric acid" **[DOC, ChangeLog]**. | 1 |

C02 is counted only under `lab_confirmed` here. Its care label also said 100% cotton.

**Distribution by evidence category** (swatch counts):

| Band | Supplier-documented | Uncorroborated (care label) | Lab-confirmed |
|---|---|---|---|
| Very high | 35 | 4 | 1 (C02) |
| High | 1 | 2 | 0 |
| Medium | 3 | 5 | 0 |
| Low | 2 | 3 | 0 |
| Outside bands | 0 | 3 | 0 |
| No cotton | 30 | 5 | 0 |
| **Total** | 71 | 22 | 1 |

No swatch in this dataset has a lab-confirmed blend composition. The one lab confirmation, C02, is a 100%-cotton dissolution result.

## 3. Physical-sample independence

### 3.1 Relationship evidence

| Relationship | Members | Evidence | Status |
|---|---|---|---|
| Catalogue 400M, bleached, desized, mercerised cotton print cloth, 107 gsm | A42, A43, A44 (dyed navy, light blue, pink), B15 (undyed) | Identical Catalogue ID; same gsm and description | **[DOC]** shared catalogue. **[INF]** same base cloth (strong). |
| Catalogue 365, nylon 6,6 spun plain weave, 190 gsm | A45, A46, A47, B16 | Identical Catalogue ID, gsm and description | [DOC] / [INF] strong |
| Catalogue 266 / 266E, viscose challis, 138 gsm | A48, A49, A50, B17 | Same catalogue base, gsm and description | [DOC] / [INF] strong |
| Catalogue 527, wool flannel, 268 gsm | A51, A52, A53, B18 | Identical Catalogue ID, gsm and description | [DOC] / [INF] strong |
| 7409 polyester/cotton 65/35 broadcloth, 104 gsm | B05 (7409OPT, with optical brightener), B07 (7409-WOB, without) | Same catalogue base; differ only in finish | **[INF]** probable |
| 400-series cotton print cloth | B11 (400R, greige, 122 gsm), B12 (400, bleached and desized, 100 gsm), joining the 400M group | Same catalogue stem; consistent with greige → bleached → mercerised processing states | **[INF]** possible |
| Lot E-460-01 | A01, A04, A05, A12, A15, A24, A26, A28 (all 100% cotton, AISE) | Identical lot string | [DOC] shared lot. **[INF]** meaning unknown. |
| Lot 103 | A07, A09, A14 (cotton) **and A41 (nylon)** | Identical lot string | [DOC]. Lot shared across fibres. |
| Lot 104A | A32 (polyester) **and A36, A37 (nylon)** (A32 is written "104 A") | Identical lot string after removing spaces | [DOC]. Lot shared across fibres. |

**Lot numbers don't identify base cloth.** Lot strings are shared by swatches of **different fibres**: lot 103 by cotton and nylon, lot 104A by polyester and nylon. Those swatches can't share a base cloth. So lot numbers are **not** a reliable base-cloth identifier. They may be dye-batch or AISE-set codes. The strict scenario still merges shared lots, but only within the same fibre composition, as a precaution.

**Sources C and D:**
- **[DOC]** C comes from one secondhand store and D from one research partner. Manufacturers and lots are unknown.
- They are treated as independent because their compositions and constructions differ.
- That independence is **unverified**. D in particular may come from a single mill.

**[INF, not applied] AISE colour standards:** all the AISE 100% cotton colour standards (A01 to A29) may be dyed on one standard cotton cloth. Nothing in the metadata confirms or rules this out. If they are, the very-high supplier count shrinks much further (§3.3, note).

### 3.2 Grouping scenarios

| Scenario | Rule | Independent materials (of 94) |
|---|---|---|
| **Relaxed** | Merge only identical catalogue bases: 400M, 365, 266/266E, 527 | **82** |
| Intermediate | Relaxed, plus 7409 (B05 + B07) and the 400-series (B11, B12 join 400M) | **79** |
| **Strict** | Intermediate, plus same-fibre shared lots: E-460-01 (8 cotton), 103-cotton (A07, A09, A14), 104A-nylon (A36, A37) | **69** |

The earlier estimate of 72 to 82 merged only E-460-01. The strict figure here is lower, at 69, because it also merges the same-fibre parts of lots 103 and 104A.

### 3.3 Independent materials per BWorth band

**All evidence categories:**

| Band | Swatches | Relaxed | Intermediate | Strict |
|---|---|---|---|---|
| Very high | 40 | 37 | 35 | **26** |
| High | 3 | 3 | 3 | **3** |
| Medium | 8 | 8 | 8 | **8** |
| Low | 5 | 5 | 4 | **4** |
| Outside bands | 3 | 3 | 3 | 3 |
| No cotton | 35 | 26 | 26 | 25 |

**Supplier-documented only**, the best evidence available for training:

| Band | Swatches | Relaxed | Intermediate | Strict | Members (strict) |
|---|---|---|---|---|---|
| Very high | 35 | 32 | 30 | **21** | 31 dyed AISE/custom woven cotton + B11, B12, B14, B15 |
| High | 1 | 1 | 1 | **1** | B01 |
| Medium | 3 | 3 | 3 | **3** | B04, B06, B08 |
| Low | 2 | 2 | 1 | **1** | B05/B07 (7409) |
| No cotton | 30 | 21 | 21 | 20 | |

**Uncorroborated only:** very high 4, high 2, medium 5, low 3, outside bands 3, no cotton 5. These are the same under every scenario, since no relationships are documented. **Lab-confirmed:** 1 very high (C02).

> **Note:** if the AISE cotton standards share one base cloth (§3.1, not applied), the supplier very-high count could fall to single digits.

---

## 4. Image mapping

**[DOC] Naming:** `<SampleID>T_<rr>C.tiff`, where T is the stereo microscope, C is colour and rr is the replicate (01 to 06). All 564 files follow this pattern, and each of the 94 swatches has exactly six (`image_replicates` column in the CSV).

| Replicate | Side | Scale (px per 0.1 mm) [DOC] | Approximate field of view [INF from scale] |
|---|---|---|---|
| 01 | Front | 34.3 | 7.1 × 6.0 mm |
| 02 | Front | 47.4 | 5.2 × 4.3 mm |
| 03 | Front | 76.2 | 3.2 × 2.7 mm |
| 04 | Back | 34.3 | 7.1 × 6.0 mm |
| 05 | Back | 47.4 | 5.2 × 4.3 mm |
| 06 | Back | 76.2 | 3.2 × 2.7 mm |

**Redundancy:**
- **[DOC]** The six images show one physical swatch. The specimen's orientation was kept for each side, and it was centred under the lens.
- **[INF]** So the three magnifications of a side probably cover nested, overlapping areas. This was not verified pixel by pixel.
- For statistics, **each swatch is n = 1**, not n = 6. Front and back are different faces of the same fabric, not different fabrics.
- **[MEAS]** No two swatches have byte-identical images (SHA-256, earlier audit). No near-duplicate pairs appeared across swatches: the highest correlation between low-resolution thumbnails was 0.55, between A20 and A21.

**[MEAS] Exposure and suitability.** Small specular highlights are normal: the median image has 3% of pixels at ≥ 250. Eleven swatches have at least one image with more than 30% of pixels clipped:

| Problem | Swatches |
|---|---|
| Shadow clipping | A39 black nylon knit (up to 80%), A03 green cotton (66%), C07 (52%), C06 (42%) |
| Highlight clipping | A52 (63%), A53 (45%), A19 (44%), A32 (43%), A44 (42%), A18 (35%), A47 (34%) |

- These are flagged in the CSV `notes`.
- A39 and A03 are the most likely to be unusable for texture features.
- None is missing or corrupt.
- **[DOC]** Exposure and white balance were set per sample (`ColorParameters.csv`: 4 white-balance presets; gain NA, 0, 1, 2 or 22). Absolute colour is **not comparable across swatches**.

**[MEAS] Focus:** quarter-resolution sharpness is lower for replicates 03 and 06 (median 594 and 629, against about 1,050 to 1,250 for the others). That is expected at the highest magnification and does not indicate blur. No front/back pair differs sharply in focus.

---

## 5. Training feasibility

| Question | Assessment |
|---|---|
| **Independent samples per band** | Supplier-documented, strict: very high 21, high 1, medium 3, low 1. Relaxed: 32 / 1 / 3 / 2. |
| **Class imbalance** | About 21:1:3:1 strict, or 32:1:3:2 relaxed. Three of the four bands have three or fewer independent materials. |
| **Evidence quality** | Supplier specifications for A and B. Care labels for C and D. A single lab confirmation, C02 (100% cotton). No lab-confirmed blends. |
| **Confounding [DOC]** | All supplier-documented swatches outside very high (B01, B04 to B08) are **undyed white Source B**. 31 of the 35 supplier very-high swatches are **dyed Source A** woven fabrics. Band is therefore almost perfectly confounded with dye, source, colour and exposure settings. A classifier could reach high accuracy without seeing fibre. |
| **Four-class experiment** | **Not feasible.** With one independent material in each of high and low, grouped cross-validation can't hold out a band member and still train on one. Any per-class metric would rest on n = 1. |
| **Three-class** (for example very high / medium / high+low) | **Not defensible.** At most 4 non-very-high materials per merged class, all from one confounded subset. |
| **Binary: very high vs blended (30 to 95)** | **Not defensible for training.** Supplier evidence gives 21 to 32 against 5 to 6 materials. The only colour-matched comparison is inside undyed Source B: very high B14 plus the 400-series group, against 5 blend materials. That is **about 7 independent materials in total**. |
| **Binary: cotton-containing vs no cotton** | More balanced (strict 44 vs 25 overall, all evidence; supplier only 26 vs 20). But this is fibre identification, not BWorth banding, and it is still confounded by dye class and finish (for example, acid-dyed nylon and wool against reactive or vat-dyed cotton). Exploratory only. |
| **Supplier-documented alone** | **Insufficient coverage** of high, medium and low. |
| **Uncorroborated C/D** | They would add 10 in-band blend materials (2 high, 5 medium, 3 low), but with unverified labels. **They should not be training labels.** At most they could be a small, separately reported exploratory probe, and their labels must not be presented as ground truth. |

**Domain gap [DOC]:** these are stereo-microscope images at 3 to 7 mm fields of view, with per-sample exposure and white balance. BWorth's Phase 4 protocol uses phone or camera photos of whole fabrics. A model or result from NIST would not transfer to the BWorth image domain without separate validation.

## 6. Leakage-safe splitting, for any exploratory use

1. **Unit of splitting = strict provenance group** (`group_strict` in the CSV). All 6 images of a swatch, and every swatch in the same group, stay in one fold. Never split images randomly.
2. **Run a sensitivity check under `group_relaxed`.** If a result changes materially between strict and relaxed grouping, treat it as leakage-driven.
3. **Use leave-one-group-out**, or grouped k-fold stratified by band where a band has enough groups. With only 69 to 82 groups, report the result for each group, not just one pooled score.
4. **Aggregate predictions to swatch level**, by averaging the six images, before scoring. Keep magnification and side as recorded covariates.
5. **Keep the evidence categories apart.** Supplier-documented swatches go in training and validation. C and D go only into a separately reported exploratory probe. C02 serves as a single lab-confirmed check case.
6. **Add a confound control.** Repeat any analysis within undyed Source B only, and compare against a colour-only baseline (mean RGB). If the colour baseline matches the model, the model isn't using fibre information.
7. **Isolate from BWorth.** Never mix NIST swatches into the Phase 4 verified registry, Phase 5 snapshots, the lockbox, or BWorth training or test splits.

## 7. Recommendation: SUITABLE ONLY FOR EXPLORATORY ANALYSIS

**Why not four-class or reduced-class:**
- Under supplier evidence, the non-very-high bands hold 1 to 3 independent materials each (strict: 1 high, 3 medium, 1 low).
- No blend has lab confirmation.
- Band is confounded with dye and source.
- The extra C/D blends carry uncorroborated care-label compositions, which the project's data-integrity rules exclude as ground truth.

**Why not "insufficient data":**
- The dataset is well documented: 94 swatches, complete six-image sets, no duplicates, and checksums matching the official release.
- It can support work that makes **no claims about BWorth band accuracy**:
  - running the Phase 5 pipeline end to end on real images;
  - a study of microscopy-domain image embeddings, for example whether 100% cotton clusters apart from 100% synthetics within one dye class;
  - an exploratory cotton vs non-cotton study with the confound controls in §6.
- Any such work should be labelled exploratory and kept out of BWorth's model selection and reporting.

**For the BWorth band task itself, this dataset is insufficient**, and it does not reduce the 120-fabric collection target. That matches the earlier public-dataset research.

---

## Summary: swatches and independent materials per BWorth band

| Band | Swatches (all evidence) | Relaxed | Strict | Supplier-documented swatches | Supplier, relaxed | Supplier, strict |
|---|---|---|---|---|---|---|
| Very high (98 to 100) | 40 | 37 | 26 | 35 | 32 | 21 |
| High (80 to 95) | 3 | 3 | 3 | 1 | 1 | 1 |
| Medium (50 to 66) | 8 | 8 | 8 | 3 | 3 | 3 |
| Low (30 to 40) | 5 | 5 | 4 | 2 | 2 | 1 |
| **In-band total** | **56** | **53** | **41** | **41** | **38** | **26** |

**Other swatches:** 3 fall outside the bands (D01, D07, D08, all care-label). 35 have no cotton (26 relaxed, 25 strict).

**All 94 swatches:** 82 independent materials under relaxed grouping, 79 intermediate, 69 strict.

**Lab confirmation:** C02 is the only lab-confirmed sample, at 100% cotton (very high).
