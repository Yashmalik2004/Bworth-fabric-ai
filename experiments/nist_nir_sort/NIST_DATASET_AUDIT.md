# NIST NIR-SORT: read-only structural audit

- **Audit date:** 27 November 2026
- **Location audited:** `datasets/external/NIST_NIR_SORT/raw_download/NIST_SORT_DATA/`. This folder is git-ignored by the `datasets/*` rule.
- **Dataset:** "Near Infrared Spectra of Origin-defined and Real-world Textiles (NIR-SORT)", NIST, version 2.0.0, DOI [10.18434/mds2-3325](https://doi.org/10.18434/mds2-3325).
- **Method:**
  - Files were opened only for reading. Nothing in the dataset folder was modified, renamed, moved, deleted or converted.
  - None of the dataset's own scripts were executed.
  - The analysis script lived outside the dataset and repository, in the session scratchpad, and was run with `python -I`.
  - The script hashed every image (SHA-256), read each TIFF header and checked one image per source for pixel content.
- **Untouched:** BWorth datasets, training code, model checkpoints, Git branches and experiment results were not changed. This report is the only file written.

> **Scope note.** This extract holds only the **stereo colour microscopy** part of NIR-SORT, plus the top-level documentation. These parts of the full publication are **not present**:
> - the NIR spectra: `Spectroscopy/` (Benchtop_N, Handheld_F, Handheld_S, Standoff_Q), about 4,248 files;
> - the **Mono** and **Polarized** microscopy folders;
> - the uncertainty and test-condition files referenced in `DataDictionary.csv` (`EstimateUncertainties_*.xlsx`, `TestConditions_*.csv`).

---

## 1. Directory tree

The download arrived as five separate archive extracts, `NIST-Data-2026-11-27T05-50-1` to `-5`. Each extract recreates part of the `mds2-3325/` tree.

```
NIST_SORT_DATA/                                          5.5 GB total
├── NIST-Data-2026-11-27T05-50-1/mds2-3325/Microscopy/Optical/Stereo/Color/Source_A/Source_A/
│       306 × .tiff   (A01_a100x000x000_00_00_T_01C.tiff … A53_i100x000x000_00_00_T_06C.tiff)
├── NIST-Data-2026-11-27T05-50-2/mds2-3325/Microscopy/Optical/Stereo/Color/Source_B/Source_B/
│       120 × .tiff   (B01_a092e008x000_00_01_T_01C.tiff … B20_c100x000x000_00_00_T_06C.tiff)
├── NIST-Data-2026-11-27T05-50-3/mds2-3325/Microscopy/Optical/Stereo/Color/Source_C/Source_C/
│        66 × .tiff   (C01_… … C11_…)
├── NIST-Data-2026-11-27T05-50-4/mds2-3325/Microscopy/Optical/Stereo/Color/Source_D/Source_D/
│        72 × .tiff   (D01_… … D12_…)
└── NIST-Data-2026-11-27T05-50-5/mds2-3325/
    ├── ChangeLog.txt
    ├── DataDictionary.csv
    ├── DataFramework.txt
    ├── FileNamingConvention.pdf
    ├── license.txt
    ├── README_main.txt
    ├── RelationalDiagram.pdf
    ├── SampleInformation.csv
    ├── Microscopy/Optical/Stereo/
    │   ├── README_stereo.txt
    │   └── Color/ColorParameters.csv
    └── UtilityScripts/
        ├── 1_README_for_scripts.txt
        ├── 2_HowToRun.txt
        ├── NIR_SORT_py_functions.py
        ├── data_collection_from_folders.py
        ├── data_collection_from_folders.sh
        ├── example_basic_data_visualization.py
        ├── example_benchtop_to_handheld_comparison.py
        └── example_metadata_comparison.py
```

The image folders are nested twice (`Source_X/Source_X/`), probably because each source was zipped and unzipped separately. No `.zip` archives remain in the folder.

## 2. File inventory by type

| Type | Count | Files |
|---|---|---|
| CSV | 3 | `SampleInformation.csv` (155 rows), `DataDictionary.csv` (75 variables), `Microscopy/Optical/Stereo/Color/ColorParameters.csv` (94 rows) |
| TXT | 7 | `README_main.txt`, `README_stereo.txt`, `license.txt`, `ChangeLog.txt`, `DataFramework.txt`, `UtilityScripts/1_README_for_scripts.txt`, `UtilityScripts/2_HowToRun.txt` |
| PDF | 2 | `FileNamingConvention.pdf` (a one-page diagram of the filename fields), `RelationalDiagram.pdf` (a one-page diagram of the folder structure) |
| Python | 5 | All in `UtilityScripts/`. They are NIR-spectra helpers and were not run. |
| Shell | 1 | `UtilityScripts/data_collection_from_folders.sh` (not run) |
| Image | 564 | All `.tiff`. There are no `.bmp` files, which are the Mono-folder format. |

## 3. Image counts per source

| Source | Images | Physical samples (sample IDs) | Images per sample | Size on disk | `README_stereo.txt` states |
|---|---|---|---|---|---|
| Source_A | **306** | 51 | 6 | 3.05 GiB | 308 files, 3.06 GB |
| Source_B | **120** | 20 | 6 | 1.03 GiB | 120 files, 1.02 GB |
| Source_C | **66** | 11 | 6 | 0.65 GiB | 66 files, 666 MB |
| Source_D | **72** | 12 | 6 | 0.69 GiB | 72 files, 703 MB |
| **Total** | **564** | **94** | 6 | 5.41 GiB | 566 files |

**Discrepancy:** `README_stereo.txt` lists 308 Source A files, but 306 are present. 306 is exactly 51 samples × 6 replicates, and every one of the 51 samples has a complete set (01 to 06). 308 does not divide by 6.

Either the README count is off by two, or two extra files in the original Source A archive weren't extracted. The local files can't settle which. The fix is to compare against the official NIST file listing.

## 4. Image format and dimensions

All 564 images are identical in format:

| Property | Value |
|---|---|
| Container | TIFF, single frame |
| Pixel mode | RGBA, 8 bits per channel (32-bit RGB-alpha, matching the README) |
| Dimensions | 2448 × 2048 px (width × height) |
| File size | 8.1 to 12.0 MB each |
| Alpha channel | Constant 255 (fully opaque) in all four images spot-checked, one per source |

- **Camera:** FLIR Blackfly S colour, BFS-U3-50S4C-CS, with a Sony IMX547 sensor and a 670 nm IR-cut filter.
- **Microscope:** stereo inspection microscope (manufacturer unknown).
- **Capture software:** Spinnaker SDK / SpinView 4.0.0.116.
- The image scale depends on the replicate number (section 9).

## 5. `SampleInformation.csv` and `DataDictionary.csv`

### `SampleInformation.csv`

- 155 rows, 24 columns.
- Each Sample ID appears once.

The columns are:

`Sample ID, Source Class, Source, Source Number, Component 1, Component 1 Subclass, Component 1 %, Component 2, Component 2 Subclass, Component 2 %, Component 3, Component 3 Subclass, Component 3 %, Batch, Form Factor, Dye Type, Dye Concentration, Dye Color, Visual Inspected Color, Finish, Approx. Fabric Weight (gsm), Catalogue ID, Lot Number, Notes`

Rows per source: A 110, B 22, C 11, D 12.

### `DataDictionary.csv`

- 75 variables.
- Variables 1 to 21 describe `SampleInformation.csv`. The rest describe spectroscopy files that are **not in this extract**.
- Two discrepancies between the dictionary and the file:
  - The dictionary has no entries for the `Source`, `Dye Concentration` and `Notes` columns.
  - The dictionary gives `Sample ID` a length of 23 characters. That matches the IDs, which include the trailing underscore.

The key definitions, quoted from the dictionary:

- **Component N / Subclass / %:** "designated by source supplier or user". `%` is an integer from 0 to 100.
- **Note on every composition field:** *"For Source C and D, nominal fiber content is taken from garment care label or similar and currently uncorroborated."*
- **Form factor:** "designated by source supplier or user".
- **Catalogue ID / Lot Number:** "designated by source supplier".
- **Approx. Fabric Weight (gsm):** "if specified by manufacturer".

### `ColorParameters.csv`

- 94 rows, one per imaged sample.
- Columns: `Balance Ratio`, `Gain`, `Light Source`, `Notes`.
- Light-source preset: General 62, Shade 8000K 14, Sunny 5000K 10, Cloudy 6500K 8.
- Some rows record a change of white-balance ratio partway through capture ("2.19 / 2.05 post changes").
- `A21` is noted as "archive sample".
- So white balance and gain were **not constant** across samples.

## 6. Sample IDs, composition fields and provenance

### Sample ID structure

The ID structure comes from `DataFramework.txt` and `FileNamingConvention.pdf`:

```
A01_a100x000x000_00_00_
│   │            │  └─ form factor: 00 woven, 01 knit, 02 nonwoven, 03 yarn, 04 fibres (shredded), 05 fasteners
│   │            └──── batch/lot: parent material the specimen was cut from
│   └───────────────── fibre content: 3 × (class letter + 3-digit nominal %)
│                       a cotton, b polyester, c synthetic polyamide, d acrylic, e spandex,
│                       f other PU, g regenerated cellulosic, h other natural cellulosic, i natural polyamide, x empty
└───────────────────── source letter + number
```

For all 94 imaged samples, the fibre percentages in the ID match `Component 1/2/3 %` in the CSV exactly (0 mismatches).

### Provenance by source

These are the dataset's own statements:

| Source | Class | Origin (README_main) | Composition authority | Lot / catalogue IDs |
|---|---|---|---|---|
| A (51 imaged) | AISE colour-maintenance standards (A01 to A41) and custom-dyed fabrics (A42 to A53) | Bought from Testfabrics, Inc. (West Pittston, PA) | Supplier specification | Present: `CFT AISE n`, or catalogue numbers 400M, 365, 266, 527 |
| B (20 imaged) | Undyed blends and undyed fabrics | Testfabrics, Inc. | Supplier specification | Present for nearly all (B13 has no lot; B15 to B18 have lot "unknown") |
| C (11 imaged) | Post-consumer garments | "local secondhand store"; manufacturer unknown | **Garment care label, uncorroborated.** Exception: **C02** was confirmed as 100% cotton by full dissolution in 70% sulphuric acid (ChangeLog). | "unknown" |
| D (12 imaged) | Pre-consumer production samples | "production samples from a research partner"; manufacturer unknown | **Care label "or similar", uncorroborated** | "unknown" |

None of the composition values is described as a laboratory test of that specimen, except C02. The Source A and B values are **supplier specifications**, which is not the same as lab verification.

### Composition of the 94 imaged samples

Cotton percentages are mapped below onto the BWorth Phase 5 bands from `experiments/phase5/protocol.json`:

- Low: 30 to 40
- Medium: 50 to 66
- High: 80 to 95
- Very high: 98 to 100

The mapping is for orientation only.

| Cotton content | A (supplier) | B (supplier) | C (care label) | D (care label) | Total |
|---|---|---|---|---|---|
| Very high band (100%) | 31 | 4 | 5 (incl. C02, acid-confirmed) | 0 | 40 |
| High band (80 to 95%) | 0 | 1 (B01, 92%) | 0 | 2 (D05, D09, 85%) | 3 |
| Medium band (50 to 66%) | 0 | 3 (B04, B06, B08, 50%) | 3 (C01, C06 60%; C11 52%) | 2 (D02, D04, 50%) | 8 |
| Low band (30 to 40%) | 0 | 2 (B05, B07, 35%) | 1 (C05, 35%) | 2 (D03 38%; D06 30%) | 5 |
| Cotton between bands | 0 | 0 | 0 | 3 (D01 14%; D07 44%; D08 75%) | 3 |
| No cotton | 20 | 10 | 2 | 3 | 35 |
| **Total** | 51 | 20 | 11 | 12 | **94** |

The 35 samples with no cotton are:

- polyester;
- nylon 6 and 6,6;
- acrylic;
- viscose rayon;
- wool;
- regenerated-cellulosic or nylon blends with spandex.

**Shared base fabrics.** The physical samples are distinct, but some were not made from independent materials.

These groups share a catalogue number. They are the same base cloth, dyed in different colours or left undyed:

| Catalogue | Base cloth | Samples |
|---|---|---|
| 400M | cotton print cloth | A42, A43, A44, B15 |
| 365 | nylon 6,6 | A45, A46, A47, B16 |
| 266 / 266E | viscose rayon | A48, A49, A50, B17 |
| 527 | wool flannel | A51, A52, A53, B18 |

These groups are probably related:

- **7409 broadcloth:** B05 is "7409OPT" and B07 is "7409-WOB", the same polyester/cotton fabric with and without optical brightener.
- **400-series cotton print cloth:** B11 is "400R" (greige), B12 is "400" (bleached) and B15 is "400M" (mercerised).

This group may be related:

- **Lot E-460-01:** eight 100% cotton AISE fabrics share this lot: A01, A04, A05, A12, A15, A24, A26 and A28. Whether that means the same greige base cloth can't be determined from the files.

## 7. Mapping from image filenames to physical sample IDs

Each image filename is the sample ID followed by `T_<replicate><format>.tiff`:

```
A01_a100x000x000_00_00_  T  _01C .tiff
└──── Sample ID ───────┘ │   │└─ C = colour
                         │   └── replicate 01–06 (see section 9)
                         └────── instrument T = stereo optical microscope
```

- All 564 filenames match this pattern.
- Every filename uses instrument `T` and format `C`.
- Each image's source letter matches its `Source_X` folder.
- Each of the 94 sample IDs has exactly six images, replicates 01 to 06.
- **All six images of a sample show the same physical swatch:** three magnifications, front and back. Each image is not a separate fabric. For machine learning, the unit of grouping must be the sample ID, or the base-cloth group from section 6.

## 8. Missing files, duplicate filenames and duplicate hashes

| Check | Result |
|---|---|
| Images whose sample ID is not in `SampleInformation.csv` | **0** |
| Imaged samples with fewer than 6 replicates | **0** |
| Duplicate filenames (across all folders) | **0** |
| Duplicate SHA-256 image hashes | **0**: all 564 images are byte-distinct |
| Duplicate sample IDs in `SampleInformation.csv` | **0** |
| `SampleInformation.csv` IDs with **no** colour images | **61** (see below) |
| `ColorParameters.csv` ID mismatches | 2 cosmetic: `B19_c100x000x000_00_01` and `B20_c100x000x000_00_00` lack the trailing `_`. Their images are present. |
| Gaps in the source numbering | A06 and A34 appear nowhere, in the CSV or the images. This is a gap in the numbering, not a missing file. |
| Source A count against the README | 306 present, 308 stated (section 3) |

**The 61 sample IDs without colour images are all form factor `04` (shredded fibres), not fabrics:**

- A55 to A73, batches 01 to 03 (57 IDs): custom cotton/polyester fibre blends from 95/5 to 5/95 in steps of 5%.
- A07, batches 01 and 02 (2 IDs): the 100% cotton fibre reference for those blends.
- B09, batches 01 and 02 (2 IDs): the 100% polyester fibre reference.

`README_stereo.txt` says these fibre blends were imaged with the **polarised** camera on microscope slides. That folder isn't in this extract. Their absence from the colour folder is expected and doesn't count as missing data.

The README's "113 fabrics" is 94 fabric-form samples plus 19 fibre-blend compositions (A55 to A73).

No pixel-level near-duplicate analysis was done. The check above covers exact byte duplicates only.

## 9. Image acquisition (from `README_stereo.txt` and `README_main.txt`)

- **Preparation:** "Each fabric specimen was laid flat, centered under the objective lens, and held in place by stage clips." The specimen's orientation was kept the same for all front images and for all back images.
- **Front and back:** replicates **01, 02 and 03 show the front** and **04, 05 and 06 show the back**. Each swatch was physically marked "F" on its front.
- **Magnification (image scale):**

  | Replicates | Pixels per 0.1 mm | Approximate field of view |
  |---|---|---|
  | 01 and 04 | 34.3 ± 0.17 | about 7.1 × 6.0 mm |
  | 02 and 05 | 47.4 ± 0.30 | about 5.2 × 4.3 mm |
  | 03 and 06 | 76.2 ± 0.29 | about 3.2 × 2.7 mm |

  The field-of-view column is derived from 2448 × 2048 px and the stated scale.

- **Exposure and colour:**
  - "Brightness was adjusted between samples to obtain the most accurate image."
  - The white-balance preset, balance ratio and gain varied by sample (`ColorParameters.csv`).
  - "No Spinview adjustments were made to resolve glare on Source B undyed fabrics."
- **Swatch size:** about 10 × 10 cm, the same specimens used for NIR.
- **Lab conditions:** ambient lab conditions at NIST Gaithersburg. Version 2.0 data was collected from August 2024 to December 2025.
- **Consequence:** these are fixed-geometry, close-up stereo-microscope images with operator-adjusted exposure and per-sample white balance. They aren't the phone or camera photos the BWorth collection protocol uses. The colour isn't calibrated across samples.

## 10. Licence

The text below is quoted from `license.txt`, with no interpretation added:

- **Data:** "This data/work was created by employees of the National Institute of Standards and Technology (NIST), an agency of the Federal Government. Pursuant to title 17 United States Code Section 105, works of NIST employees are not subject to copyright protection in the United States. This data/work may be subject to foreign copyright."
- "To the extent that NIST may hold copyright in countries other than the United States, you are hereby granted the non-exclusive irrevocable and unconditional right to print, publish, prepare derivative works and distribute the NIST data, in any medium, or authorize others to do so on your behalf, on a royalty-free basis throughout the world."
- **Conditions stated:**
  - "Modified works should carry a notice stating that you changed the data and should note the date and nature of any such change."
  - "Please explicitly acknowledge the National Institute of Standards and Technology as the source of the data."
  - "Permission to use this data is contingent upon your acceptance of the terms of this agreement and upon your providing appropriate acknowledgments of NIST's creation of the data/work."
- **Warranty:** provided "AS IS", with no warranty, including of "DATA ACCURACY".
- **Code** (the `UtilityScripts/` folder) has a separate NIST software notice. It requires keeping the notice intact and marking any modifications.
- **Citation requested** (`README_main.txt`): K.E. Goodge, C.J. Vederman, R.V. Patel, C.M. Wentz, Z. Tsinas, A.K. Landauer, A.L. Forster, "Near Infrared Spectra of Origin-defined and Real-world Textiles (NIR-SORT)…", NIST, Version 2.0.0, https://doi.org/10.18434/mds2-3325.
- `README_main.txt` also points to NIST's copyright, fair-use and licensing statements (nist.gov/director/copyright-fair-use-and-licensing-statements-srd-data-and-software).

The licence text doesn't name commercial use one way or the other. This audit draws no conclusion on commercial use. That question, and the note on foreign copyright, should be reviewed by whoever approves external data use for BWorth.

---

## Conclusion: usable samples

**94 distinct physical fabric samples (swatches) have colour microscopy images together with composition metadata.** Each has 6 images (564 in total), with no missing replicates and no duplicates. These are 94 swatches, not 564 independent fabrics.

These 94 differ in how reliable their composition is, and how independent they are:

| | Samples |
|---|---|
| Supplier-specified composition (Testfabrics, sources A and B) | **71** |
| Care-label or "similar" composition, uncorroborated (sources C and D) | **23**. One of these, C02, was acid-dissolution confirmed as 100% cotton. |
| **Containing any cotton** | **59**: 41 supplier-specified, 18 care-label |
| Cotton within a BWorth Phase 5 band | 56: 41 supplier-specified (35 very high, 1 high, 3 medium, 2 low) and 15 care-label |

On independence:

- Merging the samples cut from a shared catalogue base cloth (400M, 365, 266, 527) leaves at most **82** independent base materials among the 94.
- Also merging the probably related 7409 pair and the 400-series print cloths leaves **79**.
- If lot E-460-01 turns out to mean one shared base cloth, the count falls further, to **72**.

No sample in this extract carries laboratory-verified composition for that specimen, except C02. None should be entered into the Phase 4 verified registry as lab-verified on the basis of this dataset.
