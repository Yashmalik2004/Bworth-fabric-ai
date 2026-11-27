# Beglarian Fabrics evaluation sample: read-only dataset audit

- **Date:** 27 November 2026
- **Dataset audited:** `datasets/external/beglarian_fabrics/`. Git-ignored by the existing `datasets/*` rule.
- **What it is:** "Beglarian Fabrics Textile Dataset (evaluation sample)". This is a free sample of a larger commercially licensed dataset, published on Hugging Face as `BeglarianFabrics/textile-dataset-sample`.
- **Method:**
  - Every file was opened read-only. Nothing in the dataset folder was modified, renamed or moved.
  - No dataset code was executed.
  - The audit script is `audit_beglarian.py` (this folder), run with `python -I` (see "Reproducing the audit").
  - No model was trained, no package was installed and nothing was downloaded.
- **Data source and attribution:** *Beglarian Fabrics Textile Dataset (evaluation sample)*, © Beglarian Fabrics, licensed CC BY-NC 4.0 (https://creativecommons.org/licenses/by-nc/4.0/). This report and `composition_distribution.csv` are derived summaries (counts, band assignments and anonymous sample IDs). No images or catalogue descriptions are redistributed here.
- **Outputs:**
  - In the repository:
    - `BEGLARIAN_AUDIT_REPORT.md`;
    - `composition_distribution.csv`: band, category, cotton-percentage and fibre tables;
    - `audit_beglarian.py`.
  - **Local only, not committed.** Each reproduces per-fabric catalogue labels or per-file hashes of the gated sample, so they are kept out of the public repository and regenerated with the script:
    - `fabric_inventory.csv`: 100 rows, one per fabric;
    - `image_inventory.csv`: 486 rows, one per image.

## Verdict: EXPLORATORY ONLY

**What's good:**
- The sample is clean and well structured: 100 fabrics, 486 readable images, complete metadata, consistent labels and a fixed capture protocol.
- Composition is given with exact percentages.

**Why it can't support BWorth's composition system:**
1. **Licence.** The sample is CC BY-NC 4.0 and was obtained through a gate that asks users to confirm evaluation and non-commercial use. The README states that "training or fine-tuning commercial models requires a commercial licence for the full dataset."
2. **Too few cotton fabrics.** Only 28 of 100 fabrics contain cotton, and BWorth's bands hold 11 / 5 / 1 / 2 fabrics (very high / high / medium / low). Only one fabric is a cotton–polyester blend.
3. **Seller labels, not tests.** Labels come from the seller's product catalogue, not independent laboratory tests.
4. **Composition is confounded with fabric type.** All 6 poplins and all 4 corduroys contain cotton. Every satin, sateen, crepe, chiffon and jacquard contains none.

The sample is useful for **evaluating whether to license the full dataset** and for non-commercial exploratory analysis. It is not useful as BWorth training data.

---

## 1. Documentation and metadata

### Layout (as found)

The layout differs from the structure described in the request:
- Images sit **directly in `data/train/`** next to `metadata.jsonl`. There is no `images/` subfolder.
- The folder also contains Hugging Face download leftovers: `.cache/huggingface/` (492 small cache metadata files) and `.gitattributes`.

```
datasets/external/beglarian_fabrics/            127 MB
├── README.md                                    dataset card (licence, fields, gating)
├── .gitattributes                               Hugging Face LFS rules
├── .cache/huggingface/…                         download-tool cache (metadata only)
└── data/train/
    ├── metadata.jsonl                           486 lines, one JSON object per image
    └── bf-<10 hex>_<view>.jpg                   486 JPEG images
```

### README

The dataset card claims:
- "100 real fabrics · 486 studio photos · 33 fabric types · expert labels". The metadata actually has 35 distinct `fabric_type` values. The difference is probably the generic "Woven" and "Knitted" entries; that's an inference.
- Fabrics photographed in-house by Beglarian Fabrics, "a European deadstock fabric house".
- Capture protocol since 2023: "a swirl fold, a flat view with a ruler for scale, close-ups and the reverse side", on a light background.
- Images resized to 1024 px on the long side; 3 to 7 photos per fabric.
- Rows of the same fabric share a `sample_id`.

### `metadata.jsonl` structure

The structure was verified against the actual records:
- 486 records, all with the **same 18 keys** and consistent types.
- Every record's `file_name` is unique.

| Field | Type | Meaning (README, confirmed in the data) |
|---|---|---|
| `file_name` | str | Image filename in `data/train/` (**the image path field**) |
| `sample_id` | str | Anonymous fabric ID, `bf-<10 hex>` (**the fabric ID**; all views of one fabric share it) |
| `view` | int | Photo index, 0 = main view; contiguous 0..n−1 for every fabric |
| `captured` | str | Capture month, YYYY-MM; range 2023-10 to 2026-09 |
| `fabric_type` | list[str] | Fabric type(s), e.g. `["Poplin","Woven"]`; 35 distinct values |
| `color` | list[str] | Colour family and pattern (multi-label, e.g. `Shades of blue`, `Printed`, `Plain`); 17 values |
| `composition` | list[{fiber, pct}] | **Fibre composition** with exact percentages |
| `weave` | list[str] | Weave or construction; 12 values |
| `stretch` | list[str] | Stretch class; 4 values |
| `features` | list[str] | Finish and pattern features |
| `intended_use` | list[str] | End uses |
| `country_of_origin` | list[str] | Italy, France or Japan (**the only provenance field**) |
| `width_cm`, `weight_gsm` | float | Roll width and **fabric weight (GSM)**. **No thickness field exists.** |
| `caption` | str | "Short caption generated from the labels" |
| `title_en`, `description_en`, `description_fr` | str | Catalogue title and descriptions |

**Fields that don't exist:**
- No supplier, mill, lot, product or SKU field: only the anonymous `sample_id`.
- No thickness field.
- No field saying how composition was determined.

### Representative record

This uses the example already published in the dataset card, which was checked against `metadata.jsonl` and matches it exactly. Text fields (`caption`, `title_en`, descriptions) are omitted here; see the card.

```json
{"file_name": "bf-f95ec0f58b_0.jpg", "sample_id": "bf-f95ec0f58b", "view": 0, "captured": "2024-11",
 "fabric_type": ["Tweed", "Woven"], "color": ["Shades of gray"],
 "composition": [{"fiber": "Polyamid", "pct": 85.0}, {"fiber": "Cotton", "pct": 15.0}],
 "weave": ["Twill Weave"], "stretch": ["Non-stretch"], "features": ["Panel Print", "Checked", "Houndstooth"],
 "intended_use": ["Dress/Blouse", "Suit"], "country_of_origin": ["Italy"], "width_cm": 142.0, "weight_gsm": 270.0}
```

All 486 records share these 18 keys. The fields shown above, plus `caption`, `title_en`, `description_en` and `description_fr`, make up the full set.

### Label provenance

The README states that labels "come from the product catalogue of the team that buys, measures and sells each fabric". They are **seller-catalogue labels**, called "expert labels" in the card.

Nothing documents how fibre composition was determined: lab test, mill specification or label. For deadstock fabric, composition usually comes from the original mill's documentation, but **that is an inference, not stated**. These labels must be treated as **supplier/seller-reported, not independently verified.**

## 2. Image inventory (`image_inventory.csv`)

| Check | Result |
|---|---|
| Image files | **486**, all `.jpg`, all decode as JPEG RGB (`PIL verify` + full load); **0 unreadable** |
| Resolution | All portrait, long side 1024 px. 479 are 768 × 1024; 7 are 668 to 788 × 1024. |
| Total size | 129.8 MB |
| Byte-identical duplicates (SHA-256) | **0** |
| Images without metadata / metadata without image | **0 / 0** |
| Near-duplicate check across fabrics (8 × 8 difference hash, Hamming ≤ 6) | 19 pairs (12 distinct images) flagged, almost all view 3 or 4. **All 12 images were inspected visually.** Every pair is two different fabrics sharing the **photo layout** (a flat shot on the same table). No cross-fabric duplicate images were found. |

The `view` index is not a fixed view type. This comes from the full view sets of 4 fabrics inspected, plus the 12 flagged view-3/4 images.
- The ruler shot is view 3 for some fabrics and view 4 for others.
- One inspected fabric, `bf-f95ec0f58b` (views 0 to 5), has no ruler shot at all, despite the README's protocol description.
- The views seen include swirl fold, edge or selvedge, close-up drape, and flat shots with or without the ruler.

Any per-view analysis would need the view type labelled by hand.

## 3. Independent fabrics (`fabric_inventory.csv`)

| | Count |
|---|---|
| Unique `sample_id` (fabrics) | **100** |
| Images per fabric | 3 (9 fabrics), 4 (11), 5 (66), 6 (13), 7 (1) |
| Fabrics whose labels differ between their images | **0**: all 14 label fields are identical across every image of every fabric |

**Do the images represent the same fabric?** Yes: per the README, the filenames and the visual check, every image sharing a `sample_id` is a different view of one physical fabric. **The unit of analysis is the fabric (n = 100), never the image (n = 486).**

**Could different `sample_id`s be the same underlying fabric?**
- **REL-01** (`bf-56af579b68`, `bf-8c99c01c1a`): identical composition (98% viscose, 2% lurex), 90 g/m², 145 cm width and origin (France). The prints differ. This is **probably the same base cloth printed with different designs**, so treat it as related.
- **REL-02** (`bf-27a4e354bb`, `bf-7647311790`): identical composition (100% viscose) and weight (100 g/m²). Width (137 vs 138 cm) and type (sateen vs crepe) differ, so they are **probably different cloths**. It is flagged conservatively.
- The two fabrics titled "Multicolour Cotton Poplin" are visibly different fabrics.

**Result:** 100 fabrics, **98 independent groups under strict grouping** (99 if only REL-01 is grouped). All 28 cotton-containing fabrics are in separate groups.

**Is `sample_id` a reliable identifier?** Within this sample, yes: it's consistent, there are no label conflicts, and image mapping is complete. The README says IDs are anonymised, and there is no supplier, lot or product field. So same-mill or same-collection relationships **can't be checked**. Only relationships visible in the specifications (above) can be detected.

## 4. Composition (`composition_distribution.csv`)

**Data quality:**
- Every fabric has a composition.
- All percentages **sum to exactly 100**.
- Labels are identical across each fabric's images.
- One label is ambiguous: `bf-1c43f51ccc` includes "Other Fibres 2%".
- The fibre vocabulary mixes names for the same fibre: "Polyamid" (147 image records) and "Nylon" (5), "Wool" and "Virgin Wool".

**Composition categories** (fabrics, with images in brackets):

| Category | Fabrics (images) |
|---|---|
| 100% cotton | **7** (34) |
| Cotton + elastane / polyurethane only (CSV label `cotton + elastane/polyurethane only`; includes the two 80% cotton / 20% polyurethane fabrics) | 8 (35) |
| Cotton–polyester blend (cotton and polyester only) | **1** (5): `bf-6f503dae35`, 70/30 broderie anglaise |
| Cotton + other fibres (linen, viscose, polyamide, wool, lurex, …) | 12 (60) |
| 100% polyester | **5** (25) |
| No cotton, other fibres | 67 (327) |
| **Total** | **100** (486) |

- 28 fabrics contain cotton and 72 contain none.
- Cotton percentages present: 10, 15, 22, 25, 30, 35, 43, 50, 70, 75, 80 (×3), 82, 93, 96, 97, 98 (×2), 99 (×2), 100 (×7). The median among cotton fabrics is 87.5%.
- The most common fibres, by number of fabrics containing each (with "Polyamid" and "Nylon" merged, and "Virgin Wool" counted as wool):

  | Fibre | Fabrics |
  |---|---|
  | Polyamide/nylon | 32 |
  | Viscose | 28 |
  | Cotton | 28 |
  | Elastane | 21 |
  | Polyester | 21 |
  | Silk | 16 |
  | Wool | 16 |
  | Lurex | 12 |

  The full table is in the CSV.

**BWorth bands.** The band definitions are unchanged; out-of-band values are counted separately.

| Band | Fabrics | Images | Sample IDs (cotton %) |
|---|---|---|---|
| Very high (98 to 100) | **11** | 53 | 7 × 100%, `bf-713435bf7b` 98, `bf-4aa7d7d3c7` 98, `bf-717157c724` 99, `bf-38fa8cd269` 99 |
| High (80 to 95) | **5** | 22 | `bf-fedc8142ed` 80 (with PU), `bf-23b41adefa` 80 (with linen), `bf-3a0256a69f` 80 (with PU; eco leather), `bf-3661fd4650` 82, `bf-1ec8215afd` 93 |
| Medium (50 to 66) | **1** | 5 | `bf-c787992ecd` 50 (cotton/linen) |
| Low (30 to 40) | **2** | 10 | `bf-a44c991e89` 30 (cotton/viscose corduroy), `bf-d2d0cf44fe` 35 (cotton/linen) |
| Outside, below 30 | 4 | 21 | 10, 15, 22, 25 |
| Outside, gap 40 to 50 | 1 | 4 | 43 |
| Outside, gap 66 to 80 | 2 | 10 | 70, 75 |
| Outside, gap 95 to 98 | 2 | 9 | 96, 97 |
| No cotton | 72 | 352 | |

**Enough samples per band?** No. Medium has **1** fabric and low has **2**. Even very high has only 11, and they span poplin, corduroy, twill, flannel and bouclé.

Several in-band blends aren't cotton–polyester at all: cotton–linen, cotton–viscose, cotton–polyurethane. Whether such blends matter depends on the fabrics BWorth is meant to assess; that's a product decision this audit doesn't settle. Cotton–polyester, the most common commercial blend, is barely covered (1 fabric).

`bf-3a0256a69f` ("Eco leather", 80% cotton, 20% polyurethane) is probably a polyurethane-coated cotton substrate, inferred from the type name and not checked in the images. If so, its visible surface is the coating, not the cotton.

## 5. Quality and potential confounding

Here "cotton group" means: no cotton (72 fabrics), cotton 1–49% (7), 50–97% (10), ≥ 98% (11).

**Fabric type and weave (strong shortcut):**
- **All 6 poplins and all 4 corduroys contain cotton.** Five of the 11 fabrics with ≥ 98% cotton are poplins.
- No fabric of these types contains cotton:

  | Type | Fabrics |
  |---|---|
  | Satin / sateen | 10 |
  | Crepe | 7 |
  | Chiffon | 7 |
  | Jacquard | 6 |
  | Organza / velour | 1 each |

  Mesh is almost entirely non-cotton: 6 of 7 mesh fabrics have none, and the seventh is 10% cotton.

- The ≥ 98% cotton group is 7 of 11 plain weave and has no satin, crepe or mesh. A model could learn "poplin or corduroy appearance → cotton".

**Pattern and colour:**
- 43 fabrics are **printed**; in many photos the print dominates the image.
- Colour labels are spread across groups. "Shades of white" leans cotton (5 of 7 white fabrics contain cotton). The other colours are mixed.
- Mean brightness of view 0 is similar across groups (medians 89 to 110 on a 0–255 scale). The Spearman correlation between cotton % and brightness is −0.18.

**Weight (GSM):**
- Range 35 to 800 g/m² (median 140).
- Medians by group: no cotton 135, cotton ≥ 98% 140, 50–97% 185, 1–49% 270.
- Spearman correlation with cotton % is 0.11: weak.

**Other attributes:**
- Origin is 70 Italy, 29 France and 1 Japan; Italy is the majority in every cotton group.
- Stretch is mostly non-stretch.
- Capture dates run 2023 to 2026 (53 fabrics in 2025, 38 in 2026).

**Other dominance:** the "no cotton" class is dominated by viscose, polyamide and silk fashion fabrics (crepe, chiffon, sateen, jacquard). These look nothing like the 100% polyester and cotton–polyester basics BWorth targets.

**Shortcuts a model could learn:**
1. Fabric type and structure (poplin, corduroy, terry → cotton; satin, crepe, chiffon, mesh → no cotton).
2. Print design.
3. Drape and sheen visible in the swirl-fold view (viscose and silk drape differently).
4. Studio-protocol cues common to all images: the ruler, the table and the background. These don't vary with label, but they make the images look unlike field photos.

## 6. Licence and commercial suitability

| Question | Answer from the documentation |
|---|---|
| Licence of the downloaded sample | **CC BY-NC 4.0** (README YAML `license: cc-by-nc-4.0` and the "Licence" section). There is no separate LICENSE file. |
| Commercial use | **Not allowed** under CC BY-NC (NonCommercial). |
| Access terms | The gated-access form includes the checkbox "I will use this sample for evaluation and non-commercial purposes only". Gated Hugging Face datasets can only be downloaded after the form is submitted, so whoever downloaded this sample presumably accepted it. That is an inference from how gating works. Its "Intended use" options include "Evaluating a commercial licence". |
| Model training | The README: "evaluation, research and other non-commercial use … **Training or fine-tuning commercial models requires a commercial licence for the full dataset.**" Training a model for BWorth's commercial product on this sample is therefore not permitted without a commercial licence. |
| Full dataset | Separate commercial, non-exclusive licence ("50,000+ fabrics, 170,427 HD photos and 46,758 drape videos"), via data.beglarianfabrics.com / kb@beglarianfabrics.com. Downloading the sample grants nothing for the full dataset. |
| Attribution | Required: "with attribution to Beglarian Fabrics" (the BY term). |

Permission to download is not permission for commercial AI training. Whether any internal research use counts as "non-commercial" for BWorth is a legal question. It should be confirmed with the licensor before the sample is used beyond evaluation.

## 7. Compatibility with BWorth tasks

| Task | Available | Label quality | Assessment |
|---|---|---|---|
| **A. Exact cotton-% regression** | 28 cotton fabrics (10 to 100%) + 72 at 0% | Seller catalogue, exact percentages, unverified | **Not feasible.** Values are zero-inflated, the middle range is sparse (1 fabric at 50%, none from 51 to 69%), and most blends aren't cotton–polyester. |
| **B. Four-band classification** | 11 / 5 / 1 / 2 fabrics; 9 outside the bands | Same | **Not feasible.** No grouped split can hold out a medium or low fabric and still train on one. |
| **C. Cotton vs no cotton** | 28 vs 72 fabrics (98 independent groups) | Same | **Exploratory only.** It is technically possible as a small pilot, but heavily confounded with fabric type (§5). It would also be a fibre-presence task over very different fabric families, not BWorth's band task. Commercial training is barred by the licence. |
| **D. Fabric-type classification** | 35 type labels over 100 fabrics; most types have 1 to 8 fabrics. The most frequent specific types: sateen 8, tweed 7, crepe 7, chiffon 7. | Catalogue labels, multi-label | **Exploratory only.** Too few per type, apart from coarse woven (84) vs knitted or terry (8). |
| **E. GSM prediction** | 100 fabrics, 35 to 800 g/m², all with a value | "Measured" by the seller per the README; method undocumented | **Exploratory only.** With 100 fabrics and wide variety, a ruler view gives scale but not thickness. Useful at most as a feasibility probe for the full dataset. |

## 8. Recommendations

1. **Don't use the sample to train or tune BWorth models.** Use it only for evaluation and non-commercial exploratory analysis, with attribution, as its licence and access terms require.
2. **Don't add any Beglarian fabric to the Phase 4 verified registry.** The labels are seller-reported, with no lab evidence.
3. **If the full dataset is considered:**
   - Ask the licensor for counts of cotton/polyester fabrics per BWorth band. The sample's 28% cotton share and single cotton–polyester blend suggest the full set may also be fashion-fabric heavy.
   - Ask how composition labels are sourced (mill specification, label or lab).
   - Get the commercial licence terms in writing.
   - Plan independent lab spot-checks, consistent with the approved verification strategy.
4. **If exploratory work goes ahead:**
   - Group by `sample_id`, and treat REL-01 as related.
   - Normalise the fibre names ("Polyamid" = "Nylon"; "Virgin Wool" = "Wool").
   - Control for fabric type, for example by comparing within a type.

---

## Reproducing the audit

**Requirements:**
- The gated sample, downloaded to `datasets/external/beglarian_fabrics/` (git-ignored; access requires accepting Beglarian's terms on Hugging Face).
- Python with numpy and Pillow.

**Command,** from the repository root:

```
python -I experiments/beglarian_dataset_audit/audit_beglarian.py \
    datasets/external/beglarian_fabrics experiments/beglarian_dataset_audit [summary.json]
```

**What the script does:**
- Reads the dataset read-only and executes nothing from it.
- Writes the three CSVs below. The optional `summary.json` holds near-duplicate pairs, resolutions and relationship candidates.
- Near-duplicates use an 8 × 8 difference hash with cross-fabric Hamming distance ≤ 6.
- Composition categories: "Polyamid"/"Nylon" and "Wool"/"Virgin Wool" are merged for fibre counting. Cotton is detected by the exact fibre name "Cotton".

The two local-only inventories are written to the same folder. **Keep them out of commits.**

### Output files

| File | In Git | Contents |
|---|---|---|
| `composition_distribution.csv` | yes | Long table: BWorth band, composition category, each cotton-% value and fibre presence, with fabric and image counts and anonymous sample IDs |
| `audit_beglarian.py` | yes | Audit script |
| `fabric_inventory.csv` | **no (local only)** | 100 rows: sample_id, image count and views, capture month, composition (raw), sum, cotton %, BWorth band, category, normalised fibres, ambiguous labels, fabric type, weave, stretch, colour, features, use, origin, width, GSM, title, label-consistency flag, possible related group and basis, strict independent group, view-0 brightness and RGB |
| `image_inventory.csv` | **no (local only)** | 486 rows: file_name, sample_id, view, metadata match, readable, format, mode, width, height, bytes, SHA-256, mean RGB, mean brightness, the fabric's cotton % and band |
