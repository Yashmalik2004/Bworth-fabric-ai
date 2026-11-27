# NIST NIR-SORT: Source A colour-image count (306 present, 308 stated)

- **Date:** 27 November 2026
- **Follows on from:** `NIST_DATASET_AUDIT.md` §3
- **Data source:** NIST NIR-SORT, Version 2.0.0 (metadata record 2.0.1), https://doi.org/10.18434/mds2-3325
- **What was not changed:** no dataset file was modified, and no project file was changed except this report. No duplicate archive was downloaded.
- **What was downloaded from NIST, all to the session scratchpad, outside the repository:**
  - the public metadata record (JSON, 99 KB);
  - the published `.sha256` file for `Color/Source_A.zip`;
  - the **last 256 KB** of each of the 10 microscopy archives, fetched with HTTP range requests. A ZIP's table of contents (its central directory) sits at the end of the file, so this tail is enough to list every file inside the archive.
- **Total downloaded:** about 2.7 MB, out of 10 GB of archives.

## Finding

**Nothing is missing.** The official `Microscopy/Optical/Stereo/Color/Source_A/Source_A.zip` contains **exactly 306 files, all `.tiff`**. They are the same 306 filenames as the local extract, with identical byte sizes and CRC-32 checksums.

The figure of 308 appears only in the prose of `README_stereo.txt`. No published archive matches it. It is most likely a **documentation error**: a count taken from an earlier working copy that had two more image-sized files. The evidence is in §5.

No specific missing filenames exist to report. The two extra files the README counted can't be named from any published material.

## 1. Local Source A directory

Path: `datasets/external/NIST_NIR_SORT/raw_download/NIST_SORT_DATA/NIST-Data-2026-11-27T05-50-1/mds2-3325/Microscopy/Optical/Stereo/Color/Source_A/Source_A/`

| Check | Result |
|---|---|
| All files, recursive, including hidden files | **306** |
| Image files (`.tiff`) | **306** |
| Non-image, hidden or zero-byte files | 0 |
| Unique sample IDs (first 23 characters of each filename) | **51** |
| Images per sample ID | 6 for every ID (replicates 01 to 06) |
| Total bytes | 3,270,580,330 (3.046 GiB) |

## 2. Official NIST metadata (`https://data.nist.gov/od/id/mds2-3325`)

The record came from the NIST PDR metadata service, `https://data.nist.gov/rmm/records?@id=ark:/88434/mds2-3325`.

**Record details:**
- Version **2.0.1**, a "metadata update" dated 2026-01-28. Earlier releases:
  - 2.0.0 (2026-02-06): "updated data, reorganized for easier downloads";
  - 1.0.0 (2025-09-18): initial release.
- The dates come from the record as published. 2.0.1 is listed with an earlier date than 2.0.0.
- Licence field: `https://www.nist.gov/open/license`.

**What the record publishes, file by file:**
- 170 entries.
- For every file: path, byte size, SHA-256 and download URL.
- Archives are listed at the per-source ZIP level. There is **no list of the files inside any ZIP**, no file count and no separate manifest file.
- The only occurrences of "306" or "308" in the record are inside unrelated SHA-256 hashes.

**Text counts:** the record's description repeats README_main: microscopy is "about 10.6 GB from ca. 1044 files".

**Local documentation against the published checksums:** all 18 local documentation and script files match NIST's published SHA-256 exactly. These include `README_stereo.txt`, `README_main.txt`, `SampleInformation.csv` and `ColorParameters.csv`. So the local README is the current official one, not a stale copy.

## 3. Published checksum, size and internal manifest for `Color/Source_A.zip`

| Item | Published value |
|---|---|
| Archive size | 3,209,087,156 bytes |
| SHA-256, from the record and from `Source_A.zip.sha256` | `2e5c54f256afe6afd712d0e6c666bb1114d54c9b5374bca8b3400e6e9d9d283d` |
| File count | Not published |
| Separate internal manifest | Not published |

The local `.zip` no longer exists; only the extracted folder remains. So the whole-archive SHA-256 can't be checked against the local copy. The per-file CRC-32 comparison in §4 covers integrity instead.

## 4. Official archive's internal listing compared with the local folder

| | Official archive (from its table of contents) | Local extract |
|---|---|---|
| Entries | **306** (the end-of-archive record also says 306; standard ZIP, not ZIP64) | 306 |
| Folder entries inside the archive | 0 (all files sit at the archive root) | — |
| File types | 306 × `.tiff` | 306 × `.tiff` |
| Non-image files | 0 | 0 |
| Total uncompressed bytes | 3,270,580,330 | 3,270,580,330 |
| Filenames only in this copy | — | — |
| Size or CRC-32 mismatches | **0 of 306** | |
| Entry timestamps | All 2025-12-24 11:10 | |

Every local file has the same name, size and CRC-32 as its entry in the official archive. The local Source A extract is a complete, exact copy of what NIST publishes.

The nested `Source_A/Source_A/` folder locally comes from how the archive was extracted. The archive itself has no folders inside.

## 5. Does 308 count non-image files, or is it a documentation error?

**Every other microscopy count in the README matches its archive exactly.** All archives below were read the same way:

| Archive | README_stereo.txt | Entries in official ZIP | Match |
|---|---|---|---|
| Color / Source_A | **308** | **306** (tiff) | **No** |
| Color / Source_B | 120 | 120 (tiff) | Yes |
| Color / Source_C | 66 | 66 (tiff) | Yes |
| Color / Source_D | 72 | 72 (tiff) | Yes |
| Mono / Source_A | 120 | 120 (bmp) | Yes |
| Mono / Source_B | 44 | 44 (bmp) | Yes |
| Mono / Source_C | 44 | 44 (bmp) | Yes |
| Mono / Source_D | 24 | 24 (tiff) | Yes |
| Polarized / Source_A | 236 | 236 (tiff) | Yes |
| Polarized / Source_B | 8 | 8 (tiff) | Yes |

None of the 10 archives contains any non-image file.

**Could the 308 include non-image files?** Not as published: the archive contains none. A count that took in stray files such as `Thumbs.db` or `desktop.ini` from the authors' working folder can't be ruled out completely. The size figure argues against it, though.

**Size figure:**
- For sources B, C and D, the README's sizes equal the actual totals cut off, not rounded:
  - B is 1.027 GiB, written "1.02 GB";
  - C is 666.5 MiB, written "666 MB";
  - D is 703.8 MiB, written "703 MB".
- On that convention, the 306 actual Source A files (3.046 GiB) would be written "3.04 GB".
- The README says **"3.06 GB"**. Two more files of average Source A size (about 10.2 MiB each) would give 3.066 GiB, which cuts off to exactly "3.06".
- Tiny non-image files would not move the total.
- So both the README's count and its size describe a folder with **two more image-sized files** than NIST published.

**The README's own total is consistent with 308:**
- The README colour, mono and polarised counts add up to 1,042.
- Adding the two microscopy documentation files gives 1,044, the "ca. 1044 files" in README_main and in the metadata description.
- With the actual 306, the same sum is 1,042.
- So the 1,044 total appears to come from the same 308 figure, not from an independent count.

**Timing:** the archive entries are dated 2025-12-24, while README_stereo.txt says it was written on 16 January 2026.

**Conclusion:** 308 is most likely a documentation error. It is probably a count and size taken from a working folder that held two extra images (for example, retakes) which were not packaged. This can't be proven from the published material.

Two more points rule out other explanations:
- **A whole missing swatch would mean 6 files**, not 2. The numbering gaps A06 and A34 appear nowhere in `SampleInformation.csv` or `ColorParameters.csv`.
- **All 51 listed Source A swatches have their full 01 to 06 set** in both copies.

## 6. Specific missing filenames

**None.** The official archive and the local folder hold identical sets of 306 filenames. Nothing published is absent locally. The two extra files implied by the README can't be identified from anything NIST publishes.

Only the dataset authors could confirm this. The contact listed in README_main is Amanda Forster or Katarina Goodge at NIST.

## Impact on the audit

No change is needed to `NIST_DATASET_AUDIT.md`'s counts:

- 51 Source A swatches;
- 94 swatches in total;
- 564 colour images.

The local copy is complete against the official release. Its §3 discrepancy can be marked resolved, as a documentation count error, the next time that file is edited. The audit file was **not** edited here.

---

### Method notes

- **Range requests:** each archive was read from `bytes=(size − 262144)–(size − 1)`. Every request returned HTTP 206 with 262,144 bytes. A small parser in the scratchpad read the end-of-archive record and table of contents from those bytes. The Source A table of contents is 31,518 bytes and starts at offset 3,209,055,616, well inside the fetched tail.
- **Local CRC-32:** computed by reading each local TIFF once, read-only.
- **Scratchpad files:** the metadata JSON, the archive tails and the parsed listings are in the session scratchpad. They are not part of the repository.
