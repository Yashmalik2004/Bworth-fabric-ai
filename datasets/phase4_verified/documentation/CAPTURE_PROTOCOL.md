# Capture protocol: photographing physical fabrics

Goal: 4–6 photographs per physical fabric, consistent enough to compare across fabrics, and varied enough
that a model cannot learn the photo session instead of the fabric.

**Several photographs of one fabric are one sample, not several.** Band targets (30 fabrics per band) count
fabrics, never photographs.

## 1. Required set per fabric

| # | `image_type` | Required | How |
|---|---|---|---|
| 1 | `full_fabric` | **yes** | Whole swatch flat, filling about 80 % of the frame, overhead (`overhead_90`). Shows colour, pattern repeat and edges. |
| 2 | `macro_texture` | **yes** | Close-up of the weave/knit, about 2–5 cm of fabric across the frame, sharp yarn detail. Use macro mode or a clip-on macro lens. Avoid seams, prints and damage unless they are the fabric. |
| 3 | `natural_light` | recommended | Indirect daylight near a window, no direct sun, overhead. |
| 4 | `alternate_angle` | recommended | About 45° (`angled_45`) to show sheen and surface relief. |
| 5–6 | `controlled` | optional | Controlled lighting box or 5000 K LED (`controlled_led_5000k`), fixed distance. Use for colour-critical shots or a second macro area. |

## 2. Conditions
* **Flat and clean:** remove lint; flatten creases (do not iron if it changes the surface).
* **Background:** plain neutral grey or white, not part of the frame in macro shots.
* **No filters**, beauty modes or HDR "enhance". Turn off digital zoom, and move the camera closer instead.
* **Focus and exposure:** tap to focus on the fabric; retake blurred or over-exposed shots.
* **Resolution:** at least 12 MP for full shots. Save JPEG at high quality, or PNG/TIFF.
* **Vary the session:** do not photograph all fabrics of one band in one session or under one light.
  Interleave bands, so lighting and device are not correlated with composition.
* **One fabric per photograph.** No labels, hands or tags in frame, except a separate reference shot of the tag.
  Tag photos go in the verification evidence, not in `image_registry`.

## 3. Recording
For each batch, the entry app (`streamlit run data_collection/phase4/entry_app.py`, tab 2) records a **capture
session id** (one id per photo session; used to check that sessions are not correlated with band), and:
* the fabric ID, image type, lighting, camera angle, capture device and date;
* the SHA-256 and resolution, computed automatically.

Each photo is stored as `images/<fabric_id>/<image_id>.<ext>`. Never rename or edit files after registration:
the validator checks every file against its SHA-256.

## 4. Do not
* Re-use one photograph for two fabrics. The app rejects identical files, and the validator flags near-identical
  photos across fabrics.
* Crop one photo into several "different" photographs.
* Photograph garments with mixed panels unless the photographed panel is the verified fabric.
* Include photographs taken before the fabric had a `fabric_id`. Register first, then photograph.
