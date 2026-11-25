"""BWorth Phase 4: data-entry interface for verified physical fabric samples.

    streamlit run data_collection/phase4/entry_app.py

Register physical fabrics, attach photographs and verification evidence, and check dataset status.
Incomplete records are saved as pending. Verification status is computed by rules
(data_collection/phase4/registry.py) and cannot be set by hand; only VERIFIED fabrics reach the
training manifest. Separate from the main BWorth demo app.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_collection.phase4 import registry as R  # noqa: E402
from data_collection.phase4.validate import validate, write_outputs  # noqa: E402

st.set_page_config(page_title="BWorth · Verified fabric collection", page_icon="🧵", layout="wide")
P = R.init_dataset()

st.title("Verified fabric collection")
st.caption(f"Dataset: `{P['root']}` · one entry per **physical fabric**; photographs and evidence attach to it.")
st.info("A composition label counts only when backed by a laboratory test, traceable manufacturer documentation, or "
        "supplier documentation identifying the exact product. **Garment care labels are recorded but never verify a "
        "label.** Incomplete entries are saved as *pending verification* and stay out of the training manifest.",
        icon="🔎")


def opt_blank(options: list[str]) -> list[str]:
    return [""] + options


fab, img, ver = R.load()
fabric_ids = fab.fabric_id.tolist()
tab_fab, tab_img, tab_ver, tab_status = st.tabs(["1 · Register fabric", "2 · Add photographs", "3 · Add verification evidence",
                                                 "4 · Dataset status"])

# --------------------------------------------------------------------------- #
with tab_fab:
    st.subheader("Register a new physical fabric")
    st.caption("Leave fields blank if unknown; the record is saved as pending. Percentages are of the whole fabric.")
    with st.form("fabric_form", clear_on_submit=True):
        c1, c2, c3 = st.columns(3)
        cotton = c1.text_input("Cotton % (as currently claimed)", placeholder="e.g. 65")
        other = c2.text_input("Other fibres %", placeholder="e.g. 35")
        other_comp = c3.text_input("Other fibre composition", placeholder="polyester:30|elastane:5")
        c4, c5, c6 = st.columns(3)
        ftype = c4.text_input("Fabric type", placeholder="e.g. poplin, jersey, denim")
        wk = c5.selectbox("Weave or knit", opt_blank(R.WEAVE_OR_KNIT))
        colour = c6.text_input("Colour", placeholder="e.g. navy")
        c7, c8 = st.columns(2)
        source = c7.text_input("Source (who supplied the fabric)", placeholder="e.g. Supplier X, lot 2026-11-A")
        location = c8.text_input("Physical sample location", placeholder="e.g. Box 3, sleeve 12")
        c9, c10 = st.columns(2)
        vmethod = c9.selectbox("Claimed verification method", opt_blank(R.EVIDENCE_TYPES),
                               help="Informational. Status is computed from the evidence records in tab 3.")
        cdate = c10.date_input("Collection date", value=date.today())
        notes = st.text_area("Notes", height=70)
        submitted = st.form_submit_button("Save fabric", type="primary")
    if submitted:
        try:
            fid = R.add_fabric({"cotton_percentage": cotton.strip(), "other_fibre_percentage": other.strip(),
                                "other_fibre_composition": other_comp.strip(), "fabric_type": ftype.strip(),
                                "weave_or_knit": wk, "colour": colour.strip(), "source": source.strip(),
                                "physical_sample_location": location.strip(), "verification_method": vmethod,
                                "collection_date": cdate.isoformat(), "notes": notes.strip()})
            st.success(f"Saved **{fid}** (status: pending verification until accepted evidence is added). "
                       f"Write **{fid}** on the physical sample's tag.")
            fab, img, ver = R.load(); fabric_ids = fab.fabric_id.tolist()
        except ValueError as exc:
            st.error(str(exc))

# --------------------------------------------------------------------------- #
with tab_img:
    st.subheader("Add photographs of one fabric")
    st.caption(f"Protocol: {R.MIN_IMAGES_PER_FABRIC}–{R.MAX_IMAGES_PER_FABRIC} photographs per fabric, including a full-fabric "
               "and a macro-texture shot. More photographs of the same fabric do not add independent samples.")
    if not fabric_ids:
        st.warning("Register a fabric first.")
    else:
        with st.form("image_form", clear_on_submit=True):
            fid = st.selectbox("Fabric", fabric_ids, key="img_fabric")
            files = st.file_uploader("Photographs (same type and conditions)", type=["jpg", "jpeg", "png", "tif", "tiff", "webp"],
                                     accept_multiple_files=True)
            c1, c2, c3 = st.columns(3)
            itype = c1.selectbox("Image type", R.IMAGE_TYPES)
            light = c2.selectbox("Lighting", R.LIGHTING)
            angle = c3.selectbox("Camera angle", R.CAMERA_ANGLES)
            c4, c5 = st.columns(2)
            device = c4.text_input("Capture device", placeholder="e.g. Pixel 8, main camera")
            cdate = c5.date_input("Capture date", value=date.today(), key="img_date")
            inotes = st.text_input("Notes")
            ok = st.form_submit_button("Save photographs", type="primary")
        if ok:
            if not files:
                st.error("Choose at least one photograph.")
            for f in files or []:
                try:
                    iid = R.add_image(fid, f.name, f.getvalue(), {"capture_device": device.strip(), "image_type": itype,
                                                                  "lighting": light, "camera_angle": angle,
                                                                  "capture_date": cdate.isoformat(), "notes": inotes.strip()})
                    st.success(f"{f.name} → {iid}")
                except ValueError as exc:
                    st.error(f"{f.name}: {exc}")
            fab, img, ver = R.load()
        if fabric_ids:
            sub = img[img.fabric_id == st.session_state.get("img_fabric", fabric_ids[0])]
            st.caption(f"{len(sub)} photograph(s) registered for this fabric: "
                       + (", ".join(f"{r.image_id} ({r.image_type})" for r in sub.itertuples()) or "none"))

# --------------------------------------------------------------------------- #
with tab_ver:
    st.subheader("Record verification evidence")
    if not fabric_ids:
        st.warning("Register a fabric first.")
    else:
        with st.form("ver_form", clear_on_submit=True):
            fid = st.selectbox("Fabric", fabric_ids, key="ver_fabric")
            etype = st.selectbox("Evidence type", R.EVIDENCE_TYPES,
                                 format_func=lambda k: f"{k} — {(R.ACCEPTED_EVIDENCE | R.NON_ACCEPTED_EVIDENCE)[k]}")
            c1, c2, c3 = st.columns(3)
            ref = c1.text_input("Evidence reference", placeholder="lab report no. / document no.")
            issuer = c2.text_input("Issuing party", placeholder="laboratory / manufacturer / supplier")
            product = c3.text_input("Product identifier", placeholder="article / SKU / lot (required for documentation)")
            c4, c5, c6 = st.columns(3)
            rcot = c4.text_input("Reported cotton %")
            rother = c5.text_input("Reported other fibres", placeholder="polyester:35")
            method = c6.text_input("Test method / standard", placeholder="e.g. ISO 1833-11 (lab tests)")
            c7, c8, c9 = st.columns(3)
            reviewer = c7.text_input("Reviewed by")
            rdate = c8.date_input("Review date", value=date.today())
            decision = c9.selectbox("Review decision", ["pending", "accepted", "rejected"])
            doc = st.file_uploader("Evidence document (PDF or image)", type=["pdf", "jpg", "jpeg", "png"])
            vnotes = st.text_area("Notes", height=70, key="ver_notes")
            ok = st.form_submit_button("Save evidence", type="primary")
        if ok:
            if etype in R.NON_ACCEPTED_EVIDENCE and decision == "accepted":
                st.warning(f"'{etype}' is recorded, but it can never verify a composition label.")
            try:
                rid = R.add_verification({"fabric_id": fid, "evidence_type": etype, "evidence_reference": ref.strip(),
                                          "issuing_party": issuer.strip(), "product_identifier": product.strip(),
                                          "reported_cotton_percentage": rcot.strip(), "reported_other_fibres": rother.strip(),
                                          "test_method_standard": method.strip(), "reviewed_by": reviewer.strip(),
                                          "review_date": rdate.isoformat(), "review_decision": decision, "notes": vnotes.strip()},
                                         document=(doc.name, doc.getvalue()) if doc else None)
                fab, img, ver = R.load()
                s = R.fabric_statuses(fab[fab.fabric_id == fid], ver, P["root"]).iloc[0]
                st.success(f"Saved {rid}. Fabric {fid} is now **{s.status}**" + (f" — {s.reasons}" if s.reasons else ""))
            except ValueError as exc:
                st.error(str(exc))

# --------------------------------------------------------------------------- #
with tab_status:
    st.subheader("Dataset status")
    if st.button("Run validation and update audit report", type="primary"):
        res = validate()
        out = write_outputs(res)
        st.session_state["val"] = res
        st.success(f"Audit report written to `{out['report'].relative_to(PROJECT_ROOT)}`; "
                   f"training manifest: {res['counts']['fabrics_in_manifest']} fabrics / {res['counts']['photographs_in_manifest']} photographs.")
    res = st.session_state.get("val")
    if res is None:
        st.caption("Press the button to validate the dataset.")
    else:
        st.markdown("##### Independent verified fabrics per band")
        st.dataframe(pd.DataFrame([{"Band": R.BAND_LABELS[b], "Verified fabrics": v["verified_fabrics_in_manifest"],
                                    "Target": v["target_fabrics"], "Pending": v["pending_fabrics"],
                                    "Photographs": v["verified_photographs_in_manifest"]}
                                   for b, v in res["bands"].items()]), hide_index=True, width="stretch")
        iss = res["issues"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Errors", int((iss.severity == "ERROR").sum()))
        c2.metric("Pending", int((iss.severity == "PENDING").sum()))
        c3.metric("Warnings", int((iss.severity == "WARNING").sum()))
        st.markdown("##### Fabric status")
        st.dataframe(res["status"], hide_index=True, width="stretch")
        st.markdown("##### Issues")
        st.dataframe(iss, hide_index=True, width="stretch")
    st.markdown("##### Registered fabrics")
    st.dataframe(fab, hide_index=True, width="stretch")
