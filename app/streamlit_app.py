"""BWorth Fabric Intelligence — cotton-composition demo.

Run from the project root:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import APP_MODELS, APP_SAMPLE_SPLITS, EVALUATION_DIR, FINE_TO_BAND, IMAGES_DIR, display_label  # noqa: E402
from inference.composition.predict import (  # noqa: E402
    CheckpointNotFoundError,
    CompositionPredictor,
    InvalidCheckpointError,
    InvalidImageError,
)
from utils.image_utils import load_rgb_image  # noqa: E402
from app.phase3_model import DISCLAIMER as P3_DISCLAIMER  # noqa: E402
from app.phase3_model import LOW_NOTE as P3_LOW_NOTE  # noqa: E402
from app.phase3_model import RANGE_NOTE as P3_RANGE_NOTE  # noqa: E402
from app.phase3_model import SELECTION_FILE as P3_SELECTION_FILE  # noqa: E402
from app.phase3_model import app_spec as p3_app_spec  # noqa: E402
from app.phase3_model import load_selection as p3_load_selection  # noqa: E402
from app.phase5_model import SYNTHETIC_WARNING as P5_SYNTHETIC_WARNING  # noqa: E402
from app.phase5_model import app_spec as p5_app_spec  # noqa: E402
from app.phase5_model import load_selection as p5_load_selection  # noqa: E402
from app.phase5_model import selection_file as p5_selection_file  # noqa: E402

ACCENT = "#2a78d6"
MUTED_BAR = "#c9c8c2"
LOGO = IMAGES_DIR / "Bworth.jpg"

st.set_page_config(
    page_title="BWorth · Fabric Intelligence",
    page_icon=str(LOGO) if LOGO.is_file() else "🧵",
    layout="wide",
)

st.markdown(
    f"""
    <style>
      .block-container {{ padding-top: 2rem; max-width: 1200px; }}
      .bw-title {{ font-size: 2rem; font-weight: 700; margin: 0; line-height: 1.2; }}
      .bw-sub {{ color: #52514e; margin-top: .25rem; font-size: 1rem; }}
      .bw-card {{ border: 1px solid #e6e5e0; border-radius: 12px; padding: 1.25rem 1.5rem; background: #fcfcfb; }}
      .bw-kicker {{ text-transform: uppercase; letter-spacing: .06em; font-size: .75rem; color: #52514e; margin-bottom: .25rem; }}
      .bw-result {{ font-size: 2.4rem; font-weight: 700; color: {ACCENT}; line-height: 1.1; }}
      .bw-score {{ font-size: 1rem; color: #0b0b0b; margin-top: .5rem; }}
      .bw-note {{ font-size: .85rem; color: #52514e; }}
      .bw-pill {{ display: inline-block; padding: .15rem .6rem; border-radius: 999px; font-size: .78rem;
                  background: #fff4e5; color: #8a5300; border: 1px solid #f3d19c; }}
      .bw-badge {{ display: inline-block; padding: .1rem .55rem; border-radius: 6px; font-size: .72rem;
                   font-weight: 700; letter-spacing: .05em; border: 1px solid #52514e; color: #0b0b0b; }}
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# Cached resources
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading model…")
def get_predictor(path: str, mtime: float) -> CompositionPredictor:  # mtime busts the cache on retrain
    return CompositionPredictor(path)


@st.cache_data
def load_json(path: str, mtime: float) -> dict | None:
    p = Path(path)
    return json.loads(p.read_text()) if p.is_file() else None


@st.cache_data
def load_test_samples(splits_path: str, mtime: float) -> pd.DataFrame:
    p = Path(splits_path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    if not p.is_file():
        return pd.DataFrame()
    df = pd.read_csv(p)
    return df[df.split == "test"].reset_index(drop=True)


def _mtime(p: Path) -> float:
    p = Path(p)
    return p.stat().st_mtime if p.exists() else 0.0


@st.cache_data
def phase3_selection(sel_mtime: float) -> dict | None:
    """Recorded Phase 3 checkpoint choice; None if missing or the checkpoint's SHA-256 does not match."""
    return p3_load_selection(verify_hash=True)


@st.cache_data
def phase5_selection(sel_path: str, sel_mtime: float) -> dict | None:
    """Phase 5 model record; None (model not offered) if absent or the checkpoint hash does not match."""
    return p5_load_selection(verify_hash=True)


def pct(x: float) -> str:
    return f"{x * 100:.1f}%"


# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
h1, h2 = st.columns([1, 6], vertical_alignment="center")
with h1:
    if LOGO.is_file():
        st.image(str(LOGO), width=110)
with h2:
    st.markdown('<p class="bw-title">Fabric Intelligence · Cotton Composition</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="bw-sub">Estimate the cotton content of a fabric from a photo — '
        "an experimental AI demo for the BWorth textile-recycling programme.</p>",
        unsafe_allow_html=True,
    )

st.warning(
    "**Experimental AI estimate.** Predictions come from an image classifier trained on a small public "
    "dataset (CottonFabricImageBD). They are not a laboratory or chemical measurement of fibre composition, "
    "and the score is the model's top class probability, not a calibrated confidence.",
    icon="⚠️",
)

# --------------------------------------------------------------------------- #
# Model selection (sidebar)
# --------------------------------------------------------------------------- #
available = {k: m for k, m in APP_MODELS.items() if _mtime(m["checkpoint"]) and Path(m["checkpoint"]).stat().st_size}
_p3 = phase3_selection(_mtime(P3_SELECTION_FILE))
if _p3 is not None:
    available["phase3_band3"] = p3_app_spec(_p3)
_p5 = phase5_selection(str(p5_selection_file()), _mtime(p5_selection_file()))
if _p5 is not None:                       # dormant until a Phase 5 model is approved and selected
    available["phase5_band4"] = p5_app_spec(_p5)
cv_summary = load_json(str(EVALUATION_DIR / "cv" / "cv_summary.json"), _mtime(EVALUATION_DIR / "cv" / "cv_summary.json"))

with st.sidebar:
    st.subheader("Model")
    if not available:
        model_key = None
    else:
        model_key = st.radio(
            "Choose the model",
            options=list(available),
            format_func=lambda k: f"{available[k]['title']}  ·  {available[k]['badge'].lower()}",
            index=0,
            help="The 13-class baseline is the default. The 4-class and 3-class models are experiments.",
        )

if model_key is None:
    st.error("No trained model is available, so predictions are disabled.", icon="🚫")
    st.markdown("Train the model from the project root, then reload this page:")
    st.code("python -m training.composition.validate_dataset\npython -m training.composition.train", language="bash")
    st.stop()

spec = available[model_key]
try:
    predictor = get_predictor(str(spec["checkpoint"]), _mtime(spec["checkpoint"]))
except (CheckpointNotFoundError, InvalidCheckpointError) as exc:
    st.error(f"The selected model could not be loaded: {exc}", icon="🚫")
    st.stop()

info = predictor.info
is_p3 = spec.get("kind") == "phase3_band3"
is_p5 = spec.get("kind") == "phase5_band4"
is_band = info["label_scheme"] in ("band4", "band3_stageB", "phase5_band4")
metrics = load_json(str(spec["metrics"]), _mtime(spec["metrics"]))
cv_entry = (cv_summary or {}).get("configs", {}).get(spec["cv_config"]) if spec.get("cv_config") else None


def lab(class_name: str) -> str:
    """Display label for the selected model (Phase 3 uses its own exact wording)."""
    return spec.get("labels", {}).get(class_name, display_label(class_name))


with st.sidebar:
    st.markdown(f'<span class="bw-badge">{spec["badge"]}</span>', unsafe_allow_html=True)
    st.caption(spec["description"])
    st.markdown(
        f"- **Architecture:** {'ResNet-50' if info['architecture'] == 'resnet50' else info['architecture']} (ImageNet transfer learning)\n"
        f"- **Output:** {info['num_classes']} {'cotton bands' if is_band else 'cotton-percentage categories'}\n"
        f"- **Colour augmentation:** {info['color_aug']}\n"
        f"- **Input:** {info['image_size']}×{info['image_size']} px · **Device:** {info['device']}"
    )
    st.subheader("Unseen-fabric performance" if (is_p3 or is_p5) else "Unseen-swatch performance")
    if is_p5:
        cv = spec.get("cv") or {}
        if "balanced_accuracy_mean" in cv:
            st.metric("Fabric balanced acc.", f"{pct(cv['balanced_accuracy_mean'])} ± {cv.get('balanced_accuracy_std', 0) * 100:.1f}")
            if cv.get("pooled_ci95"):
                st.caption(f"95% CI (fabric bootstrap): {pct(cv['pooled_ci95'][0])} – {pct(cv['pooled_ci95'][1])}. "
                           "Chance level is 25% balanced accuracy.")
        else:
            st.caption("No cross-validated Phase 5 results recorded for this model.")
        if spec.get("synthetic"):
            st.caption(P5_SYNTHETIC_WARNING)
    elif is_p3:
        cv = spec["cv"]
        c1, c2 = st.columns(2)
        c1.metric("Balanced acc.", f"{pct(cv['balanced_accuracy_mean'])} ± {cv['balanced_accuracy_std'] * 100:.1f}")
        c2.metric("Accuracy", f"{pct(cv['accuracy_mean'])} ± {cv['accuracy_std'] * 100:.1f}")
        st.metric("Macro F1", f"{cv['macro_f1_mean']:.3f} ± {cv['macro_f1_std']:.3f}")
        st.caption(
            f"5-fold cross-validation on physical fabrics never seen in training, {cv['runs']} runs (mean ± std). "
            "Chance level is 33.3% balanced accuracy. Per class (pooled): "
            + " · ".join(f"{k}: precision {v['precision']:.2f}, recall {v['recall']:.2f}" for k, v in cv["per_class"].items())
            + ". The selected checkpoint's own validation score is not shown because its validation fold was the easiest."
        )
        fab = spec["selection"]["fabrics_per_class"]
        st.caption("Independent fabrics in the data: " + " · ".join(f"{lab(c).split(':')[0]} {n}" for c, n in fab.items()))
    elif metrics:
        own = "13 classes" if not is_band else "4 bands"
        c1, c2 = st.columns(2)
        c1.metric(f"Accuracy ({own})", pct(metrics["accuracy"]))
        c2.metric("Balanced acc.", pct(metrics["balanced_accuracy"]))
        if "band4" in metrics:
            c1.metric("Scored in 4 bands", pct(metrics["band4"]["accuracy"]))
            c2.metric("4-band bal. acc.", pct(metrics["band4"]["balanced_accuracy"]))
        st.caption(f"Held-out test split v2: {metrics['n_images']} images of fabric swatches not used in training.")
    else:
        st.caption("No test metrics found for this model.")
    if cv_entry and "band4_balanced_accuracy_mean" in cv_entry:
        st.metric(
            "5-fold CV, 4-band balanced acc.",
            f"{pct(cv_entry['band4_balanced_accuracy_mean'])} ± {cv_entry['band4_balanced_accuracy_std'] * 100:.1f}",
        )
        st.caption("Same training recipe retrained on 5 swatch-grouped folds (mean ± std across folds).")
    st.caption("Expect lower accuracy on real-world photos that differ from the dataset.")
    st.subheader("Categories")
    st.caption(" · ".join(lab(c) for c in predictor.classes))
    if is_p3:
        st.caption(P3_RANGE_NOTE)
        st.caption(P3_LOW_NOTE)
    if is_p5:
        st.caption(spec["range_note"])

st.info(f"Model in use: **{spec['title']}** ({spec['badge'].lower()}) — {spec['description']}", icon="🧠")
if is_p3:
    st.error(P3_DISCLAIMER, icon="⚠️")
    st.caption(P3_LOW_NOTE)
if is_p5:
    if spec.get("synthetic"):
        st.error(P5_SYNTHETIC_WARNING, icon="🧪")
    st.error(spec["disclaimer"], icon="⚠️")

# --------------------------------------------------------------------------- #
# Input
# --------------------------------------------------------------------------- #
left, right = st.columns([5, 6], gap="large")

with left:
    st.markdown("#### 1 · Provide a fabric image")
    tab_up, tab_cam, tab_sample = st.tabs(["Upload", "Camera", "Held-out test sample"])
    source = None
    true_fine = None
    with tab_up:
        up = st.file_uploader("Upload a close-up photo of the fabric", type=["jpg", "jpeg", "png", "webp", "bmp"])
        if up is not None:
            source = ("upload", up.getvalue())
    with tab_cam:
        st.caption("Fill the frame with the fabric surface, in even light, without folds or shadows.")
        cam = st.camera_input("Capture fabric photo", label_visibility="collapsed")
        if cam is not None:
            source = ("camera", cam.getvalue())
    with tab_sample:
        sample_file = spec.get("sample_splits", APP_SAMPLE_SPLITS)
        if is_p5:   # Phase 5 held-out fabrics are in the sealed lockbox; unverified samples are never shown as truth
            samples = pd.DataFrame()
        else:
            samples = load_test_samples(str(sample_file), _mtime(sample_file))
        if samples.empty:
            st.caption("Test split file not found (run `python -m training.composition.experiment_splits`).")
        else:
            st.caption(
                "Photographs from this checkpoint's own held-out test fold — physical fabrics it never saw in training."
                if is_p3 else "Images from the held-out v2 test split — never used to train either model.")
            idx = st.selectbox(
                "Sample",
                options=list(range(len(samples))),
                format_func=lambda i: f"#{i + 1:03d} · true: {lab(samples.label[i])}",
                index=None,
                placeholder="Choose a test image…",
            )
            if idx is not None:
                source = ("sample", (PROJECT_ROOT / samples.path[idx]).read_bytes())
                true_fine = samples.label[idx]

    image = None
    if source is not None:
        try:
            image = load_rgb_image(source[1])
        except InvalidImageError as exc:
            st.error(f"That file could not be read as an image. {exc}")
    if image is not None:
        st.image(image, caption=f"Preview · {image.size[0]}×{image.size[1]} px", width="stretch")

# --------------------------------------------------------------------------- #
# Prediction
# --------------------------------------------------------------------------- #
with right:
    st.markdown("#### 2 · Composition estimate")
    if image is None:
        st.info("Upload a photo, take one with the camera, or pick a held-out test sample to begin.")
    else:
        run = st.button("Analyse fabric", type="primary", width="stretch")
        key = (hash(source[1]), model_key)
        if run:
            with st.spinner("Analysing texture…"):
                st.session_state["result"] = (key, predictor.predict(image), true_fine)
        stored = st.session_state.get("result")
        if stored and stored[0] == key:
            _, result, t_fine = stored
            kicker = "Predicted cotton band" if is_band else "Predicted cotton category"
            st.markdown(
                f"""
                <div class="bw-card">
                  <div class="bw-kicker">{kicker} · {spec['title']}</div>
                  <div class="bw-result">{lab(result.predicted_class)}</div>
                  <div class="bw-score">Prediction score: <b>{result.score * 100:.1f}%</b>
                    <span class="bw-note">(top class probability, uncalibrated)</span></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if is_p3:
                st.warning(P3_DISCLAIMER, icon="⚠️")
            if is_p5:
                st.warning(spec["disclaimer"], icon="⚠️")
            if t_fine is not None:
                if is_p3:                      # Phase 3 sample files already carry the band label
                    t_class, extra = t_fine, ""
                else:
                    t_class = FINE_TO_BAND[t_fine] if is_band else t_fine
                    extra = f" ({display_label(t_fine)})" if is_band else ""
                ok = t_class == result.predicted_class
                (st.success if ok else st.error)(
                    f"Dataset label: **{lab(t_class)}**{extra} — prediction is "
                    f"{'correct' if ok else 'incorrect'}."
                )
            if result.score < (0.5 if is_band else 0.4):
                st.markdown(
                    '<span class="bw-pill">Low score — the model is uncertain between several categories</span>',
                    unsafe_allow_html=True,
                )

            st.markdown("##### Probability across all categories")
            order = [lab(c) for c in predictor.classes]
            prob_df = pd.DataFrame(
                {
                    "Category": order,
                    "Probability": [result.probabilities[c] for c in predictor.classes],
                    "Predicted": [c == result.predicted_class for c in predictor.classes],
                }
            )
            chart = (
                alt.Chart(prob_df)
                .mark_bar(cornerRadiusEnd=4, height=16 if not is_band else 28)
                .encode(
                    y=alt.Y("Category:N", sort=order, title=None, axis=alt.Axis(labelLimit=220)),
                    x=alt.X("Probability:Q", axis=alt.Axis(format="%", grid=True), scale=alt.Scale(domain=[0, 1]),
                            title=None),
                    color=alt.condition(alt.datum.Predicted, alt.value(ACCENT), alt.value(MUTED_BAR)),
                    tooltip=[alt.Tooltip("Category:N"), alt.Tooltip("Probability:Q", format=".1%")],
                )
                .properties(height=len(order) * (26 if not is_band else 44))
            )
            st.altair_chart(chart, width="stretch")
            with st.expander("Show as table"):
                st.dataframe(
                    prob_df.drop(columns="Predicted").assign(Probability=lambda d: (d.Probability * 100).round(2))
                    .rename(columns={"Probability": "Probability (%)"}),
                    hide_index=True,
                    width="stretch",
                )
            st.caption(
                ("Bands group the cotton percentages present in the training dataset. "
                 if is_band else "Categories are the cotton percentages present in the training dataset; ")
                + "the model cannot estimate values outside them and says nothing about the other fibres in the blend."
            )
            if is_p3:
                st.caption(P3_RANGE_NOTE + " Probabilities are uncalibrated softmax outputs, not confidence levels. "
                           + P3_LOW_NOTE)
            if is_p5:
                st.caption(spec["range_note"] + " Probabilities are softmax outputs; treat them as uncalibrated unless "
                           "the model record states a measured calibration error.")
