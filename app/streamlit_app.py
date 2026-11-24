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

from configs.config import DEFAULT_CHECKPOINT, IMAGES_DIR, cotton_label  # noqa: E402
from inference.composition.predict import (  # noqa: E402
    CheckpointNotFoundError,
    CompositionPredictor,
    InvalidCheckpointError,
    InvalidImageError,
)
from utils.image_utils import load_rgb_image  # noqa: E402

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
      .bw-result {{ font-size: 2.6rem; font-weight: 700; color: {ACCENT}; line-height: 1.1; }}
      .bw-score {{ font-size: 1rem; color: #0b0b0b; margin-top: .5rem; }}
      .bw-note {{ font-size: .85rem; color: #52514e; }}
      .bw-pill {{ display: inline-block; padding: .15rem .6rem; border-radius: 999px; font-size: .78rem;
                  background: #fff4e5; color: #8a5300; border: 1px solid #f3d19c; }}
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
def load_test_samples(splits_path: str) -> pd.DataFrame:
    p = Path(splits_path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    if not p.is_file():
        return pd.DataFrame()
    df = pd.read_csv(p)
    return df[df.split == "test"].reset_index(drop=True)


def _mtime(p: Path) -> float:
    return p.stat().st_mtime if p.exists() else 0.0


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
        '<p class="bw-sub">Estimate the cotton-percentage category of a fabric from a photo — '
        "an experimental AI demo for the BWorth textile-recycling programme.</p>",
        unsafe_allow_html=True,
    )

st.warning(
    "**Experimental AI estimate.** Predictions come from an image classifier trained on a small public "
    "dataset (CottonFabricImageBD, 13 cotton-percentage categories). They are not a laboratory or chemical "
    "measurement of fibre composition, and the score is the model's top class probability, not a "
    "calibrated confidence.",
    icon="⚠️",
)

# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #
ckpt_path = DEFAULT_CHECKPOINT
predictor: CompositionPredictor | None = None
load_error: str | None = None
try:
    predictor = get_predictor(str(ckpt_path), _mtime(ckpt_path))
except CheckpointNotFoundError:
    load_error = "missing"
except InvalidCheckpointError as exc:
    load_error = str(exc)

if predictor is None:
    st.error("No trained model is available, so predictions are disabled.", icon="🚫")
    if load_error and load_error != "missing":
        st.caption(f"Details: {load_error}")
    st.markdown("Train the model from the project root, then reload this page:")
    st.code(
        "python -m training.composition.validate_dataset\n"
        "python -m training.composition.train",
        language="bash",
    )
    st.stop()

info = predictor.info
cfg = predictor.checkpoint.get("config", {})
eval_dir = PROJECT_ROOT / "evaluation" / "composition" / cfg.get("run_name", "")
test_metrics = load_json(str(eval_dir / "metrics_test.json"), _mtime(eval_dir / "metrics_test.json"))

# --------------------------------------------------------------------------- #
# Sidebar: model card
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.subheader("Model")
    st.markdown(
        f"- **Architecture:** ResNet-50 (ImageNet transfer learning)\n"
        f"- **Classes:** {info['num_classes']} cotton-percentage categories\n"
        f"- **Input:** {info['image_size']}×{info['image_size']} px\n"
        f"- **Best epoch:** {info['best_epoch']}\n"
        f"- **Device:** {info['device']}"
    )
    st.subheader("Held-out test performance")
    if test_metrics:
        c1, c2 = st.columns(2)
        c1.metric("Accuracy", f"{test_metrics['accuracy'] * 100:.1f}%")
        c2.metric("Top-3 acc.", f"{test_metrics['top3_accuracy'] * 100:.1f}%")
        c1.metric("Macro F1", f"{test_metrics['macro_f1']:.3f}")
        c2.metric("Mean error", f"{test_metrics['mean_abs_error_cotton_pct_points']:.1f} pts")
        st.caption(
            f"{test_metrics['n_images']} test images from fabric swatches not seen in training "
            f"({test_metrics['split_strategy']} split). Expect lower accuracy on real-world photos "
            "that differ from the dataset."
        )
    else:
        st.caption("No test metrics found. Run `python -m training.composition.evaluate`.")
    st.subheader("Categories")
    st.caption(", ".join(cotton_label(c) for c in predictor.classes))

# --------------------------------------------------------------------------- #
# Input
# --------------------------------------------------------------------------- #
left, right = st.columns([5, 6], gap="large")

with left:
    st.markdown("#### 1 · Provide a fabric image")
    tab_up, tab_cam, tab_sample = st.tabs(["Upload", "Camera", "Held-out test sample"])
    source = None
    true_label = None
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
        samples = load_test_samples(cfg.get("splits_path", ""))
        if samples.empty:
            st.caption("Test split file not found.")
        else:
            st.caption("Images from the held-out test split — never used for training or model selection.")
            idx = st.selectbox(
                "Sample",
                options=list(range(len(samples))),
                format_func=lambda i: f"#{i + 1:03d} · true: {cotton_label(samples.label[i])}",
                index=None,
                placeholder="Choose a test image…",
            )
            if idx is not None:
                p = PROJECT_ROOT / samples.path[idx]
                source = ("sample", p.read_bytes())
                true_label = samples.label[idx]

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
        key = hash(source[1])
        if run:
            with st.spinner("Analysing texture…"):
                st.session_state["result"] = (key, predictor.predict(image), true_label)
        stored = st.session_state.get("result")
        if stored and stored[0] == key:
            _, result, t_label = stored
            st.markdown(
                f"""
                <div class="bw-card">
                  <div class="bw-kicker">Predicted category</div>
                  <div class="bw-result">{result.predicted_label}</div>
                  <div class="bw-score">Prediction score: <b>{result.score * 100:.1f}%</b>
                    <span class="bw-note">(top class probability, uncalibrated)</span></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if t_label is not None:
                ok = t_label == result.predicted_class
                (st.success if ok else st.error)(
                    f"Dataset label: **{cotton_label(t_label)}** — prediction is {'correct' if ok else 'incorrect'}."
                )
            if result.score < 0.4:
                st.markdown(
                    '<span class="bw-pill">Low score — the model is uncertain between several categories</span>',
                    unsafe_allow_html=True,
                )

            st.markdown("##### Probability across all categories")
            prob_df = pd.DataFrame(
                {
                    "Category": [cotton_label(c) for c in result.probabilities],
                    "Probability": list(result.probabilities.values()),
                    "Predicted": [c == result.predicted_class for c in result.probabilities],
                }
            )
            order = [cotton_label(c) for c in predictor.classes]
            chart = (
                alt.Chart(prob_df)
                .mark_bar(cornerRadiusEnd=4, height=16)
                .encode(
                    y=alt.Y("Category:N", sort=order, title=None),
                    x=alt.X("Probability:Q", axis=alt.Axis(format="%", grid=True), scale=alt.Scale(domain=[0, 1]),
                            title=None),
                    color=alt.condition(alt.datum.Predicted, alt.value(ACCENT), alt.value(MUTED_BAR)),
                    tooltip=[alt.Tooltip("Category:N"), alt.Tooltip("Probability:Q", format=".1%")],
                )
                .properties(height=13 * 26)
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
                "Categories are the cotton percentages present in the training dataset; the model cannot output "
                "values outside them and does not estimate the other fibres in the blend."
            )
