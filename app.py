"""SmartPlate AI Streamlit interface."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

import streamlit as st
from PIL import Image

from smartplate.inference import SmartPlatePredictor


APP_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = APP_DIR / "artifacts"

st.set_page_config(
    page_title="SmartPlate AI",
    page_icon="🍽️",
    layout="wide",
)


@st.cache_resource
def load_predictor() -> SmartPlatePredictor:
    return SmartPlatePredictor(ARTIFACTS_DIR)


st.title("🍽️ SmartPlate AI")
st.caption("Food recognition and direct calorie/protein estimation using transfer learning")

prediction_tab, about_tab = st.tabs(["Predict", "About the project"])

with about_tab:
    st.subheader("How it works")
    st.write(
        "SmartPlate uses an ImageNet-pretrained EfficientNetV2B0 backbone fine-tuned "
        "on Nutrition5k. A multi-label head recognizes common food components, while "
        "a regression head directly estimates total calories and protein. No nutrition "
        "API is called during prediction."
    )
    st.markdown(
        "**Limitations:** portion size is difficult to infer from one RGB image, the "
        "training data is biased toward cafeteria meals, and unfamiliar cuisines may "
        "produce low-confidence predictions."
    )

with prediction_tab:
    uploaded = st.file_uploader(
        "Upload a meal image",
        type=["jpg", "jpeg", "png", "webp"],
        help="Use a clear image showing the whole plate.",
    )
    image: Image.Image | None = None
    if uploaded is not None:
        try:
            image = Image.open(uploaded).convert("RGB")
            st.image(image, caption="Uploaded meal", use_container_width=True)
        except Exception:
            st.error("The uploaded file could not be decoded as an image.")

    if st.button("Predict", type="primary", disabled=image is None):
        try:
            with st.spinner("Analyzing the meal..."):
                started = perf_counter()
                result = load_predictor().predict(image)
                elapsed_ms = (perf_counter() - started) * 1_000
        except FileNotFoundError:
            st.error(
                "The trained artifacts are not available yet. Run the preparation, "
                "training, and evaluation commands in README.md."
            )
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
        else:
            calories_low, calories_high = result["calories_interval"]
            protein_low, protein_high = result["protein_interval"]
            col1, col2, col3 = st.columns(3)
            col1.metric("Estimated calories", f"{result['calories']:.0f} kcal")
            col2.metric("Estimated protein", f"{result['protein']:.1f} g")
            col3.metric("Inference time", f"{elapsed_ms:.0f} ms")
            st.caption(
                f"{result['coverage']:.0%} empirical range: "
                f"{calories_low:.0f}-{calories_high:.0f} kcal and "
                f"{protein_low:.1f}-{protein_high:.1f} g"
            )

            st.subheader("Recognized food components")
            for item in result["ingredients"]:
                label = item["name"].replace("_", " ").title()
                confidence = float(item["confidence"])
                st.write(f"{label}: {confidence:.1%}")
                st.progress(confidence)
            if not result["ingredients"][0]["is_confident"]:
                st.warning("Low-confidence recognition: this meal may be outside the training data.")

            st.info(
                "These are research estimates, not medical or dietary advice. "
                "Actual nutrition depends on portion size and preparation method."
            )
