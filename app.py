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
    page_icon="🥗",
    layout="wide",
)


@st.cache_resource
def load_predictor() -> SmartPlatePredictor:
    return SmartPlatePredictor(ARTIFACTS_DIR)


st.markdown(
    """
    <style>
    :root {
        --navy-950: #0b1b24;
        --navy-900: #102832;
        --navy-800: #173944;
        --teal-500: #55d6bd;
        --teal-400: #73e3cd;
        --teal-100: #dff8f2;
        --ink: #13252d;
        --muted: #64777f;
        --surface: #ffffff;
        --canvas: #f2f7f6;
        --border: #dce8e5;
    }

    html, body, [class*="css"] {
        font-family: Inter, "Segoe UI", Tahoma, Arial, sans-serif;
    }

    .stApp {
        background:
            radial-gradient(circle at 10% 0%, rgba(85, 214, 189, 0.10), transparent 26rem),
            linear-gradient(180deg, #f8fbfa 0%, var(--canvas) 100%);
        color: var(--ink);
    }

    [data-testid="stHeader"] {
        background: rgba(248, 251, 250, 0.86);
        backdrop-filter: blur(12px);
    }

    [data-testid="stAppViewBlockContainer"] {
        max-width: 1180px;
        padding-top: 1.2rem;
        padding-bottom: 4rem;
    }

    #MainMenu, footer {
        visibility: hidden;
    }

    .smart-nav {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin: 0 0 1rem;
        padding: 0.2rem 0.15rem;
    }

    .brand {
        display: flex;
        align-items: center;
        gap: 0.7rem;
        color: var(--navy-950);
        font-size: 1.35rem;
        font-weight: 800;
        letter-spacing: -0.03em;
    }

    .brand-mark {
        display: grid;
        place-items: center;
        width: 2.35rem;
        height: 2.35rem;
        border-radius: 0.75rem;
        color: var(--navy-950);
        background: var(--teal-500);
        box-shadow: 0 8px 24px rgba(36, 170, 144, 0.24);
        font-size: 1rem;
        font-weight: 900;
    }

    .nav-note {
        color: var(--muted);
        font-size: 0.88rem;
        font-weight: 600;
    }

    .hero {
        position: relative;
        isolation: isolate;
        overflow: hidden;
        display: grid;
        grid-template-columns: minmax(0, 1.35fr) minmax(270px, 0.65fr);
        gap: 3rem;
        align-items: center;
        min-height: 430px;
        padding: 3.5rem;
        border: 1px solid rgba(130, 228, 208, 0.16);
        border-radius: 2.25rem;
        background: linear-gradient(135deg, var(--navy-950), #173b45 72%, #195149);
        box-shadow: 0 30px 70px rgba(13, 44, 54, 0.20);
        color: #ffffff;
    }

    .hero::before {
        content: "";
        position: absolute;
        z-index: -1;
        width: 30rem;
        height: 30rem;
        right: -10rem;
        bottom: -19rem;
        border-radius: 50%;
        border: 4rem solid rgba(85, 214, 189, 0.07);
    }

    .hero::after {
        content: "";
        position: absolute;
        z-index: -1;
        width: 14rem;
        height: 14rem;
        left: 44%;
        top: -8rem;
        border-radius: 50%;
        background: rgba(85, 214, 189, 0.08);
        filter: blur(1px);
    }

    .eyebrow {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        margin-bottom: 1.25rem;
        padding: 0.48rem 0.8rem;
        border: 1px solid rgba(115, 227, 205, 0.35);
        border-radius: 999px;
        color: var(--teal-400);
        background: rgba(85, 214, 189, 0.08);
        font-size: 0.72rem;
        font-weight: 800;
        letter-spacing: 0.12em;
    }

    .eyebrow-dot {
        width: 0.48rem;
        height: 0.48rem;
        border-radius: 50%;
        background: var(--teal-500);
        box-shadow: 0 0 0 0.3rem rgba(85, 214, 189, 0.12);
    }

    .hero h1 {
        max-width: 700px;
        margin: 0;
        color: #ffffff;
        font-size: clamp(2.6rem, 5vw, 4.5rem);
        line-height: 1.02;
        letter-spacing: -0.055em;
        font-weight: 850;
    }

    .hero h1 span {
        color: var(--teal-500);
    }

    .hero-copy > p {
        max-width: 640px;
        margin: 1.35rem 0 1.6rem;
        color: #c9dadd;
        font-size: 1.08rem;
        line-height: 1.75;
    }

    .feature-pills {
        display: flex;
        flex-wrap: wrap;
        gap: 0.65rem;
    }

    .feature-pill {
        padding: 0.58rem 0.85rem;
        border: 1px solid rgba(207, 241, 235, 0.18);
        border-radius: 999px;
        color: #e9f6f3;
        background: rgba(255, 255, 255, 0.06);
        font-size: 0.78rem;
        font-weight: 700;
    }

    .analysis-preview {
        padding: 1.3rem;
        border: 1px solid rgba(207, 241, 235, 0.22);
        border-radius: 1.6rem;
        background: rgba(255, 255, 255, 0.09);
        box-shadow: inset 0 1px rgba(255, 255, 255, 0.12), 0 22px 50px rgba(0, 0, 0, 0.14);
        backdrop-filter: blur(12px);
    }

    .preview-top {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 1.1rem;
        color: #ffffff;
        font-weight: 800;
    }

    .ready {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        color: var(--teal-400);
        font-size: 0.72rem;
    }

    .preview-row {
        display: grid;
        grid-template-columns: 2.3rem 1fr auto;
        gap: 0.8rem;
        align-items: center;
        padding: 0.9rem 0;
        border-top: 1px solid rgba(223, 248, 242, 0.12);
    }

    .preview-icon {
        display: grid;
        place-items: center;
        width: 2.2rem;
        height: 2.2rem;
        border-radius: 0.75rem;
        color: var(--teal-400);
        background: rgba(85, 214, 189, 0.11);
        font-weight: 900;
    }

    .preview-label {
        color: #ffffff;
        font-size: 0.88rem;
        font-weight: 750;
    }

    .preview-sub {
        margin-top: 0.1rem;
        color: #9eb6bb;
        font-size: 0.72rem;
    }

    .preview-value {
        color: #dff8f2;
        font-size: 0.75rem;
        font-weight: 700;
    }

    .section-intro {
        margin: 2.2rem 0 1rem;
    }

    .section-kicker {
        margin-bottom: 0.35rem;
        color: #16997f;
        font-size: 0.76rem;
        font-weight: 850;
        letter-spacing: 0.12em;
        text-transform: uppercase;
    }

    .section-intro h2, .result-heading h2 {
        margin: 0;
        color: var(--navy-950);
        font-size: 2rem;
        letter-spacing: -0.035em;
    }

    .section-intro p, .result-heading p {
        margin: 0.5rem 0 0;
        color: var(--muted);
        line-height: 1.65;
    }

    [data-baseweb="tab-list"] {
        gap: 0.35rem;
        margin-top: 1.2rem;
        padding: 0.35rem;
        border-radius: 999px;
        background: #e5efed;
        width: fit-content;
    }

    [data-baseweb="tab"] {
        min-width: 10.5rem;
        padding: 0.65rem 1rem;
        border-radius: 999px;
        color: #50666e;
        font-weight: 750;
    }

    [aria-selected="true"][data-baseweb="tab"] {
        color: #ffffff;
        background: var(--navy-900);
    }

    [data-baseweb="tab-highlight"] {
        display: none;
    }

    [data-testid="stFileUploader"] {
        margin-top: 0.8rem;
    }

    [data-testid="stFileUploader"] section {
        min-height: 180px;
        padding: 1.4rem;
        border: 1.5px dashed #8bcabb;
        border-radius: 1.4rem;
        background: rgba(255, 255, 255, 0.80);
        transition: border-color 160ms ease, transform 160ms ease, box-shadow 160ms ease;
    }

    [data-testid="stFileUploader"] section:hover {
        border-color: #31b99e;
        transform: translateY(-1px);
        box-shadow: 0 18px 36px rgba(29, 92, 84, 0.09);
    }

    [data-testid="stFileUploader"] button {
        border: 0;
        border-radius: 999px;
        color: #ffffff;
        background: var(--navy-900);
    }

    [data-testid="stImage"] img {
        border-radius: 1.3rem;
        box-shadow: 0 20px 45px rgba(13, 44, 54, 0.14);
    }

    div.stButton > button {
        min-height: 3.25rem;
        width: 100%;
        border: 0;
        border-radius: 1rem;
        color: var(--navy-950);
        background: linear-gradient(135deg, var(--teal-500), #83e8d3);
        box-shadow: 0 14px 30px rgba(40, 177, 151, 0.25);
        font-size: 1rem;
        font-weight: 850;
    }

    div.stButton > button:hover {
        color: var(--navy-950);
        border: 0;
        background: linear-gradient(135deg, #6ce1ca, #9af0df);
        transform: translateY(-1px);
    }

    div.stButton > button:disabled {
        color: #819399;
        background: #dce7e4;
        box-shadow: none;
    }

    [data-testid="stMetric"] {
        min-height: 128px;
        padding: 1.2rem 1.3rem;
        border: 1px solid var(--border);
        border-radius: 1.25rem;
        background: rgba(255, 255, 255, 0.94);
        box-shadow: 0 16px 38px rgba(19, 66, 73, 0.08);
    }

    [data-testid="stMetricLabel"] {
        color: var(--muted);
        font-weight: 700;
    }

    [data-testid="stMetricValue"] {
        color: var(--navy-950);
        font-weight: 850;
    }

    .result-heading {
        margin: 2.2rem 0 1rem;
        padding-top: 1.2rem;
        border-top: 1px solid var(--border);
    }

    [data-testid="stProgress"] > div > div {
        background: linear-gradient(90deg, #2fc0a2, var(--teal-400));
    }

    [data-testid="stAlert"] {
        border-radius: 1rem;
    }

    .about-card {
        min-height: 180px;
        padding: 1.35rem;
        border: 1px solid var(--border);
        border-radius: 1.3rem;
        background: rgba(255, 255, 255, 0.92);
        box-shadow: 0 14px 34px rgba(19, 66, 73, 0.06);
    }

    .about-number {
        display: grid;
        place-items: center;
        width: 2.25rem;
        height: 2.25rem;
        margin-bottom: 1rem;
        border-radius: 0.75rem;
        color: var(--navy-950);
        background: var(--teal-100);
        font-weight: 900;
    }

    .about-card h3 {
        margin: 0 0 0.45rem;
        color: var(--navy-950);
        font-size: 1.05rem;
    }

    .about-card p {
        margin: 0;
        color: var(--muted);
        font-size: 0.9rem;
        line-height: 1.65;
    }

    @media (max-width: 800px) {
        [data-testid="stAppViewBlockContainer"] {
            padding-left: 1rem;
            padding-right: 1rem;
        }

        .nav-note {
            display: none;
        }

        .hero {
            grid-template-columns: 1fr;
            gap: 2rem;
            min-height: auto;
            padding: 2rem 1.35rem;
            border-radius: 1.6rem;
        }

        .hero h1 {
            font-size: 2.75rem;
        }

        .analysis-preview {
            max-width: none;
        }

        [data-baseweb="tab"] {
            min-width: auto;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="smart-nav">
        <div class="brand"><span class="brand-mark">S</span> SmartPlate AI</div>
        <div class="nav-note">EfficientNetV2B0 · Nutrition5k · Local inference</div>
    </div>

    <section class="hero">
        <div class="hero-copy">
            <div class="eyebrow"><span class="eyebrow-dot"></span> AI-POWERED FOOD ANALYSIS</div>
            <h1>Understand your meal.<br><span>From one image.</span></h1>
            <p>
                Upload a clear meal photo to recognize common food components and estimate
                total calories and protein with a fine-tuned deep learning model.
            </p>
            <div class="feature-pills">
                <span class="feature-pill">50 food components</span>
                <span class="feature-pill">Calories + protein</span>
                <span class="feature-pill">95% empirical range</span>
            </div>
        </div>
        <div class="analysis-preview">
            <div class="preview-top">
                <span>Your analysis</span>
                <span class="ready"><span class="eyebrow-dot"></span> Ready</span>
            </div>
            <div class="preview-row">
                <div class="preview-icon">C</div>
                <div><div class="preview-label">Calories</div><div class="preview-sub">Direct image estimate</div></div>
                <div class="preview-value">kcal</div>
            </div>
            <div class="preview-row">
                <div class="preview-icon">P</div>
                <div><div class="preview-label">Protein</div><div class="preview-sub">Estimated amount</div></div>
                <div class="preview-value">grams</div>
            </div>
            <div class="preview-row">
                <div class="preview-icon">50</div>
                <div><div class="preview-label">Food components</div><div class="preview-sub">Multi-label recognition</div></div>
                <div class="preview-value">confidence</div>
            </div>
        </div>
    </section>
    """,
    unsafe_allow_html=True,
)

prediction_tab, about_tab = st.tabs(["Analyze a meal", "About the project"])

with prediction_tab:
    st.markdown(
        """
        <div class="section-intro">
            <div class="section-kicker">Start an analysis</div>
            <h2>Upload your meal photo</h2>
            <p>For the best result, use a well-lit image that shows the complete plate from above.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader(
        "Meal image",
        type=["jpg", "jpeg", "png", "webp"],
        help="Supported formats: JPG, JPEG, PNG, and WEBP.",
        label_visibility="collapsed",
    )
    image: Image.Image | None = None
    if uploaded is not None:
        try:
            image = Image.open(uploaded).convert("RGB")
            st.image(image, caption="Uploaded meal", use_container_width=True)
        except Exception:
            st.error("The uploaded file could not be decoded as an image.")

    if st.button("Analyze my meal", type="primary", disabled=image is None):
        try:
            with st.spinner("SmartPlate is analyzing your meal..."):
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

            st.markdown(
                """
                <div class="result-heading">
                    <div class="section-kicker">Analysis complete</div>
                    <h2>Your nutrition estimate</h2>
                    <p>Model estimates based only on the uploaded RGB image.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            col1, col2, col3 = st.columns(3)
            col1.metric("Estimated calories", f"{result['calories']:.0f} kcal")
            col2.metric("Estimated protein", f"{result['protein']:.1f} g")
            col3.metric("Inference time", f"{elapsed_ms:.0f} ms")
            st.caption(
                f"{result['coverage']:.0%} empirical range: "
                f"{calories_low:.0f}–{calories_high:.0f} kcal and "
                f"{protein_low:.1f}–{protein_high:.1f} g"
            )

            st.subheader("Recognized food components")
            for item in result["ingredients"]:
                label = item["name"].replace("_", " ").title()
                confidence = float(item["confidence"])
                st.write(f"**{label}** · {confidence:.1%}")
                st.progress(confidence)
            if not result["ingredients"][0]["is_confident"]:
                st.warning(
                    "Low-confidence recognition: this meal may be outside the training data."
                )

            st.info(
                "These are research estimates, not medical or dietary advice. "
                "Actual nutrition depends on portion size and preparation method."
            )

with about_tab:
    st.markdown(
        """
        <div class="section-intro">
            <div class="section-kicker">Behind SmartPlate</div>
            <h2>Deep learning is the main component</h2>
            <p>The prediction is produced locally by a trained model, without calling a nutrition API.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    about_col1, about_col2, about_col3 = st.columns(3)
    with about_col1:
        st.markdown(
            """
            <div class="about-card">
                <div class="about-number">01</div>
                <h3>Pre-trained backbone</h3>
                <p>EfficientNetV2B0 starts with official ImageNet weights and is adapted through transfer learning.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with about_col2:
        st.markdown(
            """
            <div class="about-card">
                <div class="about-number">02</div>
                <h3>Multi-task prediction</h3>
                <p>One head recognizes frequent components; a second head estimates calories and protein.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with about_col3:
        st.markdown(
            """
            <div class="about-card">
                <div class="about-number">03</div>
                <h3>Public research data</h3>
                <p>The model was fine-tuned and evaluated on the official Nutrition5k overhead RGB subset.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.warning(
        "Limitations: portion size is difficult to infer from one RGB image, "
        "Nutrition5k is biased toward cafeteria meals, and unfamiliar cuisines may "
        "produce low-confidence predictions."
    )
