"""
HousePredict AI — AI Powered House Price Prediction
Streamlit web app built on top of the RandomForestRegressor pipeline
trained in train_model.ipynb / train_model.py.

Run with:  streamlit run app.py
"""

import json
import time
from datetime import datetime
from io import BytesIO

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from streamlit_option_menu import option_menu

# ----------------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="HousePredict AI",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# Load artifacts
# ----------------------------------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("house_price_model.pkl")


@st.cache_data
def load_metrics():
    with open("metrics.json") as f:
        return json.load(f)


@st.cache_data
def load_feature_importance():
    with open("feature_importance.json") as f:
        return json.load(f)


@st.cache_data
def load_metadata():
    with open("metadata.json") as f:
        return json.load(f)


@st.cache_data
def load_dataset():
    return pd.read_csv("house_price_dataset.csv")


try:
    pipeline = load_model()
    metrics = load_metrics()
    feature_importance = load_feature_importance()
    metadata = load_metadata()
    dataset = load_dataset()
    ARTIFACTS_OK = True
except FileNotFoundError as e:
    ARTIFACTS_OK = False
    MISSING_FILE = str(e)

# ----------------------------------------------------------------------------
# Session state
# ----------------------------------------------------------------------------
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False
if "history" not in st.session_state:
    st.session_state.history = []  # list of dicts: inputs + prediction + timestamp
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None
if "nav" not in st.session_state:
    st.session_state.nav = "Home"

# ----------------------------------------------------------------------------
# Theme tokens (Royal Blue / Dark Navy / Teal, per brief)
# ----------------------------------------------------------------------------
LIGHT = {
    "bg": "#FFFFFF",
    "bg_soft": "#F5F8FF",
    "text": "#0F172A",
    "text_soft": "#475569",
    "card": "#FFFFFF",
    "border": "rgba(15, 23, 42, 0.08)",
}
DARK = {
    "bg": "#0B1220",
    "bg_soft": "#111A2E",
    "text": "#E5E9F5",
    "text_soft": "#9CA6BF",
    "card": "#141E33",
    "border": "rgba(255, 255, 255, 0.08)",
}
PRIMARY = "#2563EB"
PRIMARY_DARK = "#1D4ED8"
SECONDARY = "#0F172A"
ACCENT = "#14B8A6"

T = DARK if st.session_state.dark_mode else LIGHT

# ----------------------------------------------------------------------------
# Global CSS
# ----------------------------------------------------------------------------
st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;600;700;800&family=Inter:wght@400;500;600&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
    }}
    h1, h2, h3, .headline {{
        font-family: 'Sora', sans-serif;
    }}

    .stApp {{
        background: {T['bg']};
        color: {T['text']};
    }}

    /* ---------- Animations ---------- */
    @keyframes fadeIn {{
        from {{ opacity: 0; }}
        to {{ opacity: 1; }}
    }}
    @keyframes slideUp {{
        from {{ opacity: 0; transform: translateY(24px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}
    @keyframes zoomIn {{
        from {{ opacity: 0; transform: scale(0.92); }}
        to {{ opacity: 1; transform: scale(1); }}
    }}
    @keyframes floaty {{
        0%, 100% {{ transform: translateY(0px) rotate(0deg); }}
        50% {{ transform: translateY(-14px) rotate(6deg); }}
    }}
    @keyframes gradientShift {{
        0% {{ background-position: 0% 50%; }}
        50% {{ background-position: 100% 50%; }}
        100% {{ background-position: 0% 50%; }}
    }}
    @keyframes ripple {{
        0% {{ box-shadow: 0 0 0 0 rgba(37, 99, 235, 0.45); }}
        100% {{ box-shadow: 0 0 0 22px rgba(37, 99, 235, 0); }}
    }}
    @keyframes checkPop {{
        0% {{ transform: scale(0); opacity: 0; }}
        60% {{ transform: scale(1.15); opacity: 1; }}
        100% {{ transform: scale(1); opacity: 1; }}
    }}
    @keyframes countUp {{
        from {{ opacity: 0; transform: translateY(8px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}

    .fade-in {{ animation: fadeIn 0.8s ease both; }}
    .slide-up {{ animation: slideUp 0.7s ease both; }}
    .zoom-in {{ animation: zoomIn 0.6s ease both; }}
    .delay-1 {{ animation-delay: 0.15s; }}
    .delay-2 {{ animation-delay: 0.3s; }}
    .delay-3 {{ animation-delay: 0.45s; }}

    /* ---------- Hero ---------- */
    .hero {{
        position: relative;
        border-radius: 28px;
        padding: 64px 48px;
        overflow: hidden;
        background: linear-gradient(120deg, {SECONDARY} 0%, {PRIMARY_DARK} 55%, {ACCENT} 130%);
        background-size: 200% 200%;
        animation: gradientShift 12s ease infinite;
        color: white;
        margin-bottom: 28px;
    }}
    .hero-eyebrow {{
        display: inline-block;
        background: rgba(255,255,255,0.12);
        border: 1px solid rgba(255,255,255,0.25);
        padding: 6px 16px;
        border-radius: 999px;
        font-size: 0.85rem;
        letter-spacing: 0.02em;
        margin-bottom: 18px;
        backdrop-filter: blur(6px);
    }}
    .hero h1 {{
        font-size: 3rem;
        font-weight: 800;
        line-height: 1.15;
        margin-bottom: 14px;
    }}
    .hero p.sub {{
        font-size: 1.15rem;
        color: rgba(255,255,255,0.88);
        max-width: 640px;
        margin-bottom: 28px;
    }}
    .float-icon {{
        position: absolute;
        font-size: 2.2rem;
        opacity: 0.35;
        animation: floaty 6s ease-in-out infinite;
    }}

    /* ---------- Glass card ---------- */
    .glass-card {{
        background: {"rgba(20, 30, 51, 0.65)" if st.session_state.dark_mode else "rgba(255, 255, 255, 0.65)"};
        backdrop-filter: blur(14px);
        border: 1px solid {T['border']};
        border-radius: 22px;
        padding: 32px;
        box-shadow: 0 8px 30px rgba(15, 23, 42, 0.10);
        transition: transform 0.25s ease, box-shadow 0.25s ease;
    }}
    .glass-card:hover {{
        transform: translateY(-3px);
        box-shadow: 0 14px 40px rgba(15, 23, 42, 0.16);
    }}

    /* ---------- Info / feature cards ---------- */
    .info-card {{
        background: {T['card']};
        border: 1px solid {T['border']};
        border-radius: 18px;
        padding: 22px 20px;
        height: 100%;
        transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
    }}
    .info-card:hover {{
        transform: translateY(-4px);
        box-shadow: 0 10px 26px rgba(37, 99, 235, 0.16);
        border-color: {PRIMARY};
    }}
    .info-card .icon {{
        font-size: 1.8rem;
        margin-bottom: 10px;
    }}
    .info-card h4 {{
        margin: 0 0 6px 0;
        color: {T['text']};
    }}
    .info-card p {{
        color: {T['text_soft']};
        font-size: 0.92rem;
        margin: 0;
    }}

    /* ---------- Result card ---------- */
    .result-card {{
        text-align: center;
        border-radius: 26px;
        padding: 46px 30px;
        background: linear-gradient(135deg, {PRIMARY} 0%, {ACCENT} 100%);
        color: white;
        animation: zoomIn 0.5s ease both;
        box-shadow: 0 18px 40px rgba(37, 99, 235, 0.35);
    }}
    .result-card .label {{
        font-size: 1.05rem;
        opacity: 0.9;
        margin-bottom: 6px;
    }}
    .result-card .amount {{
        font-size: 3rem;
        font-weight: 800;
        animation: countUp 0.6s ease both;
    }}
    .checkmark {{
        display: inline-block;
        font-size: 2.6rem;
        animation: checkPop 0.5s ease both;
    }}

    /* ---------- Buttons ---------- */
    div.stButton > button {{
        background: linear-gradient(135deg, {PRIMARY} 0%, {PRIMARY_DARK} 100%);
        color: white;
        border: none;
        border-radius: 14px;
        padding: 0.7rem 1.6rem;
        font-weight: 600;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }}
    div.stButton > button:hover {{
        transform: scale(1.04);
        box-shadow: 0 0 0 6px rgba(37, 99, 235, 0.18), 0 10px 24px rgba(37, 99, 235, 0.35);
    }}
    div.stButton > button:active {{
        animation: ripple 0.5s ease-out;
    }}

    /* ---------- Section heading ---------- */
    .section-title {{
        font-size: 1.9rem;
        font-weight: 800;
        margin-bottom: 4px;
        color: {T['text']};
    }}
    .section-sub {{
        color: {T['text_soft']};
        margin-bottom: 24px;
    }}

    /* ---------- Footer ---------- */
    .footer {{
        text-align: center;
        padding: 28px 0 12px 0;
        color: {T['text_soft']};
        font-size: 0.9rem;
        border-top: 1px solid {T['border']};
        margin-top: 40px;
    }}

    /* Hide default streamlit chrome for a cleaner SaaS feel */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def inr(amount: float) -> str:
    """Format a number in Indian currency grouping, e.g. 3,23,94,069.72"""
    amount = round(float(amount), 2)
    whole, frac = f"{amount:,.2f}".split(".")
    whole = whole.replace(",", "")
    if len(whole) > 3:
        last3 = whole[-3:]
        rest = whole[:-3]
        parts = []
        while len(rest) > 2:
            parts.insert(0, rest[-2:])
            rest = rest[:-2]
        if rest:
            parts.insert(0, rest)
        whole = ",".join(parts) + "," + last3
    return f"₹ {whole}.{frac}"


def make_pdf_report(inputs: dict, prediction: float) -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, topMargin=30 * mm, bottomMargin=20 * mm)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("title", parent=styles["Title"], textColor=colors.HexColor(SECONDARY))
    story = [
        Paragraph("🏠 HousePredict AI — Prediction Report", title_style),
        Spacer(1, 6 * mm),
        Paragraph(f"Generated on {datetime.now().strftime('%d %b %Y, %H:%M')}", styles["Normal"]),
        Spacer(1, 10 * mm),
        Paragraph("Estimated House Price", styles["Heading2"]),
        Paragraph(inr(prediction), ParagraphStyle("price", parent=styles["Title"], textColor=colors.HexColor(PRIMARY), fontSize=22)),
        Spacer(1, 8 * mm),
        Paragraph("Input Details", styles["Heading2"]),
    ]
    data = [["Field", "Value"]] + [[k.replace("_", " ").title(), str(v)] for k, v in inputs.items()]
    table = Table(data, colWidths=[70 * mm, 80 * mm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(PRIMARY)),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F5F8FF")]),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(table)
    story.append(Spacer(1, 10 * mm))
    story.append(Paragraph("Model: Random Forest Regressor · Made with Python, Streamlit & Scikit-Learn", styles["Normal"]))
    doc.build(story)
    return buf.getvalue()


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🏠 Project Information")
    st.markdown("**Model Used**  \nRandom Forest Regressor")

    if ARTIFACTS_OK:
        st.markdown("**Accuracy Metrics**")
        c1, c2 = st.columns(2)
        c1.metric("R² Score", f"{metrics['r2']:.3f}")
        c2.metric("MAE", inr(metrics["mae"]))
        st.metric("RMSE", inr(metrics["rmse"]))
        st.caption(f"Trained on {metrics['n_rows']:,} records")
    else:
        st.warning(f"Model artifacts not found: {MISSING_FILE}")

    st.divider()

    dm = st.toggle("🌙 Dark Mode", value=st.session_state.dark_mode)
    if dm != st.session_state.dark_mode:
        st.session_state.dark_mode = dm
        st.rerun()

    st.divider()
    st.markdown("**Developer**")
    st.markdown("Vishnu Chaudhari  \nMCA (Data Science)")
    st.link_button("🔗 GitHub", "https://github.com/", use_container_width=True)
    st.link_button("💼 LinkedIn", "https://linkedin.com/", use_container_width=True)

# ----------------------------------------------------------------------------
# Sticky Navbar
# ----------------------------------------------------------------------------
selected = option_menu(
    menu_title=None,
    options=["Home", "Predict", "About", "Model", "Contact"],
    icons=["house", "calculator", "info-circle", "cpu", "envelope"],
    orientation="horizontal",
    default_index=["Home", "Predict", "About", "Model", "Contact"].index(st.session_state.nav),
    styles={
        "container": {"padding": "6px 10px", "background-color": T["card"], "border-radius": "14px",
                       "border": f"1px solid {T['border']}", "box-shadow": "0 4px 16px rgba(15,23,42,0.06)"},
        "icon": {"color": ACCENT, "font-size": "15px"},
        "nav-link": {"font-size": "15px", "font-weight": "600", "color": T["text_soft"],
                      "text-align": "center", "margin": "0 4px", "border-radius": "10px",
                      "padding": "10px 16px", "transition": "all 0.2s ease"},
        "nav-link-selected": {"background-color": PRIMARY, "color": "white"},
    },
)
st.session_state.nav = selected

# ----------------------------------------------------------------------------
# HOME
# ----------------------------------------------------------------------------
if selected == "Home":
    st.markdown(
        f"""
        <div class="hero fade-in">
            <span class="float-icon" style="top:12%; left:8%; animation-delay:0s;">🏘️</span>
            <span class="float-icon" style="top:65%; left:14%; animation-delay:1.2s;">📊</span>
            <span class="float-icon" style="top:20%; right:10%; animation-delay:0.6s;">🔑</span>
            <span class="float-icon" style="top:70%; right:16%; animation-delay:1.8s;">🏢</span>
            <div class="hero-eyebrow slide-up">Machine Learning · Random Forest Regression</div>
            <h1 class="slide-up delay-1">🏠 AI Powered House Price Prediction</h1>
            <p class="sub slide-up delay-2">Predict accurate house prices instantly using Machine Learning
            and Random Forest Regression. Trained on real estate data across major Indian cities.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    hc1, hc2, hc3 = st.columns([1, 1, 3])
    with hc1:
        if st.button("🔮 Predict Price", use_container_width=True):
            st.session_state.nav = "Predict"
            st.rerun()
    with hc2:
        if st.button("📖 Learn More", use_container_width=True):
            st.session_state.nav = "About"
            st.rerun()

    st.write("")
    st.markdown('<div class="section-title slide-up">Why HousePredict AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub slide-up">Fast, transparent, data-driven price estimates</div>', unsafe_allow_html=True)

    feats = [
        ("⚡", "Instant Predictions", "Get an estimated price in seconds, powered by a trained Random Forest model."),
        ("🎯", "High Accuracy", f"Validated with an R² score of {metrics['r2']:.2f} on held-out test data." if ARTIFACTS_OK else "Model metrics load once the pipeline is trained."),
        ("📈", "Data-Backed Insights", "Explore feature importance and market trends behind every prediction."),
        ("🔒", "Transparent Pipeline", "Built with an explainable scikit-learn pipeline, not a black box."),
    ]
    cols = st.columns(4)
    for col, (icon, title, desc) in zip(cols, feats):
        with col:
            st.markdown(
                f"""<div class="info-card zoom-in"><div class="icon">{icon}</div>
                <h4>{title}</h4><p>{desc}</p></div>""",
                unsafe_allow_html=True,
            )

    if ARTIFACTS_OK:
        st.write("")
        st.markdown('<div class="section-title">Market Snapshot</div>', unsafe_allow_html=True)
        pc1, pc2 = st.columns(2)
        with pc1:
            fig = px.histogram(dataset, x="price", nbins=40, color_discrete_sequence=[PRIMARY],
                                title="House Price Distribution")
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                               font_color=T["text"], margin=dict(t=50, b=10))
            st.plotly_chart(fig, use_container_width=True)
        with pc2:
            fig2 = px.scatter(dataset.sample(min(800, len(dataset)), random_state=1), x="area_sqft", y="price",
                               color="location", opacity=0.7, title="Area vs Price")
            fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                font_color=T["text"], margin=dict(t=50, b=10))
            st.plotly_chart(fig2, use_container_width=True)

# ----------------------------------------------------------------------------
# PREDICT
# ----------------------------------------------------------------------------
elif selected == "Predict":
    st.markdown('<div class="section-title fade-in">Predict Your House Price</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub fade-in">Fill in the property details below</div>', unsafe_allow_html=True)

    if not ARTIFACTS_OK:
        st.error(f"Model artifacts missing ({MISSING_FILE}). Run train_model.py first.")
    else:
        st.markdown('<div class="glass-card slide-up">', unsafe_allow_html=True)

        with st.form("predict_form"):
            r = metadata["ranges"]
            cats = metadata["categories"]

            c1, c2, c3 = st.columns(3)
            with c1:
                area_sqft = st.number_input("📐 Area (sq ft)", min_value=100, max_value=20000,
                                             value=r["area_sqft"]["median"], step=10)
                bedrooms = st.number_input("🛏️ Bedrooms", min_value=0, max_value=10,
                                            value=r["bedrooms"]["median"])
                bathrooms = st.number_input("🛁 Bathrooms", min_value=0, max_value=10,
                                             value=r["bathrooms"]["median"])
                balconies = st.number_input("🌤️ Balconies", min_value=0, max_value=10,
                                             value=r["balconies"]["median"])
            with c2:
                age_years = st.number_input("🏗️ Age of Property (years)", min_value=0, max_value=100,
                                             value=r["age_years"]["median"])
                total_floors = st.number_input("🏢 Total Floors", min_value=1, max_value=100,
                                                value=r["total_floors"]["median"])
                floor = st.number_input("🪜 Floor", min_value=0, max_value=int(total_floors),
                                         value=min(r["floor"]["median"], int(total_floors)))
                parking = st.number_input("🚗 Parking", min_value=0, max_value=10,
                                           value=r["parking"]["median"])
            with c3:
                location = st.selectbox("📍 Location", cats["location"])
                furnishing = st.selectbox("🛋️ Furnishing", cats["furnishing"])
                property_type = st.selectbox("🏠 Property Type", cats["property_type"])

            fc1, fc2, fc3 = st.columns([2, 1, 1])
            submitted = fc1.form_submit_button("🔮 Predict Price", use_container_width=True)
            cleared = fc2.form_submit_button("🧹 Clear", use_container_width=True)
            reset = fc3.form_submit_button("↺ Reset", use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

        if cleared or reset:
            st.session_state.last_prediction = None
            st.rerun()

        if submitted:
            inputs = {
                "area_sqft": int(area_sqft), "bedrooms": int(bedrooms), "bathrooms": int(bathrooms),
                "balconies": int(balconies), "age_years": int(age_years), "floor": int(floor),
                "total_floors": int(total_floors), "parking": int(parking), "location": location,
                "furnishing": furnishing, "property_type": property_type,
            }
            progress = st.progress(0, text="Analyzing property details...")
            for pct, msg in [(30, "Encoding features..."), (65, "Running Random Forest model..."), (100, "Finalizing estimate...")]:
                time.sleep(0.35)
                progress.progress(pct, text=msg)
            time.sleep(0.2)
            progress.empty()

            X_new = pd.DataFrame([inputs])
            pred = float(pipeline.predict(X_new)[0])
            st.session_state.last_prediction = {"inputs": inputs, "prediction": pred}
            st.session_state.history.insert(0, {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                **inputs, "predicted_price": round(pred, 2),
            })

        if st.session_state.last_prediction:
            pred = st.session_state.last_prediction["prediction"]
            inputs = st.session_state.last_prediction["inputs"]
            st.markdown(
                f"""
                <div class="result-card">
                    <div class="checkmark">✅</div>
                    <div class="label">Estimated House Price</div>
                    <div class="amount">{inr(pred)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.write("")
            dl1, dl2 = st.columns(2)
            with dl1:
                pdf_bytes = make_pdf_report(inputs, pred)
                st.download_button("⬇️ Download Prediction Report (PDF)", data=pdf_bytes,
                                    file_name="house_price_prediction.pdf", mime="application/pdf",
                                    use_container_width=True)
            with dl2:
                csv_bytes = pd.DataFrame([{**inputs, "predicted_price": pred}]).to_csv(index=False).encode()
                st.download_button("⬇️ Download as CSV", data=csv_bytes,
                                    file_name="house_price_prediction.csv", mime="text/csv",
                                    use_container_width=True)

        if st.session_state.history:
            st.write("")
            st.markdown('<div class="section-title">Prediction History</div>', unsafe_allow_html=True)
            hist_df = pd.DataFrame(st.session_state.history)
            hist_df["predicted_price"] = hist_df["predicted_price"].apply(inr)
            st.dataframe(hist_df, use_container_width=True, hide_index=True)
            if st.button("🗑️ Clear History"):
                st.session_state.history = []
                st.rerun()

# ----------------------------------------------------------------------------
# ABOUT
# ----------------------------------------------------------------------------
elif selected == "About":
    st.markdown('<div class="section-title fade-in">About This Project</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub fade-in">How HousePredict AI works, end to end</div>', unsafe_allow_html=True)

    about_items = [
        ("📊", "Dataset", "Property records including area, rooms, age, floor, parking, "
                           "location, furnishing and property type, paired with sale price."),
        ("⚙️", "ML Pipeline", "A scikit-learn Pipeline chains preprocessing and modeling so "
                               "the exact same transformations are applied at train and predict time."),
        ("🧩", "Feature Engineering", "Numerical fields are standardized; categorical fields "
                                      "(location, furnishing, property type) are one-hot encoded."),
        ("🌳", "Model Training", "A Random Forest Regressor (200 trees) is trained on an 80/20 "
                                  "train-test split with a fixed random seed for reproducibility."),
        ("📐", "Evaluation Metrics", "Performance is measured with MAE, RMSE and R² on held-out "
                                      "test data, shown live in the sidebar and Model tab."),
        ("🚀", "Future Improvements", "Planned: gradient boosting comparison, hyperparameter "
                                       "tuning, live data ingestion, and explainability (SHAP) views."),
    ]
    cols = st.columns(3)
    for i, (icon, title, desc) in enumerate(about_items):
        with cols[i % 3]:
            st.markdown(
                f"""<div class="info-card slide-up"><div class="icon">{icon}</div>
                <h4>{title}</h4><p>{desc}</p></div>""",
                unsafe_allow_html=True,
            )
        if i % 3 == 2:
            st.write("")

# ----------------------------------------------------------------------------
# MODEL
# ----------------------------------------------------------------------------
elif selected == "Model":
    st.markdown('<div class="section-title fade-in">Model Insights</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub fade-in">Random Forest Regressor — performance & behavior</div>', unsafe_allow_html=True)

    if not ARTIFACTS_OK:
        st.error(f"Model artifacts missing ({MISSING_FILE}). Run train_model.py first.")
    else:
        m1, m2, m3 = st.columns(3)
        m1.markdown(f'<div class="info-card"><h4>MAE</h4><p style="font-size:1.4rem;color:{PRIMARY};font-weight:700;">{inr(metrics["mae"])}</p></div>', unsafe_allow_html=True)
        m2.markdown(f'<div class="info-card"><h4>RMSE</h4><p style="font-size:1.4rem;color:{PRIMARY};font-weight:700;">{inr(metrics["rmse"])}</p></div>', unsafe_allow_html=True)
        m3.markdown(f'<div class="info-card"><h4>R² Score</h4><p style="font-size:1.4rem;color:{PRIMARY};font-weight:700;">{metrics["r2"]:.4f}</p></div>', unsafe_allow_html=True)

        st.write("")
        fc1, fc2 = st.columns(2)
        with fc1:
            fi = pd.Series(feature_importance).sort_values(ascending=True)
            fig = go.Figure(go.Bar(x=fi.values, y=fi.index, orientation="h",
                                    marker_color=ACCENT))
            fig.update_layout(title="Feature Importance", paper_bgcolor="rgba(0,0,0,0)",
                               plot_bgcolor="rgba(0,0,0,0)", font_color=T["text"],
                               margin=dict(t=50, b=10))
            st.plotly_chart(fig, use_container_width=True)
        with fc2:
            loc_counts = dataset["location"].value_counts()
            fig2 = px.pie(values=loc_counts.values, names=loc_counts.index, hole=0.45,
                          title="Location Distribution",
                          color_discrete_sequence=px.colors.sequential.Blues_r)
            fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color=T["text"], margin=dict(t=50, b=10))
            st.plotly_chart(fig2, use_container_width=True)

        st.write("")
        st.markdown('<div class="section-title">Predicted vs Actual (sample)</div>', unsafe_allow_html=True)
        sample = dataset.sample(min(500, len(dataset)), random_state=7)
        sample_preds = pipeline.predict(sample.drop(columns=["price"]))
        fig3 = px.scatter(x=sample["price"], y=sample_preds, opacity=0.6,
                           labels={"x": "Actual Price", "y": "Predicted Price"},
                           color_discrete_sequence=[PRIMARY])
        max_v = float(max(sample["price"].max(), sample_preds.max()))
        fig3.add_shape(type="line", x0=0, y0=0, x1=max_v, y1=max_v,
                        line=dict(color=ACCENT, dash="dash"))
        fig3.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                            font_color=T["text"], margin=dict(t=20, b=10))
        st.plotly_chart(fig3, use_container_width=True)

# ----------------------------------------------------------------------------
# CONTACT
# ----------------------------------------------------------------------------
elif selected == "Contact":
    st.markdown('<div class="section-title fade-in">Get In Touch</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub fade-in">Questions, feedback, or collaboration ideas</div>', unsafe_allow_html=True)

    cc1, cc2 = st.columns([2, 1])
    with cc1:
        st.markdown('<div class="glass-card slide-up">', unsafe_allow_html=True)
        with st.form("contact_form"):
            name = st.text_input("Your Name")
            email = st.text_input("Your Email")
            message = st.text_area("Message", height=140)
            sent = st.form_submit_button("✉️ Send Message", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        if sent:
            if not name or not email or not message:
                st.warning("Please fill in all fields.")
            else:
                st.success("Thanks! Your message has been noted. (Demo form — not wired to a backend yet.)")
    with cc2:
        st.markdown(
            f"""<div class="info-card"><div class="icon">👤</div>
            <h4>Vishnu Chaudhari</h4><p>MCA (Data Science)</p>
            <p style="margin-top:10px;">🔗 GitHub: github.com<br>💼 LinkedIn: linkedin.com</p>
            </div>""",
            unsafe_allow_html=True,
        )

# ----------------------------------------------------------------------------
# Footer
# ----------------------------------------------------------------------------
st.markdown(
    f"""
    <div class="footer">
        © 2026 Vishnu Chaudhari · Made with ❤️ using Python, Streamlit and Scikit-Learn
    </div>
    """,
    unsafe_allow_html=True,
)
