import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from pathlib import Path

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="CreditWise | Loan Approval Predictor",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS — FRONTEND ONLY
# ============================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    * {
        font-family: 'Inter', sans-serif;
    }

    #MainMenu, footer {
        visibility: hidden;
    }

    .stApp {
        background: #f6f8fb;
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    [data-testid="stSidebar"] {
        background: #101828;
        border-right: 1px solid #1d2939;
    }

    [data-testid="stSidebar"] * {
        color: #f2f4f7;
    }

    .brand {
        padding: 8px 4px 26px 4px;
        border-bottom: 1px solid #344054;
        margin-bottom: 24px;
    }

    .brand-name {
        font-size: 25px;
        font-weight: 800;
        letter-spacing: -0.8px;
        color: #ffffff;
    }

    .brand-subtitle {
        margin-top: 5px;
        font-size: 9px;
        font-weight: 700;
        letter-spacing: 1.5px;
        color: #98a2b3;
    }

    .side-section {
        font-size: 10px;
        font-weight: 700;
        letter-spacing: 1px;
        color: #667085;
        margin: 20px 4px 9px;
    }

    .nav-item {
        padding: 11px 12px;
        margin: 4px 0;
        border-radius: 8px;
        font-size: 13px;
        font-weight: 500;
        color: #d0d5dd;
    }

    .nav-item.active {
        background: #1d2939;
        color: #ffffff;
        font-weight: 600;
    }

    .sidebar-bottom {
        position: fixed;
        bottom: 24px;
        left: 24px;
        width: 210px;
        border-top: 1px solid #344054;
        padding-top: 15px;
    }

    .sidebar-version {
        font-size: 11px;
        color: #98a2b3;
    }

    .sidebar-product {
        font-size: 10px;
        color: #667085;
        margin-top: 4px;
    }

    .topbar {
        display: flex;
        align-items: flex-start;
        justify-content: space-between;
        margin-bottom: 28px;
    }

    .eyebrow {
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 1.4px;
        color: #667085;
        margin-bottom: 7px;
        text-transform: uppercase;
    }

    .page-title {
        font-size: 31px;
        line-height: 1.15;
        font-weight: 800;
        letter-spacing: -1px;
        color: #101828;
        margin: 0;
    }

    .page-subtitle {
        color: #667085;
        font-size: 14px;
        margin-top: 8px;
        line-height: 1.6;
    }

    .online-badge {
        display: inline-flex;
        align-items: center;
        gap: 7px;
        background: #ecfdf3;
        color: #027a48;
        border: 1px solid #abefc6;
        padding: 7px 11px;
        border-radius: 999px;
        font-size: 11px;
        font-weight: 700;
        white-space: nowrap;
    }

    .hero {
        background: #101828;
        border-radius: 16px;
        padding: 32px;
        margin-bottom: 22px;
        border: 1px solid #1d2939;
    }

    .hero-kicker {
        color: #98a2b3;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 1.3px;
        text-transform: uppercase;
    }

    .hero-title {
        color: #ffffff;
        font-size: 29px;
        font-weight: 800;
        letter-spacing: -0.8px;
        margin-top: 9px;
    }

    .hero-text {
        color: #d0d5dd;
        font-size: 14px;
        line-height: 1.65;
        max-width: 760px;
        margin-top: 8px;
    }

    .feature-card {
        background: #ffffff;
        border: 1px solid #eaecf0;
        border-radius: 13px;
        padding: 22px;
        min-height: 142px;
        box-shadow: 0 2px 7px rgba(16, 24, 40, 0.035);
    }

    .feature-number {
        color: #98a2b3;
        font-size: 10px;
        font-weight: 800;
        letter-spacing: 1px;
    }

    .feature-title {
        color: #101828;
        font-size: 15px;
        font-weight: 700;
        margin-top: 10px;
    }

    .feature-text {
        color: #667085;
        font-size: 12px;
        line-height: 1.55;
        margin-top: 7px;
    }

    .section-card {
        background: #ffffff;
        border: 1px solid #eaecf0;
        border-radius: 14px;
        padding: 24px;
        margin-bottom: 18px;
        box-shadow: 0 2px 7px rgba(16, 24, 40, 0.035);
    }

    .section-header {
        display: flex;
        align-items: center;
        gap: 13px;
        margin-bottom: 20px;
    }

    .section-number {
        width: 31px;
        height: 31px;
        border-radius: 8px;
        background: #f2f4f7;
        color: #344054;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 11px;
        font-weight: 800;
    }

    .section-title {
        font-size: 16px;
        font-weight: 700;
        color: #101828;
    }

    .section-desc {
        color: #667085;
        font-size: 11px;
        margin-top: 2px;
    }

    .progress-wrap {
        background: #ffffff;
        border: 1px solid #eaecf0;
        border-radius: 12px;
        padding: 15px 18px;
        margin-bottom: 18px;
    }

    .progress-label {
        color: #475467;
        font-size: 11px;
        font-weight: 600;
        margin-bottom: 9px;
    }

    .progress-steps {
        display: flex;
        justify-content: space-between;
        gap: 6px;
    }

    .progress-step {
        flex: 1;
        text-align: center;
        padding: 7px 4px;
        border-radius: 7px;
        background: #f2f4f7;
        color: #667085;
        font-size: 10px;
        font-weight: 600;
    }

    .cta-note {
        color: #667085;
        font-size: 11px;
        margin-top: 8px;
    }

    div.stButton > button {
        width: 100%;
        border-radius: 9px;
        height: 48px;
        border: 1px solid #101828;
        background: #101828;
        color: #ffffff;
        font-size: 13px;
        font-weight: 700;
        transition: all 0.15s ease;
    }

    div.stButton > button:hover {
        background: #1d2939;
        border-color: #1d2939;
    }

    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #eaecf0;
        border-radius: 11px;
        padding: 15px;
    }

    .result-approved {
        background: #ecfdf3;
        border: 1px solid #abefc6;
        border-radius: 15px;
        padding: 26px;
        margin-bottom: 18px;
    }

    .result-rejected {
        background: #fff4f2;
        border: 1px solid #fecdca;
        border-radius: 15px;
        padding: 26px;
        margin-bottom: 18px;
    }

    .result-label {
        font-size: 10px;
        font-weight: 800;
        letter-spacing: 1.4px;
        color: #667085;
    }

    .result-status {
        font-size: 30px;
        font-weight: 800;
        letter-spacing: -0.8px;
        color: #101828;
        margin-top: 6px;
    }

    .result-description {
        color: #475467;
        font-size: 13px;
        line-height: 1.6;
        margin-top: 7px;
    }

    .info-card {
        background: #ffffff;
        border: 1px solid #eaecf0;
        border-radius: 12px;
        padding: 18px;
        min-height: 90px;
    }

    .info-label {
        color: #667085;
        font-size: 10px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .info-value {
        color: #101828;
        font-size: 18px;
        font-weight: 700;
        margin-top: 8px;
    }

    .info-small {
        color: #667085;
        font-size: 11px;
        margin-top: 3px;
    }

    .risk-box {
        border: 1px solid #eaecf0;
        background: #ffffff;
        border-radius: 13px;
        padding: 22px;
    }

    .risk-title {
        font-size: 15px;
        font-weight: 700;
        color: #101828;
    }

    .risk-caption {
        color: #667085;
        font-size: 11px;
        line-height: 1.5;
        margin-top: 4px;
    }

    .risk-pill {
        display: inline-block;
        padding: 6px 10px;
        border-radius: 999px;
        font-size: 10px;
        font-weight: 800;
        letter-spacing: 0.5px;
        background: #f2f4f7;
        color: #344054;
        margin-top: 12px;
    }

    .disclaimer {
        background: #f9fafb;
        border: 1px solid #eaecf0;
        border-radius: 11px;
        padding: 15px 17px;
        color: #667085;
        font-size: 11px;
        line-height: 1.55;
        margin-top: 20px;
    }

    .footer {
        border-top: 1px solid #eaecf0;
        margin-top: 42px;
        padding: 24px 0 10px;
        text-align: center;
        color: #98a2b3;
        font-size: 10px;
        line-height: 1.7;
    }

    .footer strong {
        color: #475467;
        font-size: 12px;
    }

    label {
        font-size: 12px !important;
        font-weight: 600 !important;
        color: #344054 !important;
    }

    .stTextInput input,
    .stNumberInput input,
    .stSelectbox div[data-baseweb="select"] > div {
        border-radius: 8px;
    }

    .stExpander {
        border: 1px solid #eaecf0 !important;
        border-radius: 11px !important;
    }

    @media (max-width: 800px) {
        .page-title {
            font-size: 25px;
        }
        .hero {
            padding: 24px;
        }
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# MODEL FILES — PRESERVED EXACTLY
# ============================================================
BASE_DIR = Path(__file__).resolve().parent

MODEL_FILE = BASE_DIR / "creditwise_naive_bayes_v1.pkl"
SCALER_FILE = BASE_DIR / "creditwise_scaler_v1.pkl"
ENCODER_FILE = BASE_DIR / "creditwise_encoder_v1.pkl"
EDUCATION_ENCODER_FILE = (
    BASE_DIR / "creditwise_education_encoder_v1.pkl"
)
TARGET_ENCODER_FILE = (
    BASE_DIR / "creditwise_target_encoder_v1.pkl"
)
FEATURE_FILE = (
    BASE_DIR / "creditwise_feature_columns_v1.pkl"
)

# ============================================================
# MODEL LOADING — PRESERVED
# ============================================================
@st.cache_resource
def load_model():
    required_files = [
        MODEL_FILE,
        SCALER_FILE,
        ENCODER_FILE,
        EDUCATION_ENCODER_FILE,
        TARGET_ENCODER_FILE,
        FEATURE_FILE
    ]

    missing = [
        file.name
        for file in required_files
        if not file.exists()
    ]

    if missing:
        return None, missing

    model = joblib.load(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)
    encoder = joblib.load(ENCODER_FILE)
    education_encoder = joblib.load(EDUCATION_ENCODER_FILE)
    target_encoder = joblib.load(TARGET_ENCODER_FILE)
    feature_columns = joblib.load(FEATURE_FILE)

    return {
        "model": model,
        "scaler": scaler,
        "encoder": encoder,
        "education_encoder": education_encoder,
        "target_encoder": target_encoder,
        "feature_columns": feature_columns
    }, []

ml, missing_files = load_model()

if ml is None:
    st.error("CreditWise model files are missing.")
    st.write("Make sure the following files are in the same folder as app.py:")
    for file in missing_files:
        st.code(file)
    st.stop()

# ============================================================
# SIDEBAR NAVIGATION
# ============================================================
with st.sidebar:
    st.markdown("""
    <div class="brand">
        <div class="brand-name">CreditWise</div>
        <div class="brand-subtitle">LOAN INTELLIGENCE PLATFORM</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="side-section">NAVIGATION</div>', unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        ["Overview", "Loan Assessment", "Prediction Insights", "About"],
        label_visibility="collapsed"
    )

    st.markdown("""
    <div class="sidebar-bottom">
        <div class="sidebar-version">CreditWise v1.0</div>
        <div class="sidebar-product">ML-Powered Loan Assessment</div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# PREDICTION INSIGHTS PAGE
# ============================================================
if page == "Prediction Insights":
    st.markdown("""
    <div class="eyebrow">Prediction Insights</div>
    <div class="page-title">Latest Model Assessment</div>
    <div class="page-subtitle">
        Review the most recent prediction generated during this session.
    </div>
    """, unsafe_allow_html=True)

    if "prediction_result" not in st.session_state:
        st.markdown("""
        <div class="section-card">
            <div class="section-title">No prediction available</div>
            <div class="section-desc">
                Complete a loan assessment first. Your latest model result
                will appear here during the current session.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        result = st.session_state["prediction_result"]

        status = result["status"]
        approved = result["approved"]
        result_class = "result-approved" if approved else "result-rejected"

        st.markdown(
            f"""
            <div class="{result_class}">
                <div class="result-label">LATEST LOAN ASSESSMENT</div>
                <div class="result-status">{status}</div>
                <div class="result-description">
                    {result["description"]}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Approval Probability", f'{result["approval_probability"]:.1%}')
        with m2:
            st.metric("Risk Level", result["risk_level"])
        with m3:
            st.metric("Model Confidence", f'{result["model_confidence"]:.1%}')

        left, right = st.columns([1.15, 0.85])

        with left:
            st.markdown("""
            <div class="section-card">
                <div class="section-title">Probability Breakdown</div>
                <div class="section-desc">
                    Model probability distribution for the latest application.
                </div>
            """, unsafe_allow_html=True)

            fig = go.Figure()
            fig.add_trace(
                go.Bar(
                    x=["Approval", "Rejection"],
                    y=[
                        result["approval_probability"] * 100,
                        result["rejection_probability"] * 100
                    ],
                    text=[
                        f'{result["approval_probability"]:.1%}',
                        f'{result["rejection_probability"]:.1%}'
                    ],
                    textposition="auto",
                    marker_line_width=0,
                    hovertemplate="%{x}: %{y:.1f}%<extra></extra>"
                )
            )
            fig.update_layout(
                height=330,
                margin=dict(l=10, r=10, t=20, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter, sans-serif", color="#475467"),
                yaxis=dict(
                    range=[0, 100],
                    title="Probability (%)",
                    gridcolor="#eaecf0",
                    zeroline=False
                ),
                xaxis=dict(title="", showgrid=False),
                showlegend=False
            )
            st.plotly_chart(
                fig,
                use_container_width=True,
                config={"displayModeBar": False}
            )
            st.markdown("</div>", unsafe_allow_html=True)

        with right:
            st.markdown("""
            <div class="risk-box">
                <div class="risk-title">Risk Assessment</div>
                <div class="risk-caption">
                    Model-derived risk indicator based on the existing
                    approval-probability thresholds.
                </div>
            """, unsafe_allow_html=True)

            approval = result["approval_probability"]
            st.markdown(
                f"""
                <div class="risk-pill">{result["risk_level"].upper()}</div>
                <div style="margin-top:18px;">
                    <div style="font-size:11px;color:#667085;margin-bottom:7px;">
                        Approval probability
                    </div>
                    <div style="
                        width:100%;height:9px;background:#eaecf0;
                        border-radius:999px;overflow:hidden;
                    ">
                        <div style="
                            width:{approval * 100:.2f}%;
                            height:100%;background:#101828;
                            border-radius:999px;
                        "></div>
                    </div>
                    <div style="
                        display:flex;justify-content:space-between;
                        margin-top:7px;font-size:10px;color:#98a2b3;
                    ">
                        <span>0%</span><span>50%</span><span>100%</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown("""
                <div style="
                    margin-top:18px;padding-top:15px;
                    border-top:1px solid #eaecf0;
                    font-size:11px;line-height:1.55;color:#667085;
                ">
                    This is a model-derived indicator and is not a
                    guaranteed real-world lending risk rating.
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
        st.markdown("### Application Summary")

        summary = result["summary"]
        s1, s2, s3, s4 = st.columns(4)
        summary_items = [
            (s1, "Applicant Age", f'{summary["age"]} years'),
            (s2, "Employment Status", summary["employment_status"]),
            (s3, "Credit Score", str(summary["credit_score"])),
            (s4, "Applicant Income", f'{summary["applicant_income"]:,.0f}'),
            (s1, "Loan Amount", f'{summary["loan_amount"]:,.0f}'),
            (s2, "Loan Term", str(summary["loan_term"])),
            (s3, "Loan Purpose", summary["loan_purpose"]),
            (s4, "Property Area", summary["property_area"])
        ]

        for col, label, value in summary_items:
            with col:
                st.markdown(
                    f"""
                    <div class="info-card" style="margin-bottom:10px;">
                        <div class="info-label">{label}</div>
                        <div class="info-value">{value}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        st.markdown("""
        <div class="disclaimer">
            <strong>Important:</strong> CreditWise is an educational machine
            learning application. Predictions are generated by a trained model
            and should not be treated as real-world financial or lending decisions.
        </div>
        """, unsafe_allow_html=True)

    st.stop()

# ============================================================
# ABOUT PAGE
# ============================================================
if page == "About":
    st.markdown("""
    <div class="eyebrow">About CreditWise</div>
    <div class="page-title">Loan Intelligence Platform</div>
    <div class="page-subtitle">
        An educational machine learning application for demonstrating
        an end-to-end loan assessment workflow.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="section-card">
        <div class="section-title">CreditWise — Loan Approval Predictor</div>
        <div class="section-desc" style="margin-top:8px;line-height:1.7;">
            CreditWise accepts applicant, financial, employment, loan and
            collateral information and sends those values through the trained
            preprocessing and prediction pipeline.
        </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    about_cards = [
        ("Machine Learning", "Gaussian Naive Bayes model with the project's saved preprocessing components."),
        ("Technology", "Python, Pandas, NumPy, Scikit-learn, Joblib, Streamlit and Plotly."),
        ("Purpose", "Educational demonstration of a machine learning-powered loan assessment workflow.")
    ]

    for col, (title, text) in zip([c1, c2, c3], about_cards):
        with col:
            st.markdown(
                f"""
                <div class="feature-card">
                    <div class="feature-title">{title}</div>
                    <div class="feature-text">{text}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown("""
    <div class="disclaimer">
        <strong>Important:</strong> CreditWise is an educational machine
        learning application. Predictions are generated by a trained model
        and should not be treated as real-world financial or lending decisions.
    </div>
    """, unsafe_allow_html=True)

    st.stop()

# ============================================================
# TOP HEADER
# ============================================================
st.markdown("""
<div class="topbar">
    <div>
        <div class="eyebrow">Loan Intelligence Platform</div>
        <div class="page-title">Smart Loan Assessment</div>
        <div class="page-subtitle">
            Evaluate loan applications using a machine learning-powered
            risk assessment engine.
        </div>
    </div>
    <div class="online-badge">● Model Online</div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# OVERVIEW PAGE
# ============================================================
if page == "Overview":
    # ============================================================
    # HERO / OVERVIEW
    # ============================================================
    st.markdown("""
    <div class="hero">
        <div class="hero-kicker">CreditWise Loan Intelligence</div>
        <div class="hero-title">Data-driven loan assessment</div>
        <div class="hero-text">
            Make structured assessment decisions using applicant, financial,
            employment and loan information. CreditWise converts application
            inputs into a model prediction with probability and risk indicators.
        </div>
    </div>
    """, unsafe_allow_html=True)

    feature_cols = st.columns(3)

    features = [
        (
            "01",
            "Fast Assessment",
            "Generate a prediction instantly from a completed loan application."
        ),
        (
            "02",
            "Data-Driven",
            "Assessment is powered by a trained machine learning model."
        ),
        (
            "03",
            "Risk Insights",
            "Review approval probability and model-derived risk indicators."
        )
    ]

    for col, (num, title, text) in zip(feature_cols, features):
        with col:
            st.markdown(
                f"""
                <div class="feature-card">
                    <div class="feature-number">{num}</div>
                    <div class="feature-title">{title}</div>
                    <div class="feature-text">{text}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

    st.markdown("""
    <div class="section-card">
        <div class="section-title">Start an assessment</div>
        <div class="section-desc">Use the navigation menu to open Loan Assessment and submit an application.</div>
    </div>
    """, unsafe_allow_html=True)

    st.stop()

# ============================================================
# LOAN ASSESSMENT PAGE
# ============================================================
if page != "Loan Assessment":
    st.stop()

# ============================================================
# PROGRESS INDICATOR
# ============================================================
st.markdown("""
<div class="progress-wrap">
    <div class="progress-label">APPLICATION PROFILE</div>
    <div class="progress-steps">
        <div class="progress-step">01 Applicant</div>
        <div class="progress-step">02 Financial</div>
        <div class="progress-step">03 Employment</div>
        <div class="progress-step">04 Loan</div>
        <div class="progress-step">05 Risk</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# 01 — APPLICANT INFORMATION
# ============================================================
st.markdown("""
<div class="section-card">
    <div class="section-header">
        <div class="section-number">01</div>
        <div>
            <div class="section-title">Applicant Information</div>
            <div class="section-desc">Basic personal details of the applicant.</div>
        </div>
    </div>
""", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)

with c1:
    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=30,
        step=1
    )

with c2:
    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

with c3:
    dependents = st.number_input(
        "Dependents",
        min_value=0,
        max_value=10,
        value=0,
        step=1
    )

with c4:
    marital_status = st.selectbox(
        "Marital Status",
        ["Single", "Married"]
    )

st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# 02 — FINANCIAL PROFILE
# ============================================================
st.markdown("""
<div class="section-card">
    <div class="section-header">
        <div class="section-number">02</div>
        <div>
            <div class="section-title">Financial Profile</div>
            <div class="section-desc">Income, savings and credit-related indicators.</div>
        </div>
    </div>
""", unsafe_allow_html=True)

c1, c2 = st.columns(2)

with c1:
    applicant_income = st.number_input(
        "Applicant Income",
        min_value=0.0,
        value=50000.0,
        step=1000.0,
        format="%.2f",
        help="Applicant's income used by the model."
    )

with c2:
    coapplicant_income = st.number_input(
        "Coapplicant Income",
        min_value=0.0,
        value=0.0,
        step=1000.0,
        format="%.2f",
        help="Income contributed by the coapplicant."
    )

c1, c2, c3 = st.columns(3)

with c1:
    savings = st.number_input(
        "Savings",
        min_value=0.0,
        value=100000.0,
        step=5000.0,
        format="%.2f",
        help="Applicant savings amount."
    )

with c2:
    credit_score = st.number_input(
        "Credit Score",
        min_value=300,
        max_value=900,
        value=700,
        step=1,
        help="Credit score supplied to the model. Higher scores generally indicate stronger credit history."
    )

with c3:
    dti_ratio = st.number_input(
        "DTI Ratio",
        min_value=0.0,
        max_value=2.0,
        value=0.30,
        step=0.01,
        format="%.2f",
        help="Debt-to-income ratio. This value is also used for model feature engineering."
    )

st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# 03 — EMPLOYMENT & BACKGROUND
# ============================================================
st.markdown("""
<div class="section-card">
    <div class="section-header">
        <div class="section-number">03</div>
        <div>
            <div class="section-title">Employment & Background</div>
            <div class="section-desc">Employment, education and property information.</div>
        </div>
    </div>
""", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)

with c1:
    employment_status = st.selectbox(
        "Employment Status",
        ["Employed", "Self-Employed", "Unemployed"]
    )

with c2:
    employer_category = st.selectbox(
        "Employer Category",
        ["Private", "Government", "Business", "Other"]
    )

with c3:
    education_level = st.selectbox(
        "Education Level",
        ["Graduate", "Not Graduate"]
    )

with c4:
    property_area = st.selectbox(
        "Property Area",
        ["Urban", "Semiurban", "Rural"]
    )

st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# 04 — LOAN DETAILS
# ============================================================
st.markdown("""
<div class="section-card">
    <div class="section-header">
        <div class="section-number">04</div>
        <div>
            <div class="section-title">Loan Details</div>
            <div class="section-desc">Requested loan amount, duration and purpose.</div>
        </div>
    </div>
""", unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)

with c1:
    loan_amount = st.number_input(
        "Loan Amount",
        min_value=0.0,
        value=200000.0,
        step=5000.0,
        format="%.2f",
        help="Requested loan amount."
    )

with c2:
    loan_term = st.number_input(
        "Loan Term",
        min_value=1,
        max_value=480,
        value=120,
        step=1,
        help="Loan duration as represented in the training data."
    )

with c3:
    loan_purpose = st.selectbox(
        "Loan Purpose",
        ["Home", "Education", "Auto", "Personal", "Business", "Other"]
    )

st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# 05 — RISK & COLLATERAL
# ============================================================
st.markdown("""
<div class="section-card">
    <div class="section-header">
        <div class="section-number">05</div>
        <div>
            <div class="section-title">Risk & Collateral</div>
            <div class="section-desc">Collateral and existing loan obligations.</div>
        </div>
    </div>
""", unsafe_allow_html=True)

c1, c2 = st.columns(2)

with c1:
    collateral_value = st.number_input(
        "Collateral Value",
        min_value=0.0,
        value=0.0,
        step=5000.0,
        format="%.2f",
        help="Value of collateral associated with the application."
    )

with c2:
    existing_loans = st.number_input(
        "Existing Loans",
        min_value=0,
        max_value=20,
        value=0,
        step=1,
        help="Number of existing loans."
    )

st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# ASSESSMENT CTA
# ============================================================
st.markdown("""
<div class="section-card">
    <div class="section-title">Ready for assessment?</div>
    <div class="cta-note">
        Review the application details above, then run the trained model
        against the submitted values.
    </div>
""", unsafe_allow_html=True)

predict = st.button("Assess Loan Application →", type="primary")

st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# PREDICTION PIPELINE — PRESERVED EXACTLY
# ============================================================
if predict:
    with st.spinner("Model is analyzing the application..."):
        input_data = pd.DataFrame(
            [
                {
                    "Age": age,
                    "Gender": gender,
                    "Dependents": dependents,
                    "Education_Level": education_level,
                    "Marital_Status": marital_status,
                    "Employment_Status": employment_status,
                    "Employer_Category": employer_category,
                    "Applicant_Income": applicant_income,
                    "Coapplicant_Income": coapplicant_income,
                    "Savings": savings,
                    "Loan_Amount": loan_amount,
                    "Loan_Term": loan_term,
                    "Credit_Score": credit_score,
                    "DTI_Ratio": dti_ratio,
                    "Collateral_Value": collateral_value,
                    "Existing_Loans": existing_loans,
                    "Property_Area": property_area,
                    "Loan_Purpose": loan_purpose
                }
            ]
        )

        input_data["Education_Level"] = ml["education_encoder"].transform(
            input_data["Education_Level"]
        )

        categorical_columns = [
            "Employment_Status",
            "Marital_Status",
            "Loan_Purpose",
            "Property_Area",
            "Gender",
            "Employer_Category"
        ]

        encoded = ml["encoder"].transform(
            input_data[categorical_columns]
        )

        encoded_df = pd.DataFrame(
            encoded,
            columns=ml["encoder"].get_feature_names_out(
                categorical_columns
            )
        )

        input_data = input_data.drop(
            columns=categorical_columns
        )

        input_data = pd.concat(
            [
                input_data.reset_index(drop=True),
                encoded_df.reset_index(drop=True)
            ],
            axis=1
        )

        input_data["DTI_Ratio_sq"] = (
            input_data["DTI_Ratio"] ** 2
        )

        input_data["Credit_Score_sq"] = (
            input_data["Credit_Score"] ** 2
        )

        input_data = input_data.drop(
            columns=[
                "Credit_Score",
                "DTI_Ratio"
            ]
        )

        input_data = input_data.reindex(
            columns=ml["feature_columns"],
            fill_value=0
        )

        scaled_input = ml["scaler"].transform(
            input_data
        )

        prediction = ml["model"].predict(
            scaled_input
        )

        probabilities = ml["model"].predict_proba(
            scaled_input
        )[0]

        predicted_label = (
            ml["target_encoder"]
            .inverse_transform(prediction)[0]
        )

        target_classes = (
            ml["target_encoder"].classes_
        )

        yes_index = list(
            target_classes
        ).index("Yes")

        approval_probability = (
            probabilities[yes_index]
        )

        rejection_probability = (
            1 - approval_probability
        )

        if approval_probability >= 0.75:
            risk_level = "Low"
        elif approval_probability >= 0.50:
            risk_level = "Medium"
        else:
            risk_level = "High"

        approved = (
            predicted_label == "Yes"
        )

        if approved:
            status = "Approved"
            description = (
                "This applicant meets the criteria "
                "for loan approval based on the "
                "current risk assessment."
            )
        else:
            status = "Not Approved"
            description = (
                "This applicant does not meet the "
                "current approval criteria due to "
                "elevated risk factors."
            )

        # Save the latest model result so Prediction Insights can display it.
        st.session_state["prediction_result"] = {
            "status": status,
            "description": description,
            "approved": approved,
            "approval_probability": approval_probability,
            "rejection_probability": rejection_probability,
            "risk_level": risk_level,
            "model_confidence": max(probabilities),
            "summary": {
                "age": age,
                "employment_status": employment_status,
                "credit_score": credit_score,
                "applicant_income": applicant_income,
                "loan_amount": loan_amount,
                "loan_term": loan_term,
                "loan_purpose": loan_purpose,
                "property_area": property_area
            }
        }

    # ========================================================
    # RESULT HEADER
    # ========================================================
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    result_class = "result-approved" if approved else "result-rejected"

    st.markdown(
        f"""
        <div class="{result_class}">
            <div class="result-label">LOAN ASSESSMENT RESULT</div>
            <div class="result-status">{status}</div>
            <div class="result-description">
                {description}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # ========================================================
    # KEY RESULT METRICS
    # ========================================================
    m1, m2, m3 = st.columns(3)

    with m1:
        st.metric(
            "Approval Probability",
            f"{approval_probability:.1%}"
        )

    with m2:
        st.metric(
            "Risk Level",
            risk_level
        )

    with m3:
        st.metric(
            "Model Confidence",
            f"{max(probabilities):.1%}"
        )

    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

    # ========================================================
    # PROBABILITY VISUALIZATION
    # ========================================================
    left, right = st.columns([1.15, 0.85])

    with left:
        st.markdown("""
        <div class="section-card">
            <div class="section-title">Prediction Insights</div>
            <div class="section-desc">
                Model probability distribution for this application.
            </div>
        """, unsafe_allow_html=True)

        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=["Approval", "Rejection"],
                y=[
                    approval_probability * 100,
                    rejection_probability * 100
                ],
                text=[
                    f"{approval_probability:.1%}",
                    f"{rejection_probability:.1%}"
                ],
                textposition="auto",
                marker_line_width=0,
                hovertemplate="%{x}: %{y:.1f}%<extra></extra>"
            )
        )

        fig.update_layout(
            height=330,
            margin=dict(l=10, r=10, t=20, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(
                family="Inter, sans-serif",
                color="#475467"
            ),
            yaxis=dict(
                range=[0, 100],
                title="Probability (%)",
                gridcolor="#eaecf0",
                zeroline=False
            ),
            xaxis=dict(
                title="",
                showgrid=False
            ),
            showlegend=False
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            config={"displayModeBar": False}
        )

        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown("""
        <div class="risk-box">
            <div class="risk-title">Risk Assessment</div>
            <div class="risk-caption">
                Risk level is derived from the existing approval-probability
                thresholds in the application logic.
            </div>
        """, unsafe_allow_html=True)

        st.markdown(
            f"""
            <div class="risk-pill">{risk_level.upper()}</div>
            <div style="margin-top:18px;">
                <div style="font-size:11px;color:#667085;margin-bottom:7px;">
                    Approval probability
                </div>
                <div style="
                    width:100%;
                    height:9px;
                    background:#eaecf0;
                    border-radius:999px;
                    overflow:hidden;
                ">
                    <div style="
                        width:{approval_probability * 100:.2f}%;
                        height:100%;
                        background:#101828;
                        border-radius:999px;
                    "></div>
                </div>
                <div style="
                    display:flex;
                    justify-content:space-between;
                    margin-top:7px;
                    font-size:10px;
                    color:#98a2b3;
                ">
                    <span>0%</span>
                    <span>50%</span>
                    <span>100%</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("""
            <div style="
                margin-top:18px;
                padding-top:15px;
                border-top:1px solid #eaecf0;
                font-size:11px;
                line-height:1.55;
                color:#667085;
            ">
                This indicator is model-derived and should not be interpreted
                as a guaranteed real-world lending risk rating.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ========================================================
    # APPLICATION SUMMARY
    # ========================================================
    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

    st.markdown("""
    <div class="section-title" style="margin:12px 0 14px;">
        Application Summary
    </div>
    """, unsafe_allow_html=True)

    summary_values = [
        ("Applicant Age", f"{age} years", ""),
        ("Employment Status", employment_status, ""),
        ("Credit Score", f"{credit_score}", ""),
        ("Applicant Income", f"{applicant_income:,.0f}", ""),
        ("Loan Amount", f"{loan_amount:,.0f}", ""),
        ("Loan Term", f"{loan_term}", ""),
        ("Loan Purpose", loan_purpose, ""),
        ("Property Area", property_area, "")
    ]

    for start in range(0, len(summary_values), 4):
        cols = st.columns(4)

        for col, (label, value, small) in zip(
            cols,
            summary_values[start:start + 4]
        ):
            with col:
                st.markdown(
                    f"""
                    <div class="info-card">
                        <div class="info-label">{label}</div>
                        <div class="info-value">{value}</div>
                        {f'<div class="info-small">{small}</div>' if small else ''}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        if start == 0:
            st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    # ========================================================
    # MODEL ASSESSMENT
    # ========================================================
    st.markdown("<div style='height:22px'></div>", unsafe_allow_html=True)

    st.markdown("""
    <div class="section-card">
        <div class="section-title">Model Assessment</div>
        <div class="section-desc">
            Direct outputs generated by the trained prediction pipeline.
        </div>
    """, unsafe_allow_html=True)

    a1, a2, a3, a4, a5 = st.columns(5)

    assessment = [
        ("Prediction", status),
        ("Approval Probability", f"{approval_probability:.1%}"),
        ("Rejection Probability", f"{rejection_probability:.1%}"),
        ("Risk Level", risk_level),
        ("Model Confidence", f"{max(probabilities):.1%}")
    ]

    for col, (label, value) in zip(
        [a1, a2, a3, a4, a5],
        assessment
    ):
        with col:
            st.markdown(
                f"""
                <div style="
                    padding:12px 2px;
                    border-right:1px solid #eaecf0;
                ">
                    <div class="info-label">{label}</div>
                    <div style="
                        margin-top:7px;
                        font-size:15px;
                        font-weight:700;
                        color:#101828;
                    ">{value}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown("</div>", unsafe_allow_html=True)

    # ========================================================
    # DISCLAIMER
    # ========================================================
    st.markdown("""
    <div class="disclaimer">
        <strong>Important:</strong> CreditWise is an educational machine
        learning application. Predictions are generated by a trained model
        and should not be treated as real-world financial or lending decisions.
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# FOOTER
# ============================================================
st.markdown("""
<div class="footer">
    <strong>CreditWise</strong><br>
    Machine Learning Loan Assessment Platform<br>
    Educational Project • Machine Learning • Streamlit
</div>
""", unsafe_allow_html=True)
