"""
ShaadiSpend Analytics — Main Dashboard Entry Point
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="ShaadiSpend Analytics",
    page_icon="💍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    [data-testid="stSidebar"] { background-color: #085041; }
    [data-testid="stSidebar"] * { color: #E1F5EE !important; }
    .metric-card {
        background: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 12px;
        padding: 16px 20px;
        text-align: center;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    }
    .metric-val  { font-size: 28px; font-weight: 700; color: #1D9E75; margin: 4px 0; }
    .metric-lbl  { font-size: 13px; color: #666; }
    .metric-sub  { font-size: 11px; color: #999; margin-top: 2px; }
    .section-header {
        font-size: 22px; font-weight: 600; color: #085041;
        border-bottom: 2px solid #1D9E75; padding-bottom: 6px; margin-bottom: 16px;
    }
    .insight-box {
        background: #E1F5EE; border-left: 4px solid #1D9E75;
        border-radius: 0 8px 8px 0; padding: 12px 16px;
        font-size: 14px; color: #085041; margin: 12px 0;
    }
    .stTabs [data-baseweb="tab"] { font-size: 14px; font-weight: 500; }
</style>
""", unsafe_allow_html=True)

# ── Data loader (cached) ───────────────────────────────────────────────────────
@st.cache_data
def load_data():
    raw     = pd.read_csv("data/ShaadiSpend_RAW_2000.csv")
    cleaned = pd.read_csv("data/ShaadiSpend_CLEANED_2000.csv")
    return raw, cleaned

raw_df, df = load_data()

# ── Store in session state so all pages can use it ─────────────────────────────
st.session_state["raw_df"] = raw_df
st.session_state["df"]     = df

# ── Home / Landing page ────────────────────────────────────────────────────────
st.markdown("""
<h1 style='color:#085041; font-size:34px; font-weight:700; margin-bottom:4px;'>
💍 ShaadiSpend Analytics
</h1>
<p style='color:#555; font-size:16px; margin-top:0;'>
Data-Driven Wedding Budget Optimization & Vendor Pricing Intelligence Platform
</p>
""", unsafe_allow_html=True)

st.markdown("---")

col1, col2, col3, col4, col5 = st.columns(5)

kpis = [
    ("2,000", "Survey Respondents", "Synthetic · India-representative"),
    ("119", "Dataset Features", "Raw + Engineered + Encoded"),
    ("₹39.8L", "Avg Wedding Budget", "Post-cleaning median"),
    ("59.6%", "Platform Adoption Intent", "Yes responses (Q25)"),
    ("$104B", "Market Size (2024)", "Indian wedding industry"),
]
for col, (val, lbl, sub) in zip([col1,col2,col3,col4,col5], kpis):
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-val">{val}</div>
            <div class="metric-lbl">{lbl}</div>
            <div class="metric-sub">{sub}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("&nbsp;")

# ── Navigation guide ───────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Dashboard Navigation</div>', unsafe_allow_html=True)

pages_info = [
    ("📊", "1 — Overview",       "Market KPIs, pipeline funnel, demographic breakdown"),
    ("🔍", "2 — EDA",            "12 exploratory charts: budget, social pressure, services, WTP"),
    ("🔗", "3 — Correlation",    "Pearson heatmap + top analytically significant variable pairs"),
    ("👥", "4 — Clustering",     "K-Means customer persona discovery with interactive controls"),
    ("🎯", "5 — Classification", "Random Forest — predict if a customer will adopt the platform"),
    ("📈", "6 — Regression",     "Linear/Ridge regression — predict wedding budget from features"),
    ("🛒", "7 — ARM",            "Apriori association rules — vendor basket pattern discovery"),
]

cols = st.columns(2)
for i, (icon, name, desc) in enumerate(pages_info):
    with cols[i % 2]:
        st.markdown(f"""
        <div style='background:#f9f9f7; border:1px solid #e0e0d8; border-radius:10px;
                    padding:12px 16px; margin-bottom:10px;'>
            <span style='font-size:20px;'>{icon}</span>
            <strong style='color:#085041; margin-left:8px;'>{name}</strong><br>
            <span style='color:#666; font-size:13px; margin-left:30px;'>{desc}</span>
        </div>""", unsafe_allow_html=True)

st.markdown("&nbsp;")
st.markdown("""
<div class="insight-box">
<strong>How to use:</strong> Use the sidebar (left) to navigate between pages.
All analytics run live on the 2,000-row synthetic dataset.
Pages 4–7 include interactive controls — adjust parameters and see results update instantly.
</div>""", unsafe_allow_html=True)

st.markdown("---")
st.caption("ShaadiSpend Analytics · Business Idea Validation Dataset · v1.0 · 2025")
