import streamlit as st
from data_generator import generate_data

st.set_page_config(
    page_title="ShaadiSpend Analytics",
    page_icon="💍",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
[data-testid="stSidebar"]{background-color:#085041;}
[data-testid="stSidebar"] *{color:#E1F5EE !important;}
.metric-card{background:#fff;border:1px solid #e0e0e0;border-radius:12px;
    padding:16px 20px;text-align:center;}
.metric-val{font-size:28px;font-weight:700;color:#1D9E75;margin:4px 0;}
.metric-lbl{font-size:13px;color:#666;}
.metric-sub{font-size:11px;color:#999;margin-top:2px;}
.insight-box{background:#E1F5EE;border-left:4px solid #1D9E75;
    border-radius:0 8px 8px 0;padding:12px 16px;font-size:14px;color:#085041;margin:12px 0;}
</style>""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    return generate_data(seed=2024)

df = load_data()
st.session_state["df"] = df

st.markdown("<h1 style='color:#085041;font-size:34px;font-weight:700;'>💍 ShaadiSpend Analytics</h1>", unsafe_allow_html=True)
st.markdown("<p style='color:#555;font-size:16px;'>Data-Driven Wedding Budget Optimization & Vendor Pricing Intelligence Platform</p>", unsafe_allow_html=True)
st.markdown("---")

c1,c2,c3,c4,c5 = st.columns(5)
kpis=[("2,000","Survey Respondents","Synthetic · India-representative"),
      ("119+","Dataset Features","Raw + Engineered + Encoded"),
      (f"₹{df['Q11_Budget_num_lakhs'].median():.1f}L","Avg Wedding Budget","Median"),
      (f"{(df['Q25_Platform_intent']=='Yes').mean()*100:.1f}%","Platform Adoption Intent","Yes responses"),
      ("$104B","Market Size (2024)","Indian wedding industry")]
for col,(val,lbl,sub) in zip([c1,c2,c3,c4,c5],kpis):
    with col:
        st.markdown(f'<div class="metric-card"><div class="metric-val">{val}</div>'
                    f'<div class="metric-lbl">{lbl}</div><div class="metric-sub">{sub}</div></div>',
                    unsafe_allow_html=True)

st.markdown("&nbsp;")
st.markdown("### Dashboard Navigation")
pages_info=[("📊","1 — Overview","KPI cards, pipeline funnel, demographics"),
            ("🔍","2 — EDA","12 interactive charts with business insights"),
            ("🔗","3 — Correlation","Pearson heatmap + deep-dive scatter plots"),
            ("👥","4 — Clustering","K-Means customer persona discovery"),
            ("🎯","5 — Classification","Random Forest — predict platform adoption"),
            ("📈","6 — Regression","Predict wedding budget from key features"),
            ("🛒","7 — ARM","Apriori — vendor basket association rules")]
cols=st.columns(2)
for i,(icon,name,desc) in enumerate(pages_info):
    with cols[i%2]:
        st.markdown(f"<div style='background:#f9f9f7;border:1px solid #e0e0d8;border-radius:10px;"
                    f"padding:12px 16px;margin-bottom:10px;'><span style='font-size:20px;'>{icon}</span>"
                    f"<strong style='color:#085041;margin-left:8px;'>{name}</strong><br>"
                    f"<span style='color:#666;font-size:13px;margin-left:30px;'>{desc}</span></div>",
                    unsafe_allow_html=True)
st.markdown("---")
st.caption("ShaadiSpend Analytics · Business Idea Validation · v1.0 · 2025")
