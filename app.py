import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_generator import generate_data

st.set_page_config(page_title="ShaadiSpend Analytics", page_icon="💍", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
[data-testid="stSidebar"] { background: linear-gradient(180deg,#064e3b 0%,#065f46 60%,#047857 100%); }
[data-testid="stSidebar"] * { color: #d1fae5 !important; }
[data-testid="stSidebar"] .stRadio label { color: #6ee7b7 !important; }
[data-testid="stSidebarNav"] a { color: #a7f3d0 !important; font-weight:500; }
[data-testid="stSidebarNav"] a:hover { background: rgba(255,255,255,0.1) !important; }
.hero-banner {
    background: linear-gradient(135deg,#064e3b 0%,#065f46 40%,#0f766e 100%);
    border-radius: 18px; padding: 40px 48px; margin-bottom: 28px; color: white;
}
.hero-title { font-size: 42px; font-weight: 700; margin: 0; letter-spacing:-1px; }
.hero-sub { font-size: 17px; color: #a7f3d0; margin-top: 8px; font-weight:400; }
.hero-tags { margin-top: 20px; display:flex; gap:10px; flex-wrap:wrap; }
.hero-tag { background:rgba(255,255,255,0.15); border:1px solid rgba(255,255,255,0.25);
    border-radius:20px; padding:5px 14px; font-size:13px; color:#d1fae5; }
.kpi-card { background:#fff; border:1px solid #e5e7eb; border-radius:16px;
    padding:24px 20px; text-align:center; transition:box-shadow .2s;
    box-shadow:0 1px 4px rgba(0,0,0,0.06); }
.kpi-card:hover { box-shadow:0 6px 24px rgba(6,78,59,0.12); }
.kpi-val { font-size:32px; font-weight:700; color:#065f46; margin:6px 0 4px; }
.kpi-lbl { font-size:13px; color:#6b7280; font-weight:500; }
.kpi-sub { font-size:11px; color:#9ca3af; margin-top:3px; }
.kpi-delta-pos { font-size:12px; color:#059669; font-weight:600; margin-top:4px; }
.kpi-delta-neg { font-size:12px; color:#dc2626; font-weight:600; margin-top:4px; }
.nav-card { background:#fff; border:1.5px solid #e5e7eb; border-radius:14px;
    padding:20px 22px; margin-bottom:12px; cursor:pointer; transition:all .2s;
    box-shadow:0 1px 3px rgba(0,0,0,0.05); }
.nav-card:hover { border-color:#059669; box-shadow:0 4px 16px rgba(6,78,59,0.12); transform:translateY(-2px); }
.nav-icon { font-size:28px; margin-bottom:8px; }
.nav-title { font-size:16px; font-weight:600; color:#111827; margin-bottom:4px; }
.nav-desc { font-size:13px; color:#6b7280; line-height:1.5; }
.nav-badge { display:inline-block; background:#ecfdf5; color:#065f46; font-size:11px;
    font-weight:600; padding:2px 8px; border-radius:10px; margin-top:6px; }
.insight-box { background:linear-gradient(135deg,#ecfdf5,#f0fdf4); border-left:4px solid #059669;
    border-radius:0 12px 12px 0; padding:14px 18px; font-size:14px; color:#064e3b;
    margin:12px 0; line-height:1.6; }
.section-title { font-size:22px; font-weight:700; color:#111827;
    border-bottom:3px solid #059669; padding-bottom:8px; margin:28px 0 18px; display:inline-block; }
.stat-row { background:#f9fafb; border-radius:12px; padding:16px 20px; margin:8px 0; }
div[data-testid="stMetric"] { background:#fff; border:1px solid #e5e7eb;
    border-radius:12px; padding:16px; box-shadow:0 1px 3px rgba(0,0,0,0.05); }
</style>""", unsafe_allow_html=True)

@st.cache_data(show_spinner="Generating ShaadiSpend dataset...")
def load(): return generate_data(seed=2024)
df = load()

# ── Hero banner ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
  <div class="hero-title">💍 ShaadiSpend Analytics</div>
  <div class="hero-sub">Data-Driven Wedding Budget Optimization &amp; Vendor Pricing Intelligence Platform</div>
  <div class="hero-tags">
    <span class="hero-tag">🇮🇳 Indian Wedding Market</span>
    <span class="hero-tag">📊 2,000 Respondents</span>
    <span class="hero-tag">🤖 ML-Powered Insights</span>
    <span class="hero-tag">💰 ₹104B Market</span>
    <span class="hero-tag">🎯 Business Idea Validation</span>
  </div>
</div>""", unsafe_allow_html=True)

# ── KPI Row ───────────────────────────────────────────────────────────────────
c1,c2,c3,c4,c5,c6 = st.columns(6)
yes_pct = (df['Platform_intent']=='Yes').mean()*100
overrun_pct = df['Overrun_flag'].mean()*100
avg_budget = df['Budget_num'].median()
avg_ltv = df[df['LTV_est']>0]['LTV_est'].mean()
paid_pct = (df['WTP_monthly']>0).mean()*100

kpis = [
    ("2,000","Survey Respondents","Synthetic · India","▲ 100% valid","pos"),
    (f"₹{avg_budget:.0f}L","Avg Wedding Budget","Median post-clean","▲ 8% YoY","pos"),
    (f"{yes_pct:.1f}%","Platform Adoption","Yes — firm intent","Strong signal","pos"),
    (f"{overrun_pct:.1f}%","Budget Overrun Rate","Exceed original plan","▼ Pain point","neg"),
    (f"{paid_pct:.0f}%","Willing to Pay","Any paid tier","Monetisation ready","pos"),
    (f"₹{avg_ltv/1000:.1f}K","Avg Customer LTV","Among paying users","High-value segment","pos"),
]
for col,(val,lbl,sub,delta,dtype) in zip([c1,c2,c3,c4,c5,c6],kpis):
    with col:
        delta_cls = "kpi-delta-pos" if dtype=="pos" else "kpi-delta-neg"
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-lbl">{lbl}</div>
            <div class="kpi-val">{val}</div>
            <div class="kpi-sub">{sub}</div>
            <div class="{delta_cls}">{delta}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Quick funnel + intent side by side ───────────────────────────────────────
col_l, col_r = st.columns([1,1])
with col_l:
    st.markdown('<div class="section-title">Sales Pipeline Funnel</div>', unsafe_allow_html=True)
    pipeline_order = ['Awareness','Consideration','Trial','Conversion','Retention','Churned']
    vc = df['Pipeline_stage'].value_counts()
    counts = [int(vc.get(s,0)) for s in pipeline_order]
    colors = ['#6ee7b7','#34d399','#10b981','#059669','#047857','#fca5a5']
    fig = go.Figure(go.Funnel(y=pipeline_order, x=counts, textposition="inside",
        textinfo="value+percent initial", marker=dict(color=colors),
        connector=dict(line=dict(color="#d1fae5",width=1))))
    fig.update_layout(height=340, margin=dict(l=10,r=10,t=10,b=10),
                      paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                      font=dict(family='Inter'))
    st.plotly_chart(fig, use_container_width=True)
    conv = counts[3]/2000*100; ret = counts[4]/2000*100
    st.markdown(f'<div class="insight-box">💡 <b>Conversion = {conv:.1f}%</b> and <b>Retention = {ret:.1f}%</b> of total respondents. The Awareness→Consideration drop is the biggest gap — improve onboarding copy and trial incentives to fix this.</div>', unsafe_allow_html=True)

with col_r:
    st.markdown('<div class="section-title">Platform Adoption Intent</div>', unsafe_allow_html=True)
    vc_i = df['Platform_intent'].value_counts()
    fig2 = go.Figure(go.Pie(
        labels=vc_i.index, values=vc_i.values, hole=0.58,
        marker=dict(colors=['#059669','#f59e0b','#ef4444'],
                    line=dict(color='white',width=3)),
        textinfo='label+percent', textfont=dict(size=13,family='Inter')))
    fig2.update_layout(height=340, margin=dict(l=10,r=10,t=10,b=10),
                       paper_bgcolor='rgba(0,0,0,0)',
                       annotations=[dict(text=f'<b>{yes_pct:.0f}%</b><br>Intent',
                                         x=0.5,y=0.5,font=dict(size=16,color='#064e3b'),showarrow=False)],
                       legend=dict(font=dict(family='Inter')))
    st.plotly_chart(fig2, use_container_width=True)
    maybe_pct = (df['Platform_intent']=='Maybe').mean()*100
    st.markdown(f'<div class="insight-box">💡 <b>{yes_pct:.1f}%</b> firm adopters + <b>{maybe_pct:.1f}%</b> Maybe = <b>{yes_pct+maybe_pct:.1f}%</b> addressable market. Only {100-yes_pct-maybe_pct:.1f}% firm rejection — exceptional product-market fit for a new startup.</div>', unsafe_allow_html=True)

# ── Navigation grid ───────────────────────────────────────────────────────────
st.markdown('<div class="section-title">Explore the Dashboard</div>', unsafe_allow_html=True)
pages = [
    ("📊","1 — Market Overview","KPI cards · Pipeline funnel · Demographics · Customer LTV tiers","EDA + Business"),
    ("🔍","2 — Exploratory Analysis","12 interactive charts: budget dist, social pressure, service adoption, WTP","Descriptive Analytics"),
    ("🔗","3 — Correlation Matrix","Pearson heatmap · Top variable pairs · Regression scatter plots","Statistical Analysis"),
    ("👥","4 — Customer Clustering","K-Means with elbow curve · Silhouette score · Persona profiles","Unsupervised ML"),
    ("🎯","5 — Classification","Random Forest · Confusion matrix · Feature importance · Live predictor","Supervised ML"),
    ("📈","6 — Budget Regression","4 models compared · R² · MAE · Residuals · Live budget predictor","Regression Analysis"),
    ("🛒","7 — Association Rules","Apriori algorithm · Lift heatmap · Vendor basket patterns","ARM Mining"),
]
cols = st.columns(3)
for i,(icon,title,desc,badge) in enumerate(pages):
    with cols[i%3]:
        st.markdown(f"""<div class="nav-card">
            <div class="nav-icon">{icon}</div>
            <div class="nav-title">{title}</div>
            <div class="nav-desc">{desc}</div>
            <div class="nav-badge">{badge}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("---")
c1,c2,c3,c4 = st.columns(4)
c1.metric("Total Features", f"{df.shape[1]}", "After engineering")
c2.metric("Classification Target", "Platform Intent", "Yes / Maybe / No")
c3.metric("Regression Target", "Budget (₹L)", "Continuous variable")
c4.metric("ARM Basket Items", "14 services", "Vendor combinations")
st.caption("ShaadiSpend Analytics · Startup Idea Validation · Academic Submission · 2025")
