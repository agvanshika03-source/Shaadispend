import streamlit as st, pandas as pd, numpy as np, plotly.express as px, plotly.graph_objects as go
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_generator import generate_data

st.set_page_config(page_title="Overview | ShaadiSpend", page_icon="📊", layout="wide")
st.markdown("""<style>
[data-testid="stSidebar"]{background:linear-gradient(180deg,#064e3b,#065f46,#047857);}
[data-testid="stSidebar"] *{color:#d1fae5 !important;}
.section-title{font-size:20px;font-weight:700;color:#064e3b;border-bottom:3px solid #059669;padding-bottom:6px;margin:20px 0 14px;display:inline-block;}
.insight-box{background:linear-gradient(135deg,#ecfdf5,#f0fdf4);border-left:4px solid #059669;border-radius:0 10px 10px 0;padding:11px 15px;font-size:13px;color:#064e3b;margin:10px 0;}
</style>""", unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def load(): return generate_data(seed=2024)
df = load()

st.markdown("## 📊 Market Overview & Demographics")
st.caption("ShaadiSpend Analytics · 2,000 respondents · Indian wedding market")

m1,m2,m3,m4,m5 = st.columns(5)
m1.metric("Respondents", "2,000")
m2.metric("Avg Budget", f"₹{df['Budget_num'].median():.1f}L")
m3.metric("Avg Guests", f"{int(df['Guest_count'].median())}")
m4.metric("Overrun Rate", f"{df['Overrun_flag'].mean()*100:.1f}%")
m5.metric("Avg LTV", f"₹{df[df['LTV_est']>0]['LTV_est'].mean()/1000:.1f}K")

st.markdown("---")
c1,c2 = st.columns(2)

with c1:
    st.markdown('<div class="section-title">City Tier Distribution</div>', unsafe_allow_html=True)
    vc = df['City_tier'].value_counts()
    fig = px.bar(x=vc.index, y=vc.values, color=vc.index,
                 color_discrete_map={'Tier 1':'#059669','Tier 2':'#34d399','Tier 3/Rural':'#6ee7b7'},
                 labels={'x':'City Tier','y':'Respondents','color':'Tier'},
                 text=vc.values)
    fig.update_traces(textposition='outside')
    fig.update_layout(height=320, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                      showlegend=False, margin=dict(t=10,b=10,l=10,r=10))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('<div class="insight-box">💡 <b>Tier 2 cities dominate (42%)</b> — largest volume segment with rapidly rising aspirations. Target for freemium growth; Tier 1 for premium revenue.</div>', unsafe_allow_html=True)

with c2:
    st.markdown('<div class="section-title">Income Band Distribution</div>', unsafe_allow_html=True)
    order = ['<5L','5-10L','10-20L','20-40L','40-75L','>75L']
    vc2 = df['Income_band'].value_counts().reindex(order,fill_value=0)
    fig2 = px.bar(x=vc2.index, y=vc2.values, color=vc2.values,
                  color_continuous_scale='Greens', labels={'x':'Income Band','y':'Count','color':'Count'}, text=vc2.values)
    fig2.update_traces(textposition='outside')
    fig2.update_layout(height=320, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                       coloraxis_showscale=False, margin=dict(t=10,b=10,l=10,r=10))
    st.plotly_chart(fig2, use_container_width=True)
    st.markdown('<div class="insight-box">💡 <b>₹10–40L income band holds 52%</b> of respondents — the core target market for ShaadiSpend. Design pricing and features around this segment first.</div>', unsafe_allow_html=True)

c3,c4 = st.columns(2)
with c3:
    st.markdown('<div class="section-title">Wedding Type Preference</div>', unsafe_allow_html=True)
    vc3 = df['Wedding_type'].value_counts()
    fig3 = px.pie(names=vc3.index, values=vc3.values, hole=0.45,
                  color_discrete_sequence=['#059669','#34d399','#10b981','#6ee7b7','#a7f3d0'])
    fig3.update_layout(height=300, paper_bgcolor='rgba(0,0,0,0)', margin=dict(t=10,b=10,l=10,r=10))
    st.plotly_chart(fig3, use_container_width=True)
    st.markdown('<div class="insight-box">💡 <b>Destination weddings (30% combined)</b> have 35% higher average budgets and near-universal wedding planner adoption — the highest LTV customer segment.</div>', unsafe_allow_html=True)

with c4:
    st.markdown('<div class="section-title">Customer Value Tier (LTV-Based)</div>', unsafe_allow_html=True)
    vc4 = df['Customer_tier'].value_counts()
    colors4 = {'High-value':'#059669','Mid-value':'#34d399','Low-value':'#6ee7b7','Non-customer':'#f87171'}
    fig4 = px.bar(x=vc4.index, y=vc4.values, color=vc4.index,
                  color_discrete_map=colors4, labels={'x':'Customer Tier','y':'Count','color':'Tier'}, text=vc4.values)
    fig4.update_traces(textposition='outside')
    fig4.update_layout(height=300, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                       showlegend=False, margin=dict(t=10,b=10,l=10,r=10))
    st.plotly_chart(fig4, use_container_width=True)
    hv = (df['Customer_tier']=='High-value').mean()*100
    st.markdown(f'<div class="insight-box">💡 <b>{hv:.1f}% are High-value customers</b> (LTV > ₹20K). Focus acquisition budget on this segment for fastest path to revenue.</div>', unsafe_allow_html=True)

st.markdown("---")
st.markdown('<div class="section-title">Customer Segment Profile Table</div>', unsafe_allow_html=True)
profile = df.groupby('Customer_tier').agg(
    Count=('Respondent_ID','count'),
    Avg_Budget=('Budget_num','mean'),
    Avg_Income=('Income_num','mean'),
    Avg_LTV=('LTV_est','mean'),
    Pct_Yes=('Platform_intent',lambda x:(x=='Yes').mean()*100),
    Avg_WTP=('WTP_monthly','mean')
).round(1).reset_index()
Avg_Budget_col = 'Avg_Budget'; cols_rename = {'Customer_tier':'Tier','Count':'Count','Avg_Budget':'Avg Budget (₹L)',
    'Avg_Income':'Avg Income (₹L)','Avg_LTV':'Avg LTV (₹)','Pct_Yes':'% Yes Intent','Avg_WTP':'Avg WTP (₹/mo)'}
profile.columns = [cols_rename.get(c,c) for c in profile.columns]
st.dataframe(profile, use_container_width=True, hide_index=True)
