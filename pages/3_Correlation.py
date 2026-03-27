import streamlit as st, pandas as pd, numpy as np
import plotly.express as px, plotly.graph_objects as go
from scipy import stats
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_generator import generate_data

st.set_page_config(page_title="Correlation | ShaadiSpend", page_icon="🔗", layout="wide")
st.markdown("""<style>
[data-testid="stSidebar"]{background:linear-gradient(180deg,#064e3b,#065f46,#047857);}
[data-testid="stSidebar"] *{color:#d1fae5 !important;}
.section-title{font-size:18px;font-weight:700;color:#064e3b;border-bottom:3px solid #059669;padding-bottom:5px;margin:20px 0 12px;display:inline-block;}
.insight-box{background:linear-gradient(135deg,#ecfdf5,#f0fdf4);border-left:4px solid #059669;border-radius:0 10px 10px 0;padding:10px 14px;font-size:13px;color:#064e3b;margin:8px 0;}
.corr-stat{background:#fff;border:1px solid #e5e7eb;border-radius:10px;padding:14px;text-align:center;}
.corr-r{font-size:26px;font-weight:700;color:#059669;}
.corr-lbl{font-size:12px;color:#6b7280;margin-top:4px;}
</style>""", unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def load(): return generate_data(seed=2024)
df = load()

st.markdown("## 🔗 Correlation Analysis")
st.caption("Pearson correlation matrix + key variable relationships")

corr_vars = {
    'Income_num':'Income (₹L)', 'Budget_num':'Budget (₹L)', 'Guest_count':'Guest Count',
    'Ceremony_count':'Ceremonies', 'Social_pressure':'Social Pressure',
    'Alert_receptiveness':'Alert Recep.', 'Net_satisfaction':'Satisfaction',
    'WTP_monthly':'WTP (₹/mo)', 'LTV_est':'LTV (₹)',
    'Budget_per_guest':'Budget/Guest', 'Services_count':'Services',
    'Emotional_intensity':'Emotional Int.', 'Feature_score_avg':'Feature Score',
    'Intent_enc':'Platform Intent', 'Pipeline_enc':'Pipeline Stage',
}
avail = {k:v for k,v in corr_vars.items() if k in df.columns}
cdf = df[list(avail.keys())].copy(); cdf.columns = list(avail.values())
corr = cdf.corr()

# Key correlation stats
st.markdown('<div class="section-title">Top Correlation Highlights</div>', unsafe_allow_html=True)
pairs=[('Income_num','Budget_num'),('Social_pressure','Overrun_flag'),
       ('Feature_score_avg','Intent_enc'),('Income_num','LTV_est')]
labels_p=[('Income','Budget'),('Social Pressure','Overrun'),('Feature Score','Intent'),('Income','LTV')]
c1,c2,c3,c4=st.columns(4)
for col,(a,b),lbl in zip([c1,c2,c3,c4],pairs,labels_p):
    if a in df.columns and b in df.columns:
        r,_=stats.pearsonr(df[a].fillna(df[a].median()),df[b].fillna(df[b].median()))
        with col:
            color="059669" if r>0 else "ef4444"
            st.markdown(f'<div class="corr-stat"><div class="corr-r" style="color:#{color}">{r:+.3f}</div><div class="corr-lbl">{lbl[0]}<br>↔ {lbl[1]}</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Heatmap
st.markdown('<div class="section-title">Pearson Correlation Matrix</div>', unsafe_allow_html=True)
labels=list(corr.columns); z=corr.values
text=[[f"{v:.2f}" for v in row] for row in z]
fig=go.Figure(go.Heatmap(z=z,x=labels,y=labels,text=text,texttemplate="%{text}",
                          textfont=dict(size=9,color='black'),
                          colorscale='RdYlGn',zmin=-1,zmax=1,
                          colorbar=dict(title='r',thickness=14,len=0.9)))
fig.update_layout(height=600,xaxis=dict(tickangle=-40,tickfont=dict(size=10)),
                  yaxis=dict(tickfont=dict(size=10)),
                  paper_bgcolor='rgba(0,0,0,0)',margin=dict(t=10,b=10,l=10,r=10))
st.plotly_chart(fig,use_container_width=True)
st.markdown('<div class="insight-box">🟢 <b>Green = positive correlation</b> | 🔴 <b>Red = negative</b> | Diagonal = 1.0 (self). Values > |0.4| are analytically significant. Income↔Budget and Feature Score↔Intent are the strongest predictors for regression and classification models respectively.</div>', unsafe_allow_html=True)

# Strong pairs table
st.markdown('<div class="section-title">Significant Variable Pairs (|r| > 0.25)</div>', unsafe_allow_html=True)
pairs_list=[]
cols=list(corr.columns)
for i in range(len(cols)):
    for j in range(i+1,len(cols)):
        r=corr.iloc[i,j]
        if abs(r)>0.25:
            pairs_list.append({'Variable A':cols[i],'Variable B':cols[j],'Pearson r':round(r,3),
                               'Direction':'Positive ↑' if r>0 else 'Negative ↓',
                               'Strength':'Strong' if abs(r)>0.6 else 'Moderate' if abs(r)>0.4 else 'Weak'})
pairs_df=pd.DataFrame(pairs_list).sort_values('Pearson r',key=abs,ascending=False)
st.dataframe(pairs_df,use_container_width=True,hide_index=True)

# Deep-dive scatters
st.markdown('<div class="section-title">Deep-Dive Scatter Plots</div>', unsafe_allow_html=True)
dc1,dc2=st.columns(2)
with dc1:
    r1,_=stats.pearsonr(df['Income_num'].clip(0,120),df['Budget_num'].clip(0,150))
    st.markdown(f"**Income ↔ Budget (r = {r1:.3f})**")
    s=df[df['Budget_num']<150].sample(500,random_state=1)
    fig_s1=px.scatter(s,x='Income_num',y='Budget_num',color='City_tier',trendline='ols',opacity=0.55,
                      color_discrete_map={'Tier 1':'#059669','Tier 2':'#f59e0b','Tier 3/Rural':'#ef4444'},
                      labels={'Income_num':'Income (₹L)','Budget_num':'Budget (₹L)','City_tier':'City Tier'})
    fig_s1.update_layout(height=300,paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',margin=dict(t=10,b=10))
    st.plotly_chart(fig_s1,use_container_width=True)

with dc2:
    r2,_=stats.pearsonr(df['Feature_score_avg'],df['Intent_enc'].fillna(1))
    st.markdown(f"**Feature Score ↔ Platform Intent (r = {r2:.3f})**")
    fig_s2=px.box(df,x='Platform_intent',y='Feature_score_avg',color='Platform_intent',
                  color_discrete_map={'Yes':'#059669','Maybe':'#f59e0b','No':'#ef4444'},
                  labels={'Platform_intent':'Platform Intent','Feature_score_avg':'Avg Feature Rating'})
    fig_s2.update_layout(height=300,showlegend=False,paper_bgcolor='rgba(0,0,0,0)',
                         plot_bgcolor='rgba(0,0,0,0)',margin=dict(t=10,b=10))
    st.plotly_chart(fig_s2,use_container_width=True)
