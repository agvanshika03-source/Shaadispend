import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_generator import generate_data

st.set_page_config(page_title="Cohort Analysis | ShaadiSpend", page_icon="📉", layout="wide")
st.markdown("""
<style>
[data-testid="stSidebar"]{background:linear-gradient(180deg,#064e3b,#065f46,#047857);}
[data-testid="stSidebar"] *{color:#d1fae5 !important;}
.section-title{font-size:20px;font-weight:700;color:#064e3b;border-bottom:3px solid #059669;
    padding-bottom:6px;margin:24px 0 14px;display:inline-block;}
.insight-box{background:linear-gradient(135deg,#ecfdf5,#f0fdf4);border-left:4px solid #059669;
    border-radius:0 10px 10px 0;padding:11px 15px;font-size:13px;color:#064e3b;margin:10px 0;line-height:1.6;}
</style>""", unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def load(): return generate_data(seed=2024)
df = load()

st.markdown("""
<div style='background:linear-gradient(135deg,#064e3b,#065f46);border-radius:16px;
    padding:28px 36px;margin-bottom:20px;color:white;'>
  <div style='font-size:34px;font-weight:700;'>📉 Cohort & Cross-Tab Analysis</div>
  <div style='font-size:15px;color:#a7f3d0;margin-top:6px;'>
    Mark 3 — Multi-variable descriptive analytics with logical explanations
  </div>
</div>""", unsafe_allow_html=True)

# ── Chart 1: Overrun heatmap — income band × city tier ────────────────────────
st.markdown('<div class="section-title">Cohort 1 — Budget Overrun Rate: Income Band × City Tier (%)</div>', unsafe_allow_html=True)

income_order = ['<5L','5-10L','10-20L','20-40L','40-75L','>75L']
overrun_heat = df.groupby(['Income_band','City_tier'])['Overrun_flag'].mean().mul(100).round(1).reset_index()
overrun_heat.columns = ['Income Band','City Tier','Overrun Rate %']
overrun_pivot = overrun_heat.pivot(index='Income Band',columns='City Tier',values='Overrun Rate %')
overrun_pivot = overrun_pivot.reindex([b for b in income_order if b in overrun_pivot.index])

fig1 = px.imshow(overrun_pivot, color_continuous_scale='RdYlGn_r',
                 labels={'color':'Overrun %'}, text_auto='.1f', aspect='auto',
                 zmin=0, zmax=100)
fig1.update_layout(height=340, paper_bgcolor='rgba(0,0,0,0)', margin=dict(t=10,b=10))
st.plotly_chart(fig1, use_container_width=True)
st.markdown('<div class="insight-box">💡 <b>Key finding:</b> Overrun rates are HIGHEST in Tier 1 + low income bands (₹5-10L) — families with big-city social pressure but limited budgets. Tier 3 families are most financially disciplined. This cohort (Tier 1, ₹5-20L) is ShaadiSpend\'s most urgent target — they need the Emotional Spend Alert feature most.</div>', unsafe_allow_html=True)

# ── Chart 2: Satisfaction by wedding type × city ──────────────────────────────
st.markdown('<div class="section-title">Cohort 2 — Avg Satisfaction: Wedding Type × City Tier</div>', unsafe_allow_html=True)

sat_cross = df.groupby(['Wedding_type','City_tier'])['Sat_overall'].mean().round(2).reset_index()
fig2 = px.bar(sat_cross, x='Wedding_type', y='Sat_overall', color='City_tier',
              barmode='group', color_discrete_map={'Tier 1':'#059669','Tier 2':'#34d399','Tier 3/Rural':'#6ee7b7'},
              labels={'Wedding_type':'Wedding Type','Sat_overall':'Avg Satisfaction (0-10)','City_tier':'City Tier'},
              text='Sat_overall')
fig2.update_traces(textposition='outside', texttemplate='%{text:.1f}')
fig2.update_layout(height=360, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                   xaxis_tickangle=-15, margin=dict(t=20,b=10),
                   legend=dict(orientation='h', y=1.1))
st.plotly_chart(fig2, use_container_width=True)
st.markdown('<div class="insight-box">💡 <b>Key finding:</b> Destination wedding customers report LOWER satisfaction despite spending more — suggesting unmet expectations from premium pricing. Intimate weddings in Tier 3 show highest satisfaction-to-spend ratio. ShaadiSpend should market the "right spend" angle, not just "save money."</div>', unsafe_allow_html=True)

# ── Chart 3: WTP by age group ─────────────────────────────────────────────────
st.markdown('<div class="section-title">Cohort 3 — WTP Distribution by Age Group (Stacked %)</div>', unsafe_allow_html=True)

age_order = ['18-24','25-30','31-36','37-45','45+']
wtp_cross = pd.crosstab(df['Age_group'], df['WTP_band'], normalize='index').mul(100).round(1)
wtp_cross = wtp_cross.reindex([a for a in age_order if a in wtp_cross.index])
wtp_order = ['Free only','<₹199/mo','₹200-499/mo','₹500-999/mo','₹1000+/mo','One-time ₹999']
wtp_cols  = [c for c in wtp_order if c in wtp_cross.columns]

fig3 = px.bar(wtp_cross[wtp_cols].reset_index(), x='Age_group', y=wtp_cols,
              barmode='stack',
              color_discrete_sequence=['#d1fae5','#6ee7b7','#34d399','#059669','#047857','#f59e0b'],
              labels={'Age_group':'Age Group','value':'% Respondents','variable':'WTP Tier'})
fig3.update_layout(height=360, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                   legend=dict(orientation='h', y=-0.2), margin=dict(t=20,b=60))
st.plotly_chart(fig3, use_container_width=True)
st.markdown('<div class="insight-box">💡 <b>Key finding:</b> The 25-36 age group shows highest willingness for ₹200-499/mo plans — the prime earning and wedding-planning years. 45+ respondents (parent planners) lean toward one-time fees. <b>Price the premium plan at ₹299/mo for millennials; offer ₹999 one-time plan for parent segment.</b></div>', unsafe_allow_html=True)

# ── Chart 4: Platform intent by education × income ────────────────────────────
st.markdown('<div class="section-title">Cohort 4 — Platform Intent by Education × Income Band</div>', unsafe_allow_html=True)

edu_order  = ['Up to 12th','Undergraduate','Postgraduate','Professional','Doctoral']
intent_edu = df.groupby(['Education','Income_band'])['Intent_enc'].mean().round(2).reset_index()
intent_pivot = intent_edu.pivot(index='Education', columns='Income_band', values='Intent_enc')
intent_pivot = intent_pivot.reindex([e for e in edu_order if e in intent_pivot.index])
intent_pivot = intent_pivot.reindex(columns=[b for b in ['<5L','5-10L','10-20L','20-40L','40-75L','>75L'] if b in intent_pivot.columns])

fig4 = px.imshow(intent_pivot, color_continuous_scale='Greens',
                 labels={'color':'Avg Intent (0-2)'}, text_auto='.2f', aspect='auto',
                 zmin=0, zmax=2)
fig4.update_layout(height=320, paper_bgcolor='rgba(0,0,0,0)', margin=dict(t=10,b=10))
st.plotly_chart(fig4, use_container_width=True)
st.markdown('<div class="insight-box">💡 <b>Key finding:</b> Intent score peaks at Postgraduate + ₹20-40L income — the classic tech-savvy, financially aware professional demographic. This cohort should be the primary ICP (Ideal Customer Profile) for ShaadiSpend\'s initial user acquisition campaigns on LinkedIn and Instagram.</div>', unsafe_allow_html=True)

# ── Chart 5: Emotional intensity by social pressure × wedding type ─────────────
st.markdown('<div class="section-title">Cohort 5 — Emotional Intensity by Social Pressure Score</div>', unsafe_allow_html=True)

c5a, c5b = st.columns(2)
with c5a:
    emo_sp = df.groupby('Social_pressure')['Emotional_intensity'].mean().reset_index()
    fig5a = px.bar(emo_sp, x='Social_pressure', y='Emotional_intensity',
                   color='Emotional_intensity', color_continuous_scale='Reds',
                   labels={'Social_pressure':'Social Pressure Score','Emotional_intensity':'Avg Emotional Intensity'},
                   text=emo_sp['Emotional_intensity'].round(2))
    fig5a.update_traces(textposition='outside')
    fig5a.update_layout(height=320, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                        coloraxis_showscale=False, margin=dict(t=20,b=10))
    st.plotly_chart(fig5a, use_container_width=True)

with c5b:
    emo_wt = df.groupby('Wedding_type')['Emotional_intensity'].mean().sort_values(ascending=False)
    fig5b = px.bar(x=emo_wt.values, y=emo_wt.index, orientation='h',
                   color=emo_wt.values, color_continuous_scale='Reds',
                   labels={'x':'Avg Emotional Intensity','y':'Wedding Type'},
                   text=[f'{v:.2f}' for v in emo_wt.values])
    fig5b.update_traces(textposition='outside')
    fig5b.update_layout(height=320, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                        coloraxis_showscale=False, yaxis=dict(autorange='reversed'),
                        margin=dict(t=20,b=10,r=60))
    st.plotly_chart(fig5b, use_container_width=True)

st.markdown('<div class="insight-box">💡 <b>Key finding:</b> Emotional intensity rises sharply with social pressure — confirming the causal link that drives irrational spending. Grand weddings have the highest emotional intensity, directly explaining their disproportionate overrun rates. The Emotional Spend Detector feature should trigger alert thresholds at pressure scores ≥ 4.</div>', unsafe_allow_html=True)

# ── Chart 6: Services count by budget band (violin) ──────────────────────────
st.markdown('<div class="section-title">Cohort 6 — Services Hired Distribution by Budget Band</div>', unsafe_allow_html=True)

budget_order_v = ['<5L','5-10L','10-20L','20-40L','40-75L','75L-1Cr','>1Cr']
df_bv = df[df['Budget_band'].isin(budget_order_v)].copy()
fig6 = px.box(df_bv, x='Budget_band', y='Services_count',
              color='Budget_band', category_orders={'Budget_band': budget_order_v},
              color_discrete_sequence=px.colors.sequential.Greens_r,
              labels={'Budget_band':'Budget Band','Services_count':'Services Hired Count'})
fig6.update_layout(height=360, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                   showlegend=False, margin=dict(t=20,b=10))
st.plotly_chart(fig6, use_container_width=True)
st.markdown('<div class="insight-box">💡 <b>Key finding:</b> Higher budget bands hire significantly more services — but variance also increases sharply. The ₹20-40L band shows the widest spread (some families hire 4 services, others 12) — indicating this is the most advice-hungry segment where ShaadiSpend\'s personalized recommendations add the most value.</div>', unsafe_allow_html=True)

# ── Chart 7: Decision authority vs overrun ────────────────────────────────────
st.markdown('<div class="section-title">Cohort 7 — Overrun Rate & Spend by Decision Authority</div>', unsafe_allow_html=True)

c7a, c7b = st.columns(2)
with c7a:
    auth_overrun = df.groupby('Decision_authority')['Overrun_flag'].mean().mul(100).sort_values(ascending=False)
    fig7a = px.bar(x=auth_overrun.values, y=auth_overrun.index, orientation='h',
                   color=auth_overrun.values, color_continuous_scale='RdYlGn_r',
                   labels={'x':'Overrun Rate (%)','y':'Decision Authority'},
                   text=[f'{v:.1f}%' for v in auth_overrun.values])
    fig7a.update_traces(textposition='outside')
    fig7a.update_layout(height=300, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                        coloraxis_showscale=False, yaxis=dict(autorange='reversed'),
                        margin=dict(t=20,b=10,r=60))
    st.plotly_chart(fig7a, use_container_width=True)

with c7b:
    auth_budget = df.groupby('Decision_authority')['Budget_num'].mean().sort_values(ascending=False)
    fig7b = px.bar(x=auth_budget.values, y=auth_budget.index, orientation='h',
                   color=auth_budget.values, color_continuous_scale='Greens',
                   labels={'x':'Avg Budget (₹L)','y':'Decision Authority'},
                   text=[f'₹{v:.1f}L' for v in auth_budget.values])
    fig7b.update_traces(textposition='outside')
    fig7b.update_layout(height=300, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                        coloraxis_showscale=False, yaxis=dict(autorange='reversed'),
                        margin=dict(t=20,b=10,r=80))
    st.plotly_chart(fig7b, use_container_width=True)

st.markdown('<div class="insight-box">💡 <b>Key finding:</b> "Father primarily" decisions show highest budgets but also highest overrun rates — reflecting status-driven spending without analytical discipline. "Couple jointly" decisions show lower overrun rates despite similar budgets. <b>Marketing angle:</b> "Give the couple control — plan smarter together with ShaadiSpend."</div>', unsafe_allow_html=True)

# ── Chart 8: Geographic budget heatmap by state ───────────────────────────────
st.markdown('<div class="section-title">Cohort 8 — Average Budget & Intent by State</div>', unsafe_allow_html=True)

state_summary = df.groupby('State').agg(
    Avg_Budget=('Budget_num','mean'),
    Pct_Yes=('Platform_intent',lambda x:(x=='Yes').mean()*100),
    Count=('Respondent_ID','count'),
    Overrun_Rate=('Overrun_flag','mean')
).round(1).reset_index()
state_summary.columns = ['State','Avg Budget (₹L)','% Yes Intent','Count','Overrun Rate']

fig8 = px.scatter(state_summary, x='Avg Budget (₹L)', y='% Yes Intent',
                  size='Count', color='Overrun Rate',
                  text='State', color_continuous_scale='RdYlGn_r',
                  labels={'Avg Budget (₹L)':'Avg Wedding Budget (₹L)',
                          '% Yes Intent':'% Platform Yes Intent',
                          'Overrun Rate':'Overrun Rate'},
                  size_max=40)
fig8.update_traces(textposition='top center', textfont=dict(size=10))
fig8.update_layout(height=420, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                   coloraxis_colorbar=dict(title='Overrun Rate'), margin=dict(t=20,b=10))
st.plotly_chart(fig8, use_container_width=True)
st.markdown('<div class="insight-box">💡 <b>Key finding:</b> States in the top-right quadrant (high budget + high intent + large bubble) are priority expansion markets. States with HIGH overrun rate (red) AND high budget are the most urgent targets — families spending most and suffering most from unplanned costs. Delhi NCR and Maharashtra dominate this segment.</div>', unsafe_allow_html=True)

st.dataframe(state_summary.sort_values('Avg Budget (₹L)', ascending=False),
             use_container_width=True, hide_index=True)
