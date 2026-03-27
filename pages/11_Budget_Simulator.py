import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_generator import generate_data

st.set_page_config(page_title="Budget Simulator | ShaadiSpend", page_icon="🎛️", layout="wide")
st.markdown("""
<style>
[data-testid="stSidebar"]{background:linear-gradient(180deg,#064e3b,#065f46,#047857);}
[data-testid="stSidebar"] *{color:#d1fae5 !important;}
.section-title{font-size:20px;font-weight:700;color:#064e3b;border-bottom:3px solid #059669;
    padding-bottom:6px;margin:24px 0 14px;display:inline-block;}
.insight-box{background:linear-gradient(135deg,#ecfdf5,#f0fdf4);border-left:4px solid #059669;
    border-radius:0 10px 10px 0;padding:11px 15px;font-size:13px;color:#064e3b;margin:10px 0;line-height:1.6;}
.warn-box{background:linear-gradient(135deg,#fffbeb,#fef9c3);border-left:4px solid #f59e0b;
    border-radius:0 10px 10px 0;padding:11px 15px;font-size:13px;color:#78350f;margin:10px 0;}
.result-card{border-radius:16px;padding:24px 28px;color:white;text-align:center;margin:12px 0;}
.saving-tag{background:#dcfce7;color:#166534;border-radius:10px;padding:4px 14px;
    font-size:13px;font-weight:600;display:inline-block;margin-top:6px;}
</style>""", unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def load(): return generate_data(seed=2024)
df = load()

@st.cache_data(show_spinner="Training budget model...")
def train_model():
    feat_cols = ['Income_num','Guest_count','Ceremony_count','City_enc',
                 'Destination_flag','Services_count','Social_pressure','WTP_monthly']
    avail = [c for c in feat_cols if c in df.columns]
    data  = df[avail+['Budget_num']].dropna()
    data  = data[data['Budget_num']<200]
    X     = data[avail]; y = data['Budget_num']
    Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.2,random_state=42)
    scaler = StandardScaler()
    Xtr_s  = scaler.fit_transform(Xtr); Xte_s = scaler.transform(Xte)
    mdl    = GradientBoostingRegressor(n_estimators=200,max_depth=5,random_state=42)
    mdl.fit(Xtr_s,ytr)
    yp     = mdl.predict(Xte_s)
    r2     = 1 - np.sum((yte-yp)**2)/np.sum((yte-yte.mean())**2)
    mae    = np.mean(np.abs(yte-yp))
    return mdl, scaler, avail, round(r2,4), round(mae,2)

mdl, scaler, feat_used, r2, mae = train_model()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='background:linear-gradient(135deg,#064e3b,#065f46);border-radius:16px;
    padding:28px 36px;margin-bottom:20px;color:white;'>
  <div style='font-size:34px;font-weight:700;'>🎛️ What-If Budget Simulator</div>
  <div style='font-size:15px;color:#a7f3d0;margin-top:6px;'>
    Interactive regression model — see how each decision impacts your wedding budget in real time
  </div>
</div>""", unsafe_allow_html=True)

st.markdown(f'<div class="insight-box">🤖 <b>Powered by Gradient Boosting Regressor</b> | R² = {r2} | MAE = ₹{mae:.1f}L | Trained on 2,000 Indian wedding respondents. Adjust the inputs below and your predicted budget updates instantly.</div>', unsafe_allow_html=True)

# ── Input controls ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">Your Wedding Parameters</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("#### 👤 Profile")
    income      = st.slider("Annual household income (₹L)", 3.0, 120.0, 25.0, 1.0)
    city_tier   = st.selectbox("City tier", ["Tier 1","Tier 2","Tier 3/Rural"])
    city_enc    = {"Tier 1":3,"Tier 2":2,"Tier 3/Rural":1}[city_tier]
    social_p    = st.slider("Social pressure level (1=low, 5=high)", 1, 5, 3)
    wtp         = st.slider("Monthly budget for planning tools (₹)", 0, 1200, 299)

with col2:
    st.markdown("#### 💍 Wedding Details")
    guests      = st.slider("Total guest count", 20, 1500, 300, 10)
    ceremonies  = st.slider("Number of ceremonies", 1, 6, 3)
    dest_flag   = st.checkbox("Destination wedding?", value=False)
    services_n  = st.slider("Number of services to hire", 3, 14, 8)

with col3:
    st.markdown("#### 📊 Benchmark Comparison")
    tier_avg = df[df['City_tier']==city_tier]['Budget_num'].median()
    income_avg = df[np.abs(df['Income_num']-income)<10]['Budget_num'].median() if len(df[np.abs(df['Income_num']-income)<10])>10 else df['Budget_num'].median()
    guest_avg  = df[(df['Guest_count']>=guests*0.7)&(df['Guest_count']<=guests*1.3)]['Budget_num'].median() if len(df[(df['Guest_count']>=guests*0.7)&(df['Guest_count']<=guests*1.3)])>10 else df['Budget_num'].median()
    st.metric("Median budget — your city tier",  f"₹{tier_avg:.1f}L")
    st.metric("Median budget — similar income",   f"₹{income_avg:.1f}L")
    st.metric("Median budget — similar guests",   f"₹{guest_avg:.1f}L")

# ── Predict ───────────────────────────────────────────────────────────────────
live_row = {c: float(df[c].median()) for c in feat_used}
overrides = {'Income_num':income,'Guest_count':guests,'Ceremony_count':ceremonies,
             'City_enc':city_enc,'Destination_flag':int(dest_flag),
             'Services_count':services_n,'Social_pressure':social_p,'WTP_monthly':wtp}
for k,v in overrides.items():
    if k in live_row: live_row[k] = v

X_live    = pd.DataFrame([live_row])[feat_used]
predicted = max(1.0, mdl.predict(scaler.transform(X_live))[0])
low_est   = max(0, predicted - mae)
high_est  = predicted + mae

# Colour based on vs city median
diff_pct  = (predicted - tier_avg) / tier_avg * 100
if diff_pct < -10:
    res_color = "#059669"; res_label = "Below city average ✅"
elif diff_pct > 20:
    res_color = "#dc2626"; res_label = "Above city average ⚠️"
else:
    res_color = "#d97706"; res_label = "Near city average 🔵"

# ── Result display ─────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-title">Your Predicted Wedding Budget</div>', unsafe_allow_html=True)

pr1, pr2, pr3 = st.columns([2,1,1])
with pr1:
    st.markdown(f"""
    <div class="result-card" style="background:linear-gradient(135deg,{res_color},{res_color}cc);">
        <div style="font-size:15px;opacity:0.85;">Predicted Total Budget</div>
        <div style="font-size:54px;font-weight:700;margin:8px 0;">₹{predicted:.1f}L</div>
        <div style="font-size:14px;opacity:0.85;">Range: ₹{low_est:.1f}L – ₹{high_est:.1f}L</div>
        <div style="margin-top:10px;font-size:13px;background:rgba(255,255,255,0.2);
                    padding:6px 16px;border-radius:20px;display:inline-block;">{res_label}</div>
    </div>""", unsafe_allow_html=True)

with pr2:
    st.metric("vs City Median",    f"₹{tier_avg:.1f}L",  f"{diff_pct:+.1f}%")
    st.metric("Budget per Guest",  f"₹{predicted/guests*100:.0f}K",  "Per 100 guests")

with pr3:
    st.metric("Model Accuracy",    f"R² = {r2}",  f"MAE ±₹{mae:.1f}L")
    saving = max(0, predicted - tier_avg)
    st.metric("Potential Saving",  f"₹{saving:.1f}L",  "vs median if optimised")

# ── What-if scenarios ─────────────────────────────────────────────────────────
st.markdown('<div class="section-title">What-If Scenario Analysis</div>', unsafe_allow_html=True)
st.caption("See how changing one variable at a time affects your budget")

scenarios = []
for g in [50,100,150,200,250,300,400,500,700,1000]:
    r = live_row.copy(); r['Guest_count']=g
    p = max(0,mdl.predict(scaler.transform(pd.DataFrame([r])[feat_used]))[0])
    scenarios.append({'Variable':'Guest Count','Value':g,'Predicted Budget (₹L)':round(p,1)})

for sp in [1,2,3,4,5]:
    r = live_row.copy(); r['Social_pressure']=sp
    p = max(0,mdl.predict(scaler.transform(pd.DataFrame([r])[feat_used]))[0])
    scenarios.append({'Variable':'Social Pressure','Value':sp,'Predicted Budget (₹L)':round(p,1)})

for c in [1,2,3,4,5,6]:
    r = live_row.copy(); r['Ceremony_count']=c
    p = max(0,mdl.predict(scaler.transform(pd.DataFrame([r])[feat_used]))[0])
    scenarios.append({'Variable':'Ceremony Count','Value':c,'Predicted Budget (₹L)':round(p,1)})

scen_df = pd.DataFrame(scenarios)

tab1, tab2, tab3 = st.tabs(["Guest Count Impact","Social Pressure Impact","Ceremony Count Impact"])

with tab1:
    g_data = scen_df[scen_df['Variable']=='Guest Count']
    fig_g  = px.line(g_data, x='Value', y='Predicted Budget (₹L)', markers=True,
                     color_discrete_sequence=['#059669'],
                     labels={'Value':'Guest Count'})
    fig_g.update_traces(line_width=3, marker_size=10)
    fig_g.add_vline(x=guests, line_dash='dash', line_color='#ef4444',
                    annotation_text=f'Your choice: {guests}')
    fig_g.add_hline(y=tier_avg, line_dash='dot', line_color='gray',
                    annotation_text=f'City median ₹{tier_avg:.0f}L')
    fig_g.update_layout(height=320, paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)', margin=dict(t=20,b=10))
    st.plotly_chart(fig_g, use_container_width=True)
    g_base = g_data[g_data['Value']==guests]['Predicted Budget (₹L)'].values[0] if guests in g_data['Value'].values else predicted
    g_half = g_data[g_data['Value']==max(50,guests//2)]['Predicted Budget (₹L)'].values[0]
    st.markdown(f'<div class="insight-box">💡 Reducing your guest list from <b>{guests}</b> to <b>{max(50,guests//2)}</b> could save approximately <b>₹{g_base-g_half:.1f}L</b>. Guest count is the single most controllable budget lever.</div>', unsafe_allow_html=True)

with tab2:
    s_data = scen_df[scen_df['Variable']=='Social Pressure']
    fig_s  = px.bar(s_data, x='Value', y='Predicted Budget (₹L)',
                    color='Predicted Budget (₹L)', color_continuous_scale='RdYlGn_r',
                    labels={'Value':'Social Pressure Score (1=low, 5=high)'},
                    text='Predicted Budget (₹L)')
    fig_s.update_traces(textposition='outside', texttemplate='₹%{text:.1f}L')
    fig_s.add_vline(x=social_p, line_dash='dash', line_color='#ef4444',
                    annotation_text=f'Your score: {social_p}')
    fig_s.update_layout(height=320, paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)', coloraxis_showscale=False,
                        margin=dict(t=20,b=10))
    st.plotly_chart(fig_s, use_container_width=True)
    sp_low  = s_data[s_data['Value']==1]['Predicted Budget (₹L)'].values[0]
    sp_high = s_data[s_data['Value']==5]['Predicted Budget (₹L)'].values[0]
    st.markdown(f'<div class="warn-box">⚠️ Families under maximum social pressure (score 5) spend ₹{sp_high-sp_low:.1f}L MORE than low-pressure families — purely due to emotional decision-making. ShaadiSpend\'s Emotional Spend Alert directly targets this gap.</div>', unsafe_allow_html=True)

with tab3:
    c_data = scen_df[scen_df['Variable']=='Ceremony Count']
    fig_c  = px.area(c_data, x='Value', y='Predicted Budget (₹L)',
                     color_discrete_sequence=['#059669'],
                     labels={'Value':'Number of Ceremonies'},
                     markers=True)
    fig_c.update_traces(line_width=3, marker_size=10, fillcolor='rgba(5,150,105,0.12)')
    fig_c.add_vline(x=ceremonies, line_dash='dash', line_color='#ef4444',
                    annotation_text=f'Your choice: {ceremonies}')
    fig_c.update_layout(height=320, paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)', margin=dict(t=20,b=10))
    st.plotly_chart(fig_c, use_container_width=True)
    c1b = c_data[c_data['Value']==1]['Predicted Budget (₹L)'].values[0]
    c6b = c_data[c_data['Value']==6]['Predicted Budget (₹L)'].values[0]
    st.markdown(f'<div class="insight-box">💡 Each additional ceremony adds an average of ₹{(c6b-c1b)/5:.1f}L to total budget. With {ceremonies} ceremonies planned, consolidating to {max(1,ceremonies-1)} could free up ~₹{(c6b-c1b)/5:.1f}L for higher-priority items.</div>', unsafe_allow_html=True)

# ── Budget allocation breakdown for prediction ─────────────────────────────────
st.markdown('<div class="section-title">Predicted Budget Allocation Breakdown</div>', unsafe_allow_html=True)

dest_mult = 1.15 if dest_flag else 1.0
alloc = {
    'Venue & Catering':   round(predicted * 0.40 * dest_mult, 2),
    'Photography':        round(predicted * 0.12, 2),
    'Decoration':         round(predicted * 0.11, 2),
    'Bridal Wear':        round(predicted * 0.18, 2),
    'Entertainment':      round(predicted * 0.07, 2),
    'Wedding Planner':    round(predicted * (0.08 if dest_flag else 0.02), 2),
    'Invitations & Gifts':round(predicted * 0.04, 2),
    'Miscellaneous':      round(predicted * 0.05, 2),
}
total_alloc = sum(alloc.values())
alloc_norm  = {k: round(v/total_alloc*predicted,2) for k,v in alloc.items()}

alloc_df = pd.DataFrame({'Category':list(alloc_norm.keys()),
                          'Amount (₹L)':list(alloc_norm.values())})

ac1, ac2 = st.columns([1,1])
with ac1:
    fig_alloc = px.pie(alloc_df, names='Category', values='Amount (₹L)', hole=0.45,
                       color_discrete_sequence=['#059669','#34d399','#10b981','#6ee7b7',
                                                '#f59e0b','#fbbf24','#d1fae5','#a7f3d0'])
    fig_alloc.update_traces(textinfo='label+percent')
    fig_alloc.update_layout(height=320, paper_bgcolor='rgba(0,0,0,0)', margin=dict(t=10,b=10))
    st.plotly_chart(fig_alloc, use_container_width=True)

with ac2:
    alloc_df['Per Guest (₹)'] = (alloc_df['Amount (₹L)']*100000/guests).round(0).astype(int)
    alloc_df['% of Budget']   = (alloc_df['Amount (₹L)']/predicted*100).round(1)
    st.dataframe(alloc_df, use_container_width=True, hide_index=True)
    st.markdown(f'<div class="insight-box">💡 Venue & Catering alone costs ≈ ₹{alloc_norm["Venue & Catering"]/guests*100:.0f} per 100 guests. To stay within budget: fix this category first, then allocate remaining ₹{predicted-alloc_norm["Venue & Catering"]:.1f}L across other services.</div>', unsafe_allow_html=True)

# ── Tips ──────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">💡 Personalised Budget Optimisation Tips</div>', unsafe_allow_html=True)

tips = []
if guests > 400:
    tips.append(("🎯 Reduce guest list", f"Cutting from {guests} to {int(guests*0.7)} guests could save ₹{predicted*0.15:.1f}L (est.).", "High impact"))
if social_p >= 4:
    tips.append(("🧘 Manage social pressure", "Enable ShaadiSpend's Emotional Spend Alert — high social pressure families overrun budgets by 2.8x on average.", "High impact"))
if dest_flag:
    tips.append(("📅 Book off-season", "Destination wedding in May-September (off-peak) can reduce venue costs by 20-30%.", "Medium impact"))
if ceremonies >= 5:
    tips.append(("🎉 Combine ceremonies", f"Combining Haldi+Mehendi into one event saves ~₹{predicted*0.08:.1f}L on venue + catering.", "Medium impact"))
if city_tier == "Tier 1":
    tips.append(("🌆 Consider nearby Tier 2 venue", f"Wedding venues in Pune/Jaipur/Chandigarh cost 30-40% less than Mumbai/Delhi equivalents.", "High impact"))
tips.append(("📊 Compare 3+ vendor quotes", "Families who compare 3+ quotes spend 18% less on average than those who trust without comparison.", "Always applicable"))

for icon_title, desc, impact in tips:
    imp_color = "#059669" if "High" in impact else "#d97706"
    st.markdown(f"""
    <div style='background:#fff;border:1px solid #e5e7eb;border-radius:12px;
                padding:14px 18px;margin-bottom:8px;border-left:4px solid {imp_color};'>
        <div style='font-size:14px;font-weight:700;color:#111827;'>{icon_title}
            <span style='background:{"#dcfce7" if "High" in impact else "#fef9c3"};
                         color:{"#166534" if "High" in impact else "#854d0e"};
                         font-size:11px;padding:2px 8px;border-radius:10px;
                         font-weight:600;margin-left:8px;'>{impact}</span>
        </div>
        <div style='font-size:13px;color:#6b7280;margin-top:4px;'>{desc}</div>
    </div>""", unsafe_allow_html=True)
