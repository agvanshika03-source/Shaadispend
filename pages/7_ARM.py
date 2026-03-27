import streamlit as st, pandas as pd, numpy as np
import plotly.express as px, plotly.graph_objects as go
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_generator import generate_data

st.set_page_config(page_title="ARM | ShaadiSpend", page_icon="🛒", layout="wide")
st.markdown("""<style>
[data-testid="stSidebar"]{background:linear-gradient(180deg,#064e3b,#065f46,#047857);}
[data-testid="stSidebar"] *{color:#d1fae5 !important;}
.section-title{font-size:18px;font-weight:700;color:#064e3b;border-bottom:3px solid #059669;padding-bottom:5px;margin:20px 0 12px;display:inline-block;}
.insight-box{background:linear-gradient(135deg,#ecfdf5,#f0fdf4);border-left:4px solid #059669;border-radius:0 10px 10px 0;padding:10px 14px;font-size:13px;color:#064e3b;margin:8px 0;}
.rule-card{background:#fff;border:1px solid #d1fae5;border-radius:10px;padding:14px 16px;margin-bottom:8px;border-left:4px solid #059669;}
</style>""", unsafe_allow_html=True)

try:
    from mlxtend.frequent_patterns import apriori, association_rules
    MLXTEND_OK = True
except Exception:
    MLXTEND_OK = False

@st.cache_data(show_spinner=False)
def load(): return generate_data(seed=2024)
df = load()

st.markdown("## 🛒 Association Rule Mining — Vendor Basket Analysis")
st.caption("Apriori algorithm | Discover which wedding services are purchased together")

with st.sidebar:
    st.markdown("### ⚙️ ARM Controls")
    min_support    = st.slider("Min Support",    0.05, 0.80, 0.20, 0.01)
    min_confidence = st.slider("Min Confidence", 0.10, 0.99, 0.60, 0.01)
    min_lift       = st.slider("Min Lift",       1.0,  5.0,  1.1,  0.1)
    basket_type    = st.radio("Basket type", ["Service adoption","Bundle preferences"])

svc_cols = [c for c in df.columns if c.startswith('Svc_')]
bnd_cols = [c for c in df.columns if c.startswith('Bundle_')]

if basket_type == "Service adoption":
    item_cols = svc_cols
    item_labels = {c: c.replace('Svc_','').replace('_',' ') for c in svc_cols}
else:
    item_cols = bnd_cols
    item_labels = {c: c.replace('Bundle_','').replace('_',' ↔ ') for c in bnd_cols}

basket_df = df[item_cols].rename(columns=item_labels).astype(bool)

# Item frequency chart
st.markdown('<div class="section-title">Item Frequency (Support %)</div>', unsafe_allow_html=True)
item_freq = (basket_df.mean()*100).sort_values(ascending=False)
fig_freq = px.bar(x=item_freq.values, y=item_freq.index, orientation='h',
                  color=item_freq.values, color_continuous_scale='Greens',
                  labels={'x':'Support (%)','y':''},
                  text=[f'{v:.1f}%' for v in item_freq.values])
fig_freq.update_traces(textposition='outside')
fig_freq.update_layout(height=420, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                       coloraxis_showscale=False, yaxis=dict(autorange='reversed'),
                       margin=dict(t=10,b=10,l=10,r=70))
st.plotly_chart(fig_freq, use_container_width=True)

if not MLXTEND_OK:
    st.warning("⚠️ mlxtend library not available on this environment. Showing manual co-occurrence analysis instead.")
    st.markdown('<div class="section-title">Manual Co-occurrence Analysis</div>', unsafe_allow_html=True)
    cooc = basket_df.T.dot(basket_df.astype(int))
    np.fill_diagonal(cooc.values, 0)
    cooc_pct = (cooc / len(df) * 100).round(1)
    fig_co = px.imshow(cooc_pct, color_continuous_scale='Greens',
                       labels={'color':'Co-occurrence %'}, text_auto='.0f')
    fig_co.update_layout(height=500, paper_bgcolor='rgba(0,0,0,0)', margin=dict(t=10,b=10))
    st.plotly_chart(fig_co, use_container_width=True)
    st.markdown('<div class="insight-box">💡 Co-occurrence matrix shows how often any two services are bought together. Darker green = stronger basket association. Use for bundle recommendation engine.</div>', unsafe_allow_html=True)
    st.stop()

# Run Apriori
st.markdown('<div class="section-title">Association Rules</div>', unsafe_allow_html=True)

@st.cache_data
def run_apriori(bdf_vals, bdf_cols, min_sup, min_conf, min_lft):
    bdf = pd.DataFrame(bdf_vals, columns=bdf_cols)
    try:
        freq = apriori(bdf, min_support=min_sup, use_colnames=True)
        if len(freq) == 0:
            return None, 0
        rules = association_rules(freq, metric='confidence', min_threshold=min_conf)
        rules = rules[rules['lift'] >= min_lft].copy()
        rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(sorted(x)))
        rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(sorted(x)))
        return rules.sort_values('lift',ascending=False).reset_index(drop=True), len(freq)
    except Exception as e:
        return None, 0

rules_df, n_freq = run_apriori(basket_df.values, list(basket_df.columns), min_support, min_confidence, min_lift)

if rules_df is not None and len(rules_df) > 0:
    m1,m2,m3 = st.columns(3)
    m1.metric("Frequent Itemsets", n_freq)
    m2.metric("Rules Generated", len(rules_df))
    m3.metric("Max Lift", f"{rules_df['lift'].max():.3f}")

    disp_cols = ['antecedents','consequents','support','confidence','lift']
    disp = rules_df[disp_cols].head(20).copy()
    disp.columns = ['If Customer Wants →','→ They Also Want','Support','Confidence','Lift']
    disp['Support']    = disp['Support'].round(3)
    disp['Confidence'] = disp['Confidence'].round(3)
    disp['Lift']       = disp['Lift'].round(3)
    st.dataframe(disp, use_container_width=True, hide_index=True)

    c1,c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-title">Support vs Confidence (Bubble = Lift)</div>', unsafe_allow_html=True)
        fig_sc = px.scatter(rules_df.head(40), x='support', y='confidence', size='lift',
                            color='lift', color_continuous_scale='Greens',
                            hover_data=['antecedents','consequents'],
                            labels={'support':'Support','confidence':'Confidence','lift':'Lift'})
        fig_sc.update_layout(height=360, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(t=10,b=10))
        st.plotly_chart(fig_sc, use_container_width=True)

    with c2:
        st.markdown('<div class="section-title">Top Rules by Lift</div>', unsafe_allow_html=True)
        top10 = rules_df.head(10)
        fig_lift = px.bar(top10, x='lift', y=top10.index.astype(str),
                          orientation='h', color='lift', color_continuous_scale='Greens',
                          labels={'x':'Lift','y':'Rule Index'},
                          hover_data=['antecedents','consequents'],
                          text=top10['lift'].round(2))
        fig_lift.update_traces(textposition='outside')
        fig_lift.update_layout(height=360, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                               coloraxis_showscale=False, yaxis=dict(autorange='reversed'), margin=dict(t=10,b=10,r=60))
        st.plotly_chart(fig_lift, use_container_width=True)

    # Lift heatmap
    st.markdown('<div class="section-title">Lift Heatmap (Top Rules)</div>', unsafe_allow_html=True)
    top15 = rules_df.head(15)
    try:
        pivot = top15.pivot_table(index='antecedents', columns='consequents', values='lift', aggfunc='mean').fillna(0)
        if not pivot.empty:
            fig_hm = px.imshow(pivot, color_continuous_scale='Greens',
                               labels={'color':'Lift'}, text_auto='.2f', aspect='auto')
            fig_hm.update_layout(height=400, paper_bgcolor='rgba(0,0,0,0)', margin=dict(t=10,b=10))
            st.plotly_chart(fig_hm, use_container_width=True)
    except Exception:
        pass

    r0 = rules_df.iloc[0]
    st.markdown(f'<div class="insight-box">💡 <b>Top Rule:</b> [{r0["antecedents"]}] → [{r0["consequents"]}] | <b>Lift = {r0["lift"]:.2f}</b><br>Customers buying [{r0["antecedents"]}] are {r0["lift"]:.2f}x more likely to also want [{r0["consequents"]}].<br><b>Business action:</b> Bundle these in ShaadiSpend\'s "Customers like you also booked..." recommendation engine.</div>', unsafe_allow_html=True)

    st.download_button("📥 Download Rules (CSV)", rules_df.to_csv(index=False).encode(), "ShaadiSpend_ARM_Rules.csv", "text/csv")

elif rules_df is not None:
    st.warning("No rules found with current thresholds. Try lowering Min Support, Confidence, or Lift in the sidebar.")
    # Fallback: co-occurrence
    st.markdown('<div class="section-title">Co-occurrence Matrix (Fallback)</div>', unsafe_allow_html=True)
    cooc = basket_df.T.dot(basket_df.astype(int))
    np.fill_diagonal(cooc.values, 0)
    fig_co2 = px.imshow((cooc/len(df)*100).round(1), color_continuous_scale='Greens',
                        labels={'color':'Co-occur %'}, text_auto='.0f')
    fig_co2.update_layout(height=480, paper_bgcolor='rgba(0,0,0,0)', margin=dict(t=10,b=10))
    st.plotly_chart(fig_co2, use_container_width=True)
else:
    st.error("Not enough frequent itemsets. Lower Min Support to 0.10 or below.")

st.markdown("---")
st.markdown('<div class="insight-box"><b>How to read ARM results:</b><br>• <b>Support</b> — % of weddings containing this combo. Higher = more common.<br>• <b>Confidence</b> — P(Y|X). "How often do buyers of X also buy Y?"<br>• <b>Lift > 1</b> = positive association. <b>Lift > 1.5</b> = strong. Use high-lift rules for bundle deals.</div>', unsafe_allow_html=True)
