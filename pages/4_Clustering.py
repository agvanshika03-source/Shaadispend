import streamlit as st, pandas as pd, numpy as np
import plotly.express as px, plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_generator import generate_data

st.set_page_config(page_title="Clustering | ShaadiSpend", page_icon="👥", layout="wide")
st.markdown("""<style>
[data-testid="stSidebar"]{background:linear-gradient(180deg,#064e3b,#065f46,#047857);}
[data-testid="stSidebar"] *{color:#d1fae5 !important;}
.section-title{font-size:18px;font-weight:700;color:#064e3b;border-bottom:3px solid #059669;padding-bottom:5px;margin:20px 0 12px;display:inline-block;}
.insight-box{background:linear-gradient(135deg,#ecfdf5,#f0fdf4);border-left:4px solid #059669;border-radius:0 10px 10px 0;padding:10px 14px;font-size:13px;color:#064e3b;margin:8px 0;}
.persona-card{background:#fff;border:2px solid #059669;border-radius:14px;padding:18px;margin-bottom:10px;}
.persona-title{font-size:16px;font-weight:700;color:#064e3b;margin-bottom:6px;}
.persona-stat{font-size:13px;color:#6b7280;margin:3px 0;}
</style>""", unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def load(): return generate_data(seed=2024)
df = load()

st.markdown("## 👥 K-Means Customer Clustering")
st.caption("Unsupervised ML — discover customer personas for targeted marketing & bundling")

feature_options = ['Income_num','Budget_num','Guest_count','Social_pressure','Alert_receptiveness',
                   'Services_count','Emotional_intensity','Feature_score_avg','WTP_monthly','Budget_per_guest']
avail_feats = [f for f in feature_options if f in df.columns]

with st.sidebar:
    st.markdown("### ⚙️ Clustering Controls")
    k = st.slider("Number of clusters (K)", 2, 7, 4)
    selected_feats = st.multiselect("Features", avail_feats,
        default=['Income_num','Budget_num','Guest_count','Social_pressure','Feature_score_avg','WTP_monthly'])
    st.markdown("---")
    st.markdown("**Tip:** Use 4–5 features for best results")

if len(selected_feats) < 2:
    st.warning("Please select at least 2 features from the sidebar.")
    st.stop()

# ── Step 1: Elbow + Silhouette ─────────────────────────────────────────────────
st.markdown('<div class="section-title">Step 1 — Elbow Method & Silhouette Score</div>', unsafe_allow_html=True)

@st.cache_data
def compute_elbow(feats_tuple):
    feats = list(feats_tuple)
    X = df[feats].dropna()
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    inertias, silhouettes = [], []
    for ki in range(2, 9):
        km = KMeans(n_clusters=ki, random_state=42, n_init=10)
        lbl = km.fit_predict(Xs)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(Xs, lbl))
    return inertias, silhouettes

inertias, silhouettes = compute_elbow(tuple(selected_feats))

ce1, ce2 = st.columns(2)
with ce1:
    fig_elbow = go.Figure()
    fig_elbow.add_trace(go.Scatter(x=list(range(2,9)), y=inertias, mode='lines+markers',
        line=dict(color='#059669', width=3), marker=dict(size=10, color='#059669'),
        name='Inertia (WCSS)'))
    fig_elbow.add_vline(x=k, line_dash='dash', line_color='#ef4444',
                        annotation_text=f'Selected K={k}', annotation_font_color='#ef4444')
    fig_elbow.update_layout(title='Elbow Curve — Inertia vs K', xaxis_title='K (clusters)',
        yaxis_title='Inertia (WCSS)', height=320, paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)', margin=dict(t=40,b=10))
    st.plotly_chart(fig_elbow, use_container_width=True)

with ce2:
    fig_sil = go.Figure()
    fig_sil.add_trace(go.Scatter(x=list(range(2,9)), y=silhouettes, mode='lines+markers',
        line=dict(color='#f59e0b', width=3), marker=dict(size=10, color='#f59e0b'),
        name='Silhouette Score'))
    best_k = silhouettes.index(max(silhouettes)) + 2
    fig_sil.add_vline(x=k, line_dash='dash', line_color='#ef4444',
                      annotation_text=f'Selected K={k}', annotation_font_color='#ef4444')
    fig_sil.add_annotation(x=best_k, y=max(silhouettes), text=f"Best K={best_k}",
                           showarrow=True, arrowhead=2, font=dict(color='#059669'))
    fig_sil.update_layout(title='Silhouette Score vs K (Higher = Better)', xaxis_title='K',
        yaxis_title='Silhouette Score', height=320, paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)', margin=dict(t=40,b=10))
    st.plotly_chart(fig_sil, use_container_width=True)

# ── Step 2: Run K-Means ────────────────────────────────────────────────────────
st.markdown(f'<div class="section-title">Step 2 — K={k} Cluster Results</div>', unsafe_allow_html=True)

@st.cache_data
def run_kmeans(feats_tuple, n_clusters):
    feats = list(feats_tuple)
    X = df[feats].dropna()
    idx = X.index
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(Xs)
    pca = PCA(n_components=2, random_state=42)
    X2d = pca.fit_transform(Xs)
    result = df.loc[idx].copy()
    result['Cluster'] = [f'Persona {l+1}' for l in labels]
    result['PCA_1'] = X2d[:,0]
    result['PCA_2'] = X2d[:,1]
    sil = silhouette_score(Xs, labels)
    var_exp = pca.explained_variance_ratio_
    return result, sil, var_exp

clustered, sil_score, var_exp = run_kmeans(tuple(selected_feats), k)

m1, m2, m3 = st.columns(3)
m1.metric("Silhouette Score", f"{sil_score:.3f}", "Closer to 1.0 = better separation")
m2.metric("PCA Variance Explained", f"{sum(var_exp)*100:.1f}%", "2 components")
m3.metric("Largest Cluster", clustered['Cluster'].value_counts().index[0],
          f"{clustered['Cluster'].value_counts().iloc[0]} respondents")

cp1, cp2 = st.columns([2,1])
with cp1:
    colors = px.colors.qualitative.Set2[:k]
    fig_pca = px.scatter(clustered, x='PCA_1', y='PCA_2', color='Cluster',
                         hover_data=['Income_num','Budget_num','City_tier'] if all(c in clustered.columns for c in ['Income_num','Budget_num','City_tier']) else None,
                         color_discrete_sequence=colors,
                         labels={'PCA_1':f'PC1 ({var_exp[0]*100:.1f}% var)','PCA_2':f'PC2 ({var_exp[1]*100:.1f}% var)'})
    fig_pca.update_traces(marker=dict(size=5, opacity=0.7))
    fig_pca.update_layout(height=420, paper_bgcolor='rgba(0,0,0,0)',
                          plot_bgcolor='rgba(0,0,0,0)', margin=dict(t=20,b=10),
                          legend=dict(title='Customer Persona'))
    st.plotly_chart(fig_pca, use_container_width=True)

with cp2:
    vc_c = clustered['Cluster'].value_counts().sort_index()
    fig_pie = px.pie(names=vc_c.index, values=vc_c.values, hole=0.4,
                     color_discrete_sequence=colors)
    fig_pie.update_layout(height=220, paper_bgcolor='rgba(0,0,0,0)', margin=dict(t=10,b=0,l=0,r=0))
    st.plotly_chart(fig_pie, use_container_width=True)
    st.caption("Cluster size distribution")
    for persona, count in vc_c.items():
        st.markdown(f"**{persona}:** {count} ({count/len(clustered)*100:.1f}%)")

# ── Step 3: Cluster profiles ───────────────────────────────────────────────────
st.markdown('<div class="section-title">Step 3 — Cluster Profile Summary</div>', unsafe_allow_html=True)
prof_cols = [c for c in ['Income_num','Budget_num','Guest_count','Social_pressure',
                          'Feature_score_avg','WTP_monthly','LTV_est','Sat_overall'] if c in clustered.columns]
profile = clustered.groupby('Cluster')[prof_cols].mean().round(1)
nice_names = {'Income_num':'Income (₹L)','Budget_num':'Budget (₹L)','Guest_count':'Guests',
              'Social_pressure':'Social Pressure','Feature_score_avg':'Feature Score',
              'WTP_monthly':'WTP (₹/mo)','LTV_est':'LTV (₹)','Sat_overall':'Satisfaction'}
profile.columns = [nice_names.get(c,c) for c in profile.columns]
st.dataframe(profile.style.background_gradient(cmap='Greens', axis=0), use_container_width=True)

# ── Step 4: Intent by cluster ──────────────────────────────────────────────────
st.markdown('<div class="section-title">Step 4 — Platform Intent by Cluster</div>', unsafe_allow_html=True)
intent_cross = pd.crosstab(clustered['Cluster'], clustered['Platform_intent'], normalize='index').round(3)*100
intent_cols = [c for c in ['Yes','Maybe','No'] if c in intent_cross.columns]
fig_cross = px.bar(intent_cross[intent_cols].reset_index(), x='Cluster', y=intent_cols,
                   barmode='stack', color_discrete_map={'Yes':'#059669','Maybe':'#f59e0b','No':'#ef4444'},
                   labels={'value':'%','variable':'Intent'})
fig_cross.update_layout(height=320, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                        margin=dict(t=20,b=10), legend=dict(orientation='h',y=1.1))
st.plotly_chart(fig_cross, use_container_width=True)
st.markdown('<div class="insight-box">💡 <b>Business use:</b> Each cluster = a marketing persona. High-LTV + High-Yes clusters get premium outreach. High-Maybe clusters get free trial nudges. Low-LTV clusters get freemium onboarding. Use cluster labels to personalise every touchpoint.</div>', unsafe_allow_html=True)

# ── Radar chart ────────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">Step 5 — Cluster Radar Comparison</div>', unsafe_allow_html=True)
radar_cols = [c for c in ['Income_num','Budget_num','Social_pressure','Feature_score_avg','WTP_monthly'] if c in clustered.columns]
if radar_cols:
    radar_profile = clustered.groupby('Cluster')[radar_cols].mean()
    radar_norm = (radar_profile - radar_profile.min()) / (radar_profile.max() - radar_profile.min() + 1e-9)
    fig_radar = go.Figure()
    radar_labels = [nice_names.get(c,c) for c in radar_cols]
    for i, (persona, row) in enumerate(radar_norm.iterrows()):
        vals = list(row.values) + [row.values[0]]
        fig_radar.add_trace(go.Scatterpolar(r=vals, theta=radar_labels+[radar_labels[0]],
                                             fill='toself', name=persona,
                                             line=dict(color=colors[i%len(colors)],width=2),
                                             fillcolor=colors[i%len(colors)].replace('rgb','rgba').replace(')',',0.15)') if 'rgb' in colors[i%len(colors)] else colors[i%len(colors)]))
    fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True,range=[0,1])),
                             height=380, paper_bgcolor='rgba(0,0,0,0)', margin=dict(t=20,b=10))
    st.plotly_chart(fig_radar, use_container_width=True)

st.download_button("📥 Download Clustered Dataset (CSV)",
                   clustered.to_csv(index=False).encode(), "ShaadiSpend_Clustered.csv", "text/csv")
