import streamlit as st, pandas as pd, numpy as np
import plotly.express as px, plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_generator import generate_data

st.set_page_config(page_title="Classification | ShaadiSpend", page_icon="🎯", layout="wide")
st.markdown("""<style>
[data-testid="stSidebar"]{background:linear-gradient(180deg,#064e3b,#065f46,#047857);}
[data-testid="stSidebar"] *{color:#d1fae5 !important;}
.section-title{font-size:18px;font-weight:700;color:#064e3b;border-bottom:3px solid #059669;padding-bottom:5px;margin:20px 0 12px;display:inline-block;}
.insight-box{background:linear-gradient(135deg,#ecfdf5,#f0fdf4);border-left:4px solid #059669;border-radius:0 10px 10px 0;padding:10px 14px;font-size:13px;color:#064e3b;margin:8px 0;}
.pred-box{border-radius:14px;padding:20px;text-align:center;color:white;margin-top:16px;}
</style>""", unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def load(): return generate_data(seed=2024)
df = load()

st.markdown("## 🎯 Classification — Predict Platform Adoption")
st.caption("Random Forest Classifier | Target: Platform Intent (Yes / Maybe / No)")

feature_cols = ['Income_num','Budget_num','Guest_count','Social_pressure','Alert_receptiveness',
                'WTP_monthly','Feature_score_avg','Emotional_intensity','Services_count',
                'Budget_per_guest','Income_budget_ratio','Ceremony_count','Sat_overall','LTV_est',
                'FR_PriceBenchmark','FR_BudgetOptimiser','FR_OverspendAlert']
avail_f = [c for c in feature_cols if c in df.columns]

with st.sidebar:
    st.markdown("### ⚙️ Model Controls")
    test_size    = st.slider("Test set (%)", 10, 40, 20) / 100
    n_estimators = st.slider("Number of trees", 50, 300, 150, 50)
    max_depth    = st.slider("Max tree depth", 2, 20, 8)

@st.cache_data
def train_clf(ts, ne, md, feats):
    data = df[feats+['Platform_intent']].dropna()
    X = data[feats]; y = data['Platform_intent']
    le = LabelEncoder(); ye = le.fit_transform(y)
    Xtr,Xte,ytr,yte = train_test_split(X,ye,test_size=ts,random_state=42,stratify=ye)
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr); Xte_s = scaler.transform(Xte)
    rf = RandomForestClassifier(n_estimators=ne,max_depth=md,random_state=42,n_jobs=-1)
    rf.fit(Xtr_s,ytr)
    yp = rf.predict(Xte_s); yprob = rf.predict_proba(Xte_s)
    acc = (yp==yte).mean()
    cm = confusion_matrix(yte,yp)
    cr = classification_report(yte,yp,target_names=le.classes_,output_dict=True)
    fi = pd.DataFrame({'Feature':feats,'Importance':rf.feature_importances_}).sort_values('Importance',ascending=False)
    cv = cross_val_score(rf,Xtr_s,ytr,cv=5,scoring='accuracy').mean()
    return rf,le,scaler,acc,cm,cr,fi,cv,Xte_s,yte,yp,yprob,feats

rf,le,scaler,acc,cm,cr,fi,cv,Xte_s,yte,yp,yprob,feat_used = train_clf(test_size,n_estimators,max_depth,avail_f)

m1,m2,m3,m4 = st.columns(4)
m1.metric("Accuracy", f"{acc*100:.2f}%")
m2.metric("CV Accuracy (5-fold)", f"{cv*100:.2f}%", "Cross-validated")
m3.metric("Trees", n_estimators)
m4.metric("Max Depth", max_depth)

st.markdown("---")
c1,c2 = st.columns(2)

with c1:
    st.markdown('<div class="section-title">Confusion Matrix</div>', unsafe_allow_html=True)
    classes = le.classes_
    fig_cm = px.imshow(cm, x=classes, y=classes, text_auto=True,
                       color_continuous_scale='Greens',
                       labels={'x':'Predicted','y':'Actual','color':'Count'})
    fig_cm.update_layout(height=340, paper_bgcolor='rgba(0,0,0,0)', margin=dict(t=20,b=10))
    st.plotly_chart(fig_cm, use_container_width=True)

with c2:
    st.markdown('<div class="section-title">Classification Report</div>', unsafe_allow_html=True)
    cr_rows=[{'Class':cls,'Precision':round(cr[cls]['precision'],3),
              'Recall':round(cr[cls]['recall'],3),'F1-Score':round(cr[cls]['f1-score'],3),
              'Support':int(cr[cls]['support'])} for cls in le.classes_ if cls in cr]
    cr_df=pd.DataFrame(cr_rows)
    st.dataframe(cr_df,use_container_width=True,hide_index=True)
    st.markdown(f'<div class="insight-box">✅ <b>Accuracy: {acc*100:.2f}%</b> | Macro F1: {cr["macro avg"]["f1-score"]:.3f} | Weighted F1: {cr["weighted avg"]["f1-score"]:.3f}<br>Model reliably predicts which customers will adopt the platform — use for targeted acquisition campaigns.</div>', unsafe_allow_html=True)

st.markdown('<div class="section-title">Feature Importance (Top 15)</div>', unsafe_allow_html=True)
fig_fi = px.bar(fi.head(15), x='Importance', y='Feature', orientation='h',
                color='Importance', color_continuous_scale='Greens',
                labels={'Importance':'Importance Score','Feature':'Feature'},
                text=fi.head(15)['Importance'].round(3))
fig_fi.update_traces(textposition='outside')
fig_fi.update_layout(height=440, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                     coloraxis_showscale=False, yaxis=dict(autorange='reversed'), margin=dict(t=10,b=10,r=60))
st.plotly_chart(fig_fi, use_container_width=True)
st.markdown(f'<div class="insight-box">💡 Top predictor: <b>{fi.iloc[0]["Feature"]}</b> (importance={fi.iloc[0]["Importance"]:.3f}). Feature ratings and income/budget dominate — confirms platform value resonates most with financially-aware, higher-budget users. Use these features for lookalike audience targeting.</div>', unsafe_allow_html=True)

st.markdown('<div class="section-title">Prediction Probability by Class</div>', unsafe_allow_html=True)
prob_df = pd.DataFrame(yprob, columns=le.classes_)
prob_df['Actual'] = le.inverse_transform(yte)
fig_prob = px.histogram(prob_df.melt(id_vars='Actual',var_name='Predicted Class',value_name='Probability'),
                        x='Probability',color='Predicted Class',facet_col='Predicted Class',nbins=30,
                        color_discrete_map={'Yes':'#059669','Maybe':'#f59e0b','No':'#ef4444'})
fig_prob.update_layout(height=280,paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',
                       showlegend=False,margin=dict(t=30,b=10))
st.plotly_chart(fig_prob,use_container_width=True)

st.markdown("---")
st.markdown('<div class="section-title">🔮 Live Prediction Tool</div>', unsafe_allow_html=True)
st.caption("Adjust inputs below to predict whether a new customer will adopt ShaadiSpend")

p1,p2,p3,p4 = st.columns(4)
with p1:
    inc = st.number_input("Income (₹L)", 1.0, 120.0, 25.0)
    sp  = st.slider("Social Pressure (1-5)", 1, 5, 3)
with p2:
    bud = st.number_input("Budget (₹L)", 1.0, 200.0, 35.0)
    ar  = st.slider("Alert Receptiveness (1-5)", 1, 5, 4)
with p3:
    gc  = st.number_input("Guest Count", 20, 2000, 300)
    wtp = st.number_input("WTP (₹/mo)", 0, 1200, 349)
with p4:
    fs  = st.slider("Avg Feature Rating (1-5)", 1.0, 5.0, 3.8)
    ei  = st.slider("Emotional Intensity (0-3)", 0, 3, 1)

live_row = {c: float(df[c].median()) for c in feat_used}
overrides = {'Income_num':inc,'Budget_num':bud,'Guest_count':gc,'Social_pressure':sp,
             'Alert_receptiveness':ar,'WTP_monthly':wtp,'Feature_score_avg':fs,'Emotional_intensity':ei}
for k2,v in overrides.items():
    if k2 in live_row: live_row[k2]=v

X_live = pd.DataFrame([live_row])[feat_used]
X_live_s = scaler.transform(X_live)
proba = rf.predict_proba(X_live_s)[0]
pred = le.inverse_transform([np.argmax(proba)])[0]
color_map={'Yes':'#059669','Maybe':'#d97706','No':'#dc2626'}
classes_list=list(le.classes_)
yes_p=proba[classes_list.index('Yes')]*100 if 'Yes' in classes_list else 0
maybe_p=proba[classes_list.index('Maybe')]*100 if 'Maybe' in classes_list else 0
no_p=proba[classes_list.index('No')]*100 if 'No' in classes_list else 0

st.markdown(f"""
<div class="pred-box" style="background:{color_map.get(pred,'#059669')};">
  <div style="font-size:28px;font-weight:700;">Predicted: {pred}</div>
  <div style="font-size:15px;margin-top:8px;opacity:0.9;">
    Yes: {yes_p:.1f}% &nbsp;|&nbsp; Maybe: {maybe_p:.1f}% &nbsp;|&nbsp; No: {no_p:.1f}%
  </div>
</div>""", unsafe_allow_html=True)
