import streamlit as st, pandas as pd, numpy as np
import plotly.express as px, plotly.graph_objects as go
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_generator import generate_data

st.set_page_config(page_title="Regression | ShaadiSpend", page_icon="📈", layout="wide")
st.markdown("""<style>
[data-testid="stSidebar"]{background:linear-gradient(180deg,#064e3b,#065f46,#047857);}
[data-testid="stSidebar"] *{color:#d1fae5 !important;}
.section-title{font-size:18px;font-weight:700;color:#064e3b;border-bottom:3px solid #059669;padding-bottom:5px;margin:20px 0 12px;display:inline-block;}
.insight-box{background:linear-gradient(135deg,#ecfdf5,#f0fdf4);border-left:4px solid #059669;border-radius:0 10px 10px 0;padding:10px 14px;font-size:13px;color:#064e3b;margin:8px 0;}
.pred-result{background:linear-gradient(135deg,#059669,#047857);color:white;border-radius:14px;padding:22px;text-align:center;}
</style>""", unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def load(): return generate_data(seed=2024)
df = load()

st.markdown("## 📈 Regression — Predict Wedding Budget")
st.caption("Target: Budget_num (₹ Lakhs) | Compare 4 regression models")

feat_cols = ['Income_num','Guest_count','Ceremony_count','Social_pressure','Services_count',
             'WTP_monthly','Emotional_intensity','Feature_score_avg',
             'Alloc_Venue_pct','Alloc_Photo_pct']
avail_f = [c for c in feat_cols if c in df.columns]

with st.sidebar:
    st.markdown("### ⚙️ Model Controls")
    model_name = st.selectbox("Algorithm", ["Linear Regression","Ridge Regression","Lasso Regression","Gradient Boosting","Random Forest"])
    test_size  = st.slider("Test set (%)", 10, 40, 25) / 100
    alpha      = st.slider("Regularization α (Ridge/Lasso)", 0.01, 10.0, 1.0, 0.1)

@st.cache_data
def train_reg(mname, ts, alp, feats):
    data = df[feats+['Budget_num']].dropna()
    data = data[data['Budget_num']<200]
    X = data[feats]; y = data['Budget_num']
    Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=ts,random_state=42)
    scaler = StandardScaler()
    Xtr_s=scaler.fit_transform(Xtr); Xte_s=scaler.transform(Xte)
    models={"Linear Regression":LinearRegression(),"Ridge Regression":Ridge(alpha=alp),
            "Lasso Regression":Lasso(alpha=alp,max_iter=5000),
            "Gradient Boosting":GradientBoostingRegressor(n_estimators=150,max_depth=4,random_state=42),
            "Random Forest":RandomForestRegressor(n_estimators=100,max_depth=8,random_state=42,n_jobs=-1)}
    mdl=models[mname]; mdl.fit(Xtr_s,ytr)
    yp=mdl.predict(Xte_s)
    r2=r2_score(yte,yp); mae=mean_absolute_error(yte,yp); rmse=np.sqrt(mean_squared_error(yte,yp))
    cv=cross_val_score(mdl,Xtr_s,ytr,cv=5,scoring='r2').mean()
    coef=pd.DataFrame({'Feature':feats,'Coefficient':mdl.coef_}).sort_values('Coefficient',key=abs,ascending=False) if hasattr(mdl,'coef_') else None
    fimp=pd.DataFrame({'Feature':feats,'Importance':mdl.feature_importances_}).sort_values('Importance',ascending=False) if hasattr(mdl,'feature_importances_') else None
    return mdl,scaler,r2,mae,rmse,cv,yte.values,yp,coef,fimp,feats

mdl,scaler,r2,mae,rmse,cv,y_act,y_pred,coef,fimp,feat_used = train_reg(model_name,test_size,alpha,avail_f)

m1,m2,m3,m4 = st.columns(4)
m1.metric("R² Score", f"{r2:.4f}", "1.0 = perfect fit")
m2.metric("MAE (₹L)", f"{mae:.2f}", "Mean absolute error")
m3.metric("RMSE (₹L)", f"{rmse:.2f}", "Root mean sq error")
m4.metric("CV R² (5-fold)", f"{cv:.4f}", "Cross-validated")

st.markdown("---")

# Model comparison
st.markdown('<div class="section-title">Model Comparison — All 4 Algorithms</div>', unsafe_allow_html=True)
@st.cache_data
def compare_all(feats,ts,alp):
    results=[]
    for mn in ["Linear Regression","Ridge Regression","Lasso Regression","Gradient Boosting"]:
        _,_,r2_,mae_,rmse_,cv_,*_ = train_reg(mn,ts,alp,feats)
        results.append({'Model':mn,'R²':round(r2_,4),'MAE (₹L)':round(mae_,2),'RMSE (₹L)':round(rmse_,2),'CV R²':round(cv_,4)})
    return pd.DataFrame(results)
comp_df = compare_all(avail_f,test_size,alpha)
fig_comp = px.bar(comp_df,x='Model',y='R²',color='R²',color_continuous_scale='Greens',
                  text='R²',labels={'R²':'R² Score'})
fig_comp.update_traces(textposition='outside')
fig_comp.update_layout(height=300,paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',
                       coloraxis_showscale=False,yaxis=dict(range=[0,1.1]),margin=dict(t=10,b=10))
st.plotly_chart(fig_comp,use_container_width=True)
st.dataframe(comp_df,use_container_width=True,hide_index=True)

c1,c2 = st.columns(2)
with c1:
    st.markdown('<div class="section-title">Actual vs Predicted</div>', unsafe_allow_html=True)
    fig_ap = px.scatter(x=y_act[:400],y=y_pred[:400],opacity=0.55,
                        color_discrete_sequence=['#059669'],
                        labels={'x':'Actual Budget (₹L)','y':'Predicted Budget (₹L)'})
    mn_v=min(y_act.min(),y_pred.min()); mx_v=max(y_act.max(),y_pred.max())
    fig_ap.add_shape(type='line',x0=mn_v,x1=mx_v,y0=mn_v,y1=mx_v,
                     line=dict(color='#ef4444',dash='dash',width=2))
    fig_ap.add_annotation(x=mx_v*0.7,y=mx_v*0.85,text="Perfect prediction line",
                          showarrow=False,font=dict(color='#ef4444',size=11))
    fig_ap.update_layout(height=360,paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',margin=dict(t=10,b=10))
    st.plotly_chart(fig_ap,use_container_width=True)

with c2:
    st.markdown('<div class="section-title">Residual Distribution</div>', unsafe_allow_html=True)
    residuals = y_act - y_pred
    fig_res = px.histogram(x=residuals,nbins=50,color_discrete_sequence=['#7c3aed'],
                           labels={'x':'Residual (Actual − Predicted, ₹L)','count':'Count'})
    fig_res.add_vline(x=0,line_dash='dash',line_color='#ef4444',annotation_text='Zero error')
    fig_res.update_layout(height=360,paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',margin=dict(t=10,b=10))
    st.plotly_chart(fig_res,use_container_width=True)

st.markdown(f'<div class="insight-box">💡 <b>{model_name} — R² = {r2:.4f}</b> | Explains {r2*100:.1f}% of variance in wedding budget. MAE = ₹{mae:.1f}L means predictions are off by ~₹{mae:.1f}L on average. Residuals centred near zero = unbiased model. Use this engine in ShaadiSpend\'s Cost Forecasting feature.</div>', unsafe_allow_html=True)

# Feature importance / coefficients
st.markdown('<div class="section-title">Feature Importance / Coefficients</div>', unsafe_allow_html=True)
if fimp is not None:
    fig_fi=px.bar(fimp,x='Importance',y='Feature',orientation='h',color='Importance',
                  color_continuous_scale='Greens',text=fimp['Importance'].round(3))
    fig_fi.update_traces(textposition='outside')
    fig_fi.update_layout(height=360,paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',
                         coloraxis_showscale=False,yaxis=dict(autorange='reversed'),margin=dict(t=10,b=10,r=60))
    st.plotly_chart(fig_fi,use_container_width=True)
elif coef is not None:
    fig_c=px.bar(coef,x='Coefficient',y='Feature',orientation='h',color='Coefficient',
                 color_continuous_scale='RdYlGn',text=coef['Coefficient'].round(3))
    fig_c.update_traces(textposition='outside')
    fig_c.update_layout(height=360,paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',
                        coloraxis_showscale=False,yaxis=dict(autorange='reversed'),margin=dict(t=10,b=10,r=60))
    st.plotly_chart(fig_c,use_container_width=True)

# Live predictor
st.markdown("---")
st.markdown('<div class="section-title">🔮 Live Budget Predictor</div>', unsafe_allow_html=True)
lp1,lp2,lp3,lp4,lp5 = st.columns(5)
with lp1: l_inc=st.number_input("Income (₹L)",1.0,120.0,25.0,key='r_inc')
with lp2: l_gc=st.number_input("Guest Count",20,2000,300,key='r_gc')
with lp3: l_cc=st.number_input("Ceremonies",1,6,3,key='r_cc')
with lp4: l_sp=st.slider("Social Pressure",1,5,3,key='r_sp')
with lp5: l_sc=st.slider("Services Count",1,14,8,key='r_sc')

live_row={c:float(df[c].median()) for c in feat_used}
for k2,v in {'Income_num':l_inc,'Guest_count':l_gc,'Ceremony_count':l_cc,'Social_pressure':l_sp,'Services_count':l_sc}.items():
    if k2 in live_row: live_row[k2]=v

X_live=pd.DataFrame([live_row])[feat_used]
pred_budget=max(0,mdl.predict(scaler.transform(X_live))[0])
st.markdown(f"""
<div class="pred-result">
  <div style="font-size:14px;opacity:0.85;margin-bottom:6px;">Predicted Wedding Budget</div>
  <div style="font-size:38px;font-weight:700;">₹{pred_budget:.1f} Lakhs</div>
  <div style="font-size:13px;opacity:0.85;margin-top:6px;">Model: {model_name} | R² = {r2:.3f} | Range: ₹{max(0,pred_budget-mae):.1f}L – ₹{pred_budget+mae:.1f}L</div>
</div>""", unsafe_allow_html=True)
