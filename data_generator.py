"""
data_generator.py
Generates the ShaadiSpend synthetic dataset entirely in memory.
No CSV files needed — works on Streamlit Cloud without any data folder.
"""

import numpy as np
import pandas as pd

def generate_data(seed=2024):
    np.random.seed(seed)
    N = 2000

    def wc(opts, wts, size=N):
        w = np.array(wts, dtype=float)
        return np.random.choice(opts, size=size, p=w/w.sum())

    def noise(arr, pct=0.15):
        return np.clip(arr * np.random.normal(1, pct, len(arr)), 0, None)

    # ── Demographics ──────────────────────────────────────────────────────────
    age_group   = wc(['18-24','25-30','31-36','37-45','45+'],       [10,38,30,15,7])
    gender      = wc(['Female','Male','Non-binary/Other'],           [54,44,2])
    city_tier   = wc(['Tier 1','Tier 2','Tier 3/Rural'],            [35,42,23])
    state       = wc(['Maharashtra','Delhi NCR','Karnataka','Tamil Nadu','Uttar Pradesh',
                       'West Bengal','Rajasthan','Gujarat','Punjab','Bihar','Others'],
                      [14,13,10,9,9,7,7,7,6,5,13])
    income_band = wc(['<5L','5-10L','10-20L','20-40L','40-75L','>75L'], [12,18,28,24,12,6])
    income_map  = {'<5L':3,'5-10L':7.5,'10-20L':15,'20-40L':30,'40-75L':57.5,'>75L':100}
    income_num  = np.array([income_map[x] for x in income_band]) * noise(np.ones(N), 0.08)
    income_num  = np.clip(income_num, 1, 120)

    education   = wc(['Up to 12th','Undergraduate','Postgraduate','Professional','Doctoral'],
                      [15,33,32,16,4])
    occupation  = wc(['Salaried private','Salaried govt','Self-employed/Business',
                       'Freelancer','Student','Homemaker','Other'], [35,15,22,8,10,7,3])
    plan_status = wc(['Planning now (0-12m)','Engaged (1-2y)','Considering (2-4y)',
                       'Parent planning child','Recently married (≤3y)'], [22,25,18,20,15])
    decision_auth = wc(['Couple jointly','Bride primarily','Parents primarily',
                         'Father primarily','Mixed family committee'], [28,20,25,15,12])
    tech_adopt  = wc(['Yes, actively use finance apps','Occasionally','No, prefer manual'],
                      [38,35,27])
    info_source = wc(['WhatsApp family groups','Instagram/YouTube','Wedding apps',
                       'Relatives & friends','Hired planner','Mixed sources'], [20,22,15,25,8,10])

    # ── Wedding planning ──────────────────────────────────────────────────────
    wedding_type = wc(['Traditional local','Destination (within India)',
                        'Destination (international)','Intimate (<100 guests)','Grand (500+ guests)'],
                       [38,22,8,18,14])
    ceremony_cnt = np.where(income_num < 10,
                       np.random.choice([1,2,3], N, p=[0.50,0.35,0.15]),
                   np.where(income_num < 40,
                       np.random.choice([2,3,4,5], N, p=[0.20,0.35,0.30,0.15]),
                       np.random.choice([3,4,5,6], N, p=[0.10,0.25,0.35,0.30])))

    city_mult = np.where(city_tier=='Tier 1',1.30,np.where(city_tier=='Tier 2',1.0,0.72))
    type_mult = np.where(wedding_type=='Destination (international)',1.70,
                np.where(wedding_type=='Destination (within India)',1.35,
                np.where(wedding_type=='Grand (500+ guests)',1.25,
                np.where(wedding_type=='Intimate (<100 guests)',0.65,1.0))))
    budget_num = np.clip(noise(income_num * np.random.uniform(1.8,4.8,N) * city_mult * type_mult, 0.22), 1.5, 200)

    def budget_band(b):
        if b<5:   return '<5L'
        if b<10:  return '5-10L'
        if b<20:  return '10-20L'
        if b<40:  return '20-40L'
        if b<75:  return '40-75L'
        if b<100: return '75L-1Cr'
        return '>1Cr'
    budget_label = np.array([budget_band(b) for b in budget_num])

    guest_base = np.clip(budget_num * np.random.uniform(5,15,N), 20, 2000)
    guest_base = np.where(wedding_type=='Intimate (<100 guests)', np.random.uniform(20,100,N), guest_base)
    guest_base = np.where(wedding_type=='Grand (500+ guests)',    np.random.uniform(450,1500,N), guest_base)
    guest_num  = np.clip(np.round(noise(guest_base,0.15)).astype(int), 5, 3000)

    def guest_band(g):
        if g<50:   return '<50'
        if g<150:  return '50-150'
        if g<300:  return '150-300'
        if g<500:  return '300-500'
        if g<1000: return '500-1000'
        return '1000+'
    guest_label = np.array([guest_band(g) for g in guest_num])
    plan_horizon = wc(['<3 months','3-6 months','6-12 months','>12 months'], [12,28,38,22])

    # ── Services (ARM basket) ─────────────────────────────────────────────────
    svc_names = ['Venue','Catering','Photography','Videography','Decoration_florals',
                 'Bridal_makeup','Wedding_planner','DJ_music','Invitation_design',
                 'Mehendi_artist','Bridal_wear','Jewellery_rental','Horse_baraat','Guest_accommodation']
    svc_probs  = [0.92,0.91,0.88,0.72,0.86,0.81,0.30,0.66,0.70,0.75,0.83,0.40,0.35,0.46]
    svc_rows = []
    for i in range(N):
        b = budget_num[i]; mult = min(1.0, 0.55+b/130)
        row = {}
        for s,p in zip(svc_names,svc_probs):
            base = p*mult
            if s=='Wedding_planner' and 'Destination' in wedding_type[i]: base=min(0.88,base+0.35)
            row[f'Svc_{s}'] = int(np.random.rand()<base)
        svc_rows.append(row)
    svc_df = pd.DataFrame(svc_rows)

    # ── Budget allocation ─────────────────────────────────────────────────────
    alloc_rows = []
    for i in range(N):
        hp = svc_df.iloc[i]['Svc_Wedding_planner']
        vc=max(5,np.random.normal(40,5)); ph=max(2,np.random.normal(12,3))
        dec=max(2,np.random.normal(11,3)); bw=max(2,np.random.normal(18,4))
        ent=max(1,np.random.normal(7,2)); wp=max(0,np.random.normal(8 if hp else 1.5,1.5))
        inv=max(0.5,np.random.normal(4,1.5)); misc=max(1,np.random.normal(5,2))
        tot=vc+ph+dec+bw+ent+wp+inv+misc
        alloc_rows.append({'Alloc_Venue_Catering_pct':round(vc/tot*100,1),
                           'Alloc_Photography_pct':round(ph/tot*100,1),
                           'Alloc_Decoration_pct':round(dec/tot*100,1),
                           'Alloc_Bridal_wear_pct':round(bw/tot*100,1),
                           'Alloc_Entertainment_pct':round(ent/tot*100,1),
                           'Alloc_Planner_pct':round(wp/tot*100,1),
                           'Alloc_Invitations_pct':round(inv/tot*100,1),
                           'Alloc_Misc_pct':round(misc/tot*100,1)})
    alloc_df = pd.DataFrame(alloc_rows)

    price_assess = wc(['Compared 3+ quotes','Family/friend referral','Checked online reviews',
                        'Used wedding app','Hired planner to negotiate','Trusted without comparison'],
                       [22,28,18,12,10,10])
    overcharge   = wc(['Yes, significantly overcharged','Yes, slightly overcharged',
                        'No, pricing was fair','No, got a great deal','Unable to assess'],
                       [18,32,28,8,14])

    # ── Emotional influences ──────────────────────────────────────────────────
    sp_base = np.where(city_tier=='Tier 1',3.6,np.where(city_tier=='Tier 2',3.1,2.6))
    social_pressure = np.clip(np.round(sp_base+np.random.normal(0,0.85,N)),1,5).astype(int)

    emo_triggers = ['Emo_Fear_social_judgment','Emo_Impressing_guests','Emo_Parental_expectations',
                    'Emo_Sibling_comparison','Emo_Social_media_influence','Emo_Cultural_obligation',
                    'Emo_Once_in_lifetime_thinking','Emo_None_fully_rational']
    emo_rows = []
    for i in range(N):
        sp=social_pressure[i]
        probs=np.array([0.05+sp*0.08,0.08+sp*0.06,0.15+sp*0.05,0.05+sp*0.04,
                        0.10+sp*0.06,0.20,0.12+sp*0.05,max(0.02,0.40-sp*0.07)])
        probs=probs/probs.sum()
        sel=np.random.choice(len(emo_triggers),size=min(3,max(1,int(sp))),replace=False,p=probs)
        emo_rows.append({t:int(k in sel) for k,t in enumerate(emo_triggers)})
    emo_df = pd.DataFrame(emo_rows)

    overrun_prob = np.clip(0.10+social_pressure*0.09+(budget_num>50)*0.06,0,0.92)
    overrun_opts = ['Within budget','Exceeded 0-10%','Exceeded 10-25%','Exceeded 25-50%','Exceeded >50%','Still planning']
    overrun_wts  = np.column_stack([
        np.clip(0.30-overrun_prob*0.30,0.05,0.50), np.clip(0.25-overrun_prob*0.05,0.08,0.35),
        np.clip(0.20+overrun_prob*0.10,0.10,0.40), np.clip(0.10+overrun_prob*0.08,0.04,0.28),
        np.clip(0.05+overrun_prob*0.05,0.02,0.15), np.full(N,0.10)])
    overrun_wts = overrun_wts/overrun_wts.sum(axis=1,keepdims=True)
    overrun = np.array([np.random.choice(overrun_opts,p=overrun_wts[i]) for i in range(N)])

    overrun_cat = wc(['Venue/catering upgrade','Guest list expansion','Last-min décor upgrade',
                       'Bridal outfit/jewellery','Photography upgrade','Family pressure',
                       'Hidden vendor fees','No overrun'], [18,20,12,15,10,12,8,5])
    overrun_cat = np.where(overrun=='Within budget','No overrun',overrun_cat)

    alert_recep = np.clip(np.round(3.0+(tech_adopt=='Yes, actively use finance apps').astype(float)*0.9
                          -(social_pressure-3)*0.2+np.random.normal(0,0.75,N)),1,5).astype(int)
    fin_concern = wc(['No concern','Minor debt (<6 months)','Significant loan (1-3 years)',
                       'Severe financial strain','Prefer not to say'], [28,32,22,10,8])

    # ── Vendor & bundles ──────────────────────────────────────────────────────
    vp_opts = ['Price/affordability','Portfolio quality','Online reviews','Word of mouth',
               'Package inclusions','Brand reputation','Responsiveness','Experience']
    vp_rows = []
    for _ in range(N):
        picks=np.random.choice(vp_opts,3,replace=False)
        vp_rows.append({'Vendor_prio_1':picks[0],'Vendor_prio_2':picks[1],'Vendor_prio_3':picks[2]})
    vp_df = pd.DataFrame(vp_rows)

    bundle_names=['Bundle_Venue_Catering','Bundle_Photo_Video','Bundle_Decor_Mehendi',
                  'Bundle_Makeup_Bridal','Bundle_DJ_Band','Bundle_Planner_AllVendors',
                  'Bundle_Venue_Decor','Bundle_Photo_Makeup']
    bnd_probs=[0.73,0.66,0.51,0.56,0.39,0.29,0.61,0.43]
    bnd_rows=[]
    for i in range(N):
        b=budget_num[i]
        bnd_rows.append({bk:int(np.random.rand()<min(1.0,p+(b/220)*0.12))
                         for bk,p in zip(bundle_names,bnd_probs)})
    bnd_df = pd.DataFrame(bnd_rows)

    sat_rows=[]
    for i in range(N):
        sp=social_pressure[i]
        oc=1.5 if 'significantly' in overcharge[i] else(0.7 if 'slightly' in overcharge[i] else 0)
        base=7.5-sp*0.28-oc
        sat_rows.append({'Sat_venue':round(np.clip(np.random.normal(base,1.2),1,10),1),
                         'Sat_catering':round(np.clip(np.random.normal(base+0.3,1.0),1,10),1),
                         'Sat_photography':round(np.clip(np.random.normal(base+0.2,1.1),1,10),1),
                         'Sat_decoration':round(np.clip(np.random.normal(base-0.1,1.3),1,10),1),
                         'Sat_overall':round(np.clip(np.random.normal(base+0.1,0.9),1,10),1)})
    sat_df = pd.DataFrame(sat_rows)

    # ── Platform & pipeline ───────────────────────────────────────────────────
    feat_names=['FeatRating_PriceBenchmark','FeatRating_BudgetOptimiser',
                'FeatRating_OverspendAlert','FeatRating_CostForecast',
                'FeatRating_BundleDeals','FeatRating_PersonaProfile']
    feat_rows=[]
    for i in range(N):
        b=budget_num[i]; sp=social_pressure[i]
        base=3.1+(b/220)*0.9-(sp>3)*0.2
        feat_rows.append({f:float(np.clip(np.round(np.random.normal(base+np.random.uniform(-0.3,0.3),0.9)),1,5))
                          for f in feat_names})
    feat_df = pd.DataFrame(feat_rows)

    wtp=wc(['Free only','<₹199/month','₹200-499/month','₹500-999/month','₹1000+/month','One-time ₹999'],
           [20,22,25,18,8,7])
    wtp_map={'Free only':0,'<₹199/month':99,'₹200-499/month':349,'₹500-999/month':749,
             '₹1000+/month':1200,'One-time ₹999':999}
    wtp_num=np.array([wtp_map[x] for x in wtp])

    nss=np.clip(np.round(sat_df['Sat_overall'].values-(social_pressure-1)*0.4
                          +np.random.normal(0,0.5,N),1),0,10)

    feat_score=feat_df.mean(axis=1).values
    yes_p=np.clip(0.08+feat_score*0.13+(tech_adopt=='Yes, actively use finance apps').astype(float)*0.16
                  +(income_num>20).astype(float)*0.07-(social_pressure>4).astype(float)*0.06,0.06,0.93)
    no_p=np.clip(0.32-feat_score*0.06-(income_num>10).astype(float)*0.05,0.03,0.60)
    may_p=np.clip(1-yes_p-no_p,0.05,0.60)
    tot_p=yes_p+no_p+may_p; yes_p/=tot_p; no_p/=tot_p; may_p/=tot_p
    platform_intent=np.array([np.random.choice(['Yes','Maybe','No'],
                               p=[yes_p[i],may_p[i],no_p[i]]) for i in range(N)])

    pipeline_stage=np.where(platform_intent=='Yes',
                    np.random.choice(['Awareness','Consideration','Trial','Conversion','Retention'],
                                     N,p=[0.15,0.25,0.28,0.20,0.12]),
                    np.where(platform_intent=='Maybe',
                    np.random.choice(['Awareness','Consideration','Trial'],N,p=[0.40,0.40,0.20]),
                    np.random.choice(['Awareness','Churned'],N,p=[0.70,0.30])))

    ltv_monthly=wtp_num*12
    ltv_est=np.where(platform_intent=='Yes',ltv_monthly*np.random.uniform(1.5,3,N),
             np.where(platform_intent=='Maybe',ltv_monthly*np.random.uniform(0.3,1,N),0))
    ltv_est=np.round(ltv_est).astype(int)

    # ── Assemble ──────────────────────────────────────────────────────────────
    df = pd.DataFrame({
        'Respondent_ID':             [f'SS{str(i+1).zfill(4)}' for i in range(N)],
        'Q1_Age_group':              age_group,
        'Q1a_Gender':                gender,
        'Q2_City_tier':              city_tier,
        'Q2a_State':                 state,
        'Q3_Income_band':            income_band,
        'Q3_Income_num_lakhs':       np.round(income_num,2),
        'Q4_Planning_status':        plan_status,
        'Q5_Education':              education,
        'Q5a_Occupation':            occupation,
        'Q_Decision_authority':      decision_auth,
        'Q_Tech_adoption':           tech_adopt,
        'Q_Info_source':             info_source,
        'Q6_Wedding_type':           wedding_type,
        'Q7_Ceremony_count':         ceremony_cnt,
        'Q8_Guest_count_est':        guest_num,
        'Q8_Guest_count_band':       guest_label,
        'Q10_Planning_horizon':      plan_horizon,
        'Q11_Budget_num_lakhs':      np.round(budget_num,2),
        'Q11_Budget_band':           budget_label,
        'Q14_Budget_overrun':        overrun,
        'Q15_Overrun_category':      overrun_cat,
        'Q16_Price_assessment':      price_assess,
        'Q17_Social_pressure_1to5':  social_pressure,
        'Q19_Alert_recep_1to5':      alert_recep,
        'Q20_Financial_concern':     fin_concern,
        'Q22_Overcharge_perception': overcharge,
        'Q22a_Net_satisfaction_0to10': np.round(nss,1),
        'Q22b_NPS_proxy':            np.clip(np.round((nss-5)*15+np.random.normal(0,8,N)),-100,100).astype(int),
        'Q24_WTP_band':              wtp,
        'Q24_WTP_monthly_INR':       wtp_num,
        'Q25_Platform_intent':       platform_intent,
        'Q26_Pipeline_stage':        pipeline_stage,
        'Q27_LTV_estimate_INR':      ltv_est,
    })
    df = pd.concat([df, svc_df, alloc_df, emo_df, sat_df, vp_df, bnd_df, feat_df], axis=1)

    # ── Engineered features ───────────────────────────────────────────────────
    df['FE_Budget_per_guest_INR']  = np.round((df['Q11_Budget_num_lakhs']*100000)/df['Q8_Guest_count_est'].clip(1),0).astype(int)
    df['FE_Services_count']        = df[[c for c in df.columns if c.startswith('Svc_')]].sum(axis=1)
    df['FE_Emotional_intensity']   = df[[c for c in df.columns if c.startswith('Emo_') and 'rational' not in c]].sum(axis=1)
    df['FE_Feature_score_avg']     = df[[c for c in df.columns if c.startswith('FeatRating_')]].mean(axis=1).round(2)
    df['FE_Overrun_flag']          = (df['Q14_Budget_overrun']!='Within budget').astype(int)
    df['FE_Income_budget_ratio']   = np.round(df['Q11_Budget_num_lakhs']/df['Q3_Income_num_lakhs'].clip(0.1),3)
    df['FE_Destination_flag']      = df['Q6_Wedding_type'].str.contains('Destination').astype(int)
    df['FE_High_social_pressure']  = (df['Q17_Social_pressure_1to5']>=4).astype(int)
    df['FE_Customer_tier']         = np.where(df['Q27_LTV_estimate_INR']==0,'Non-customer',
                                      np.where(df['Q27_LTV_estimate_INR']<5000,'Low-value',
                                      np.where(df['Q27_LTV_estimate_INR']<20000,'Mid-value','High-value')))

    # ── Ordinal encodings ─────────────────────────────────────────────────────
    df['ENC_Q25_Platform_intent'] = df['Q25_Platform_intent'].map({'Yes':2,'Maybe':1,'No':0})
    df['ENC_Q26_Pipeline_stage']  = df['Q26_Pipeline_stage'].map(
        {'Churned':0,'Awareness':1,'Consideration':2,'Trial':3,'Conversion':4,'Retention':5})
    df['ENC_Q2_City_tier']        = df['Q2_City_tier'].map({'Tier 1':3,'Tier 2':2,'Tier 3/Rural':1})

    return df
