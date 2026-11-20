

import os
import io
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title='EnergyLens', page_icon='⚡', layout='wide', initial_sidebar_state='expanded')
FEATURES = ['Global_active_power','Global_reactive_power','Voltage','Global_intensity','Sub_metering_1','Sub_metering_2','Sub_metering_3']
COLORS = {'Low':'#30A779','Medium':'#E9B65B','High':'#D96569'}

st.markdown('''<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Manrope:wght@500;600;700;800&display=swap');
:root {color-scheme:light}
html,body,[class*="css"],.stApp {font-family:'DM Sans',sans-serif}
.stApp {background:#F6F9F8;color:#19352F}
.block-container {max-width:1450px;padding-top:2rem;padding-bottom:3rem}
[data-testid="stSidebar"] {background:#FFFFFF;border-right:1px solid #E3EBE7}
[data-testid="stSidebar"] * {font-family:'DM Sans',sans-serif}
[data-testid="stSidebar"] .stRadio label {padding:5px 0}
h1,h2,h3 {font-family:'Manrope',sans-serif;letter-spacing:-.6px;color:#19352F}
p,span,label {line-height:1.5}
.hero {padding:32px 36px;border-radius:22px;background:linear-gradient(112deg,#E1F3EA 0%,#F2FAF5 64%,#E7F4EF 100%);border:1px solid #D4E9DD;margin-bottom:25px;box-shadow:0 8px 30px rgba(33,83,65,.05)}
.hero .eyebrow {color:#238A61;font-size:11px;font-weight:800;letter-spacing:2.2px}
.hero h1 {margin:8px 0 8px;font-size:39px;font-weight:800;color:#153C30}
.hero p {color:#557369;margin:0;font-size:15px}
.section {color:#183F32;font-family:'Manrope',sans-serif;font-size:19px;font-weight:800;margin:27px 0 12px}
.note {padding:15px 18px;border-radius:13px;background:#EAF5EF;border:1px solid #D1E8D9;color:#376451;font-size:13px}
[data-testid="stMetric"] {background:#FFFFFF;border:1px solid #E0EAE4;border-radius:17px;padding:18px 20px;box-shadow:0 5px 22px rgba(35,80,57,.045)}
[data-testid="stMetricLabel"] {color:#698178;font-weight:600}
[data-testid="stMetricValue"] {color:#173E31;font-family:'Manrope',sans-serif;font-weight:800}
[data-testid="stMetricDelta"] {color:#238A61}
[data-testid="stDataFrame"],[data-testid="stPlotlyChart"] {border-radius:15px;overflow:hidden}
div[data-testid="stVerticalBlockBorderWrapper"] {border-radius:16px}
.stButton>button,.stDownloadButton>button {background:#217A55;color:white;border:1px solid #217A55;border-radius:10px;font-weight:700}
.stButton>button:hover,.stDownloadButton>button:hover {background:#155D40;color:white;border-color:#155D40}
[data-testid="stFileUploader"] {background:white;border-radius:12px}
hr {border-color:#E3EBE7}
@media(max-width:650px){.hero{padding:24px 20px}.hero h1{font-size:29px}.block-container{padding-top:1rem}}
</style>''',unsafe_allow_html=True)

@st.cache_resource
def load_model(path):
    return joblib.load(path)

@st.cache_data
def load_csv(file):
    return pd.read_csv(file)

def recommend(row):
    tips=[]
    if row['Global_active_power'] > 3: tips.append('Review high electricity consumption')
    if row['Global_intensity'] > 15: tips.append('Check high-current appliances')
    if row['Datetime'].hour < 6 and row['Global_active_power'] > 2: tips.append('Review overnight electricity usage')
    if row['Risk_Level']=='High': tips.append('Prioritize investigation of unusual readings')
    return '; '.join(tips) if tips else 'Continue routine energy monitoring'

def prepare(df, model):
    df=df.copy()
    if 'Datetime' not in df.columns and {'Date','Time'}.issubset(df.columns):
        df['Datetime']=pd.to_datetime(df['Date'].astype(str)+' '+df['Time'].astype(str),dayfirst=True,errors='coerce')
    elif 'Datetime' in df.columns:
        df['Datetime']=pd.to_datetime(df['Datetime'],errors='coerce')
    else:
        raise ValueError('CSV must contain Datetime or both Date and Time columns.')
    missing=[c for c in FEATURES if c not in df.columns]
    if missing: raise ValueError('Missing required columns: '+', '.join(missing))
    for c in FEATURES: df[c]=pd.to_numeric(df[c],errors='coerce')
    df=df.dropna(subset=['Datetime']+FEATURES).sort_values('Datetime')
    if df.empty: raise ValueError('No valid readings after cleaning.')
    # Original training used hourly mean features. Aggregate minute-level files to match.
    gaps=df['Datetime'].diff().dropna().dt.total_seconds()
    is_minute_level=not gaps.empty and gaps.median() < 1800
    if is_minute_level:
        df=df.set_index('Datetime')[FEATURES].resample('h').mean().dropna().reset_index()
    if model is not None:
        X=df[FEATURES]
        labels=model.predict(X)
        scores=-model.decision_function(X)
        df['Anomaly']=np.where(labels==-1,'Anomaly','Normal')
        df['Anomaly_Score']=scores
        df['Risk_Level']='Low'
        flagged=df['Anomaly']=='Anomaly'
        if flagged.any():
            cutoff=df.loc[flagged,'Anomaly_Score'].median()
            df.loc[flagged,'Risk_Level']='Medium'
            df.loc[flagged & (df['Anomaly_Score']>=cutoff),'Risk_Level']='High'
    elif {'Anomaly','Risk_Level'}.issubset(df.columns):
        df['Anomaly']=df['Anomaly'].astype(str)
        df['Risk_Level']=df['Risk_Level'].astype(str)
    else:
        raise ValueError('For an unscored CSV, add energywatch_model.pkl to the app folder. Alternatively upload your pre-scored energywatch_data.csv.')
    df['Recommendation']=df.apply(recommend,axis=1)
    df['Hour']=df['Datetime'].dt.hour




