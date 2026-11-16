import os
import io
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title='EnergyWatch AI', page_icon='⚡', layout='wide', initial_sidebar_state='expanded')
FEATURES = ['Global_active_power','Global_reactive_power','Voltage','Global_intensity','Sub_metering_1','Sub_metering_2','Sub_metering_3']
COLORS = {'Low':'#24c7a4','Medium':'#f6b85b','High':'#ff647c'}

st.markdown('''<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
html,body,[class*="css"],.stApp {font-family:'DM Sans',sans-serif;}
.stApp{background:#0b1120;color:#eaf1ff}
[data-testid="stSidebar"]{background:#101b2c;border-right:1px solid #24344b}
[data-testid="stMetric"]{background:#142238;border:1px solid #273b55;border-radius:16px;padding:18px 20px}
[data-testid="stMetricLabel"]{color:#9aacc6}
[data-testid="stMetricValue"]{color:#f3f7ff;font-family:'Space Grotesk',sans-serif}
h1,h2,h3{font-family:'Space Grotesk',sans-serif;letter-spacing:-.5px}
.hero{padding:22px 26px;border-radius:18px;background:linear-gradient(110deg,#152f43,#17213b 70%,#172842);border:1px solid #28485a;margin-bottom:20px}
.hero .eyebrow{color:#58e4d1;font-size:12px;font-weight:700;letter-spacing:2px}
.hero h1{margin:5px 0 7px;font-size:35px;color:#f4faff}
.hero p{color:#b0c4d9;margin:0}
.section{color:#e8f3ff;font-size:18px;font-weight:700;margin:22px 0 10px}
.note{padding:12px 16px;border-radius:10px;background:#142238;border:1px solid #28415a;color:#b7c8db;font-size:13px}
div[data-testid="stDataFrame"]{border-radius:12px;overflow:hidden}
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
    df['Day']=df['Datetime'].dt.date
    return df,is_minute_level

def chart_style(fig,height=350):
    fig.update_layout(template='plotly_dark',paper_bgcolor='#142238',plot_bgcolor='#142238',font=dict(color='#b7c9dd',family='DM Sans'),margin=dict(l=18,r=18,t=35,b=15),height=height,legend=dict(orientation='h',y=-.25),hovermode='x unified')
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor='#293b52',zeroline=False)
    return fig

with st.sidebar:
    st.markdown('## ⚡ EnergyWatch AI')
    st.caption('SMART ENERGY INTELLIGENCE')
    st.divider()
    page=st.radio('WORKSPACE',['Overview','Anomaly Intelligence','Recommendations','Data Explorer'],label_visibility='visible')
    st.divider()
    uploaded=st.file_uploader('Upload electricity CSV',type='csv',help='Use your energywatch_data.csv or a new file with the seven electrical features and timestamps.')
    st.caption('Default data: energywatch_data.csv in the application folder.')
    st.divider()
    st.caption('Isolation Forest · Statistical anomaly detection')

model=None
if os.path.exists('energywatch_model.pkl'):
    try: model=load_model('energywatch_model.pkl')
    except Exception as exc: st.sidebar.error(f'Model could not load: {exc}')

source=uploaded if uploaded is not None else ('energywatch_data.csv' if os.path.exists('energywatch_data.csv') else None)
if source is None:
    st.markdown('<div class="hero"><div class="eyebrow">ENERGY ANALYTICS PLATFORM</div><h1>Monitor smarter. Detect the unusual.</h1><p>Electricity monitoring, statistical anomalies and actionable usage insights in one workspace.</p></div>',unsafe_allow_html=True)
    st.info('To get started, place **energywatch_data.csv** and **energywatch_model.pkl** in this folder, or upload your scored CSV using the sidebar.')
    st.code('streamlit run app.py',language='bash')
    st.stop()

try:
    raw=load_csv(source)
    data,aggregated=prepare(raw,model if uploaded is not None else None)
except Exception as exc:
    st.error(f'Unable to process CSV: {exc}')
    st.stop()

st.markdown('<div class="hero"><div class="eyebrow">ENERGY INTELLIGENCE / LIVE ANALYSIS</div><h1>⚡ EnergyWatch AI</h1><p>Explore electricity patterns, surface unusual readings, and prioritize follow-up actions.</p></div>',unsafe_allow_html=True)
if uploaded is not None and model is None:
    st.warning('No model file found. Uploaded data can only be viewed if it already contains Anomaly and Risk_Level columns.')
if aggregated:
    st.caption('Minute-level readings were aggregated into hourly averages to match model training.')

min_date=data['Datetime'].min().date(); max_date=data['Datetime'].max().date()
if min_date!=max_date:
    start,end=st.slider('Analysis period',min_value=min_date,max_value=max_date,value=(min_date,max_date))
    view=data[data['Datetime'].dt.date.between(start,end)].copy()
else: view=data.copy()
if view.empty:
    st.warning('No readings in the selected period.')
    st.stop()

count=len(view); anomaly_count=int((view['Anomaly']=='Anomaly').sum()); rate=100*anomaly_count/count
st.caption(f"{view['Datetime'].min():%d %b %Y} – {view['Datetime'].max():%d %b %Y}  ·  {count:,} hourly readings")
a,b,c,d=st.columns(4)
a.metric('Avg Active Power',f"{view['Global_active_power'].mean():.2f} kW")
b.metric('Peak Hourly Power',f"{view['Global_active_power'].max():.2f} kW")
c.metric('Anomalies Detected',f'{anomaly_count:,}')
d.metric('Anomaly Rate',f'{rate:.2f}%')

if page=='Overview':
    st.markdown('<div class="section">Consumption & anomaly timeline</div>',unsafe_allow_html=True)
    plot=view.copy()
    if len(plot)>10000: plot=plot.iloc[::max(1,len(plot)//10000)]
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=plot['Datetime'],y=plot['Global_active_power'],name='Average power (kW)',mode='lines',line=dict(color='#39cbb7',width=1.7)))
    bad=plot[plot['Anomaly']=='Anomaly']
    fig.add_trace(go.Scatter(x=bad['Datetime'],y=bad['Global_active_power'],name='Anomaly',mode='markers',marker=dict(color='#ff647c',size=6)))
    fig.update_layout(yaxis_title='Average active power (kW)')
    st.plotly_chart(chart_style(fig,410),use_container_width=True)
    left,right=st.columns(2)
    with left:
        st.markdown('<div class="section">Daily average power</div>',unsafe_allow_html=True)
        daily=view.groupby('Day',as_index=False)['Global_active_power'].mean()
        fig=px.area(daily,x='Day',y='Global_active_power',labels={'Global_active_power':'Average kW','Day':''})
        fig.update_traces(line_color='#47d5c0',fillcolor='rgba(71,213,192,.16)')
        st.plotly_chart(chart_style(fig),use_container_width=True)
    with right:
        st.markdown('<div class="section">Average power by hour</div>',unsafe_allow_html=True)
        hourly=view.groupby('Hour',as_index=False)['Global_active_power'].mean()
        fig=px.bar(hourly,x='Hour',y='Global_active_power',color_discrete_sequence=['#63a6fa'],labels={'Global_active_power':'Average kW','Hour':'Hour of day'})
        st.plotly_chart(chart_style(fig),use_container_width=True)

elif page=='Anomaly Intelligence':
    left,right=st.columns(2)
    with left:
        st.markdown('<div class="section">Risk priority distribution</div>',unsafe_allow_html=True)
        risks=view['Risk_Level'].value_counts().reindex(['Low','Medium','High'],fill_value=0).reset_index()
        risks.columns=['Risk','Count']
        fig=px.pie(risks,names='Risk',values='Count',hole=.67,color='Risk',color_discrete_map=COLORS)
        st.plotly_chart(chart_style(fig),use_container_width=True)
    with right:
        st.markdown('<div class="section">Anomalies by hour of day</div>',unsafe_allow_html=True)
        hourly=view[view['Anomaly']=='Anomaly'].groupby('Hour').size().reindex(range(24),fill_value=0).reset_index(name='Count')
        fig=px.bar(hourly,x='Hour',y='Count',color_discrete_sequence=['#ff7a91'])
        st.plotly_chart(chart_style(fig),use_container_width=True)
    st.markdown('<div class="section">Flagged readings</div>',unsafe_allow_html=True)
    flagged=view[view['Anomaly']=='Anomaly'].copy()
    if 'Anomaly_Score' in flagged:
        flagged=flagged.sort_values('Anomaly_Score',ascending=False)
    st.dataframe(flagged[['Datetime','Global_active_power','Voltage','Global_intensity','Risk_Level']+(['Anomaly_Score'] if 'Anomaly_Score' in flagged else [])],use_container_width=True,hide_index=True)
    st.markdown('<div class="note">Risk levels describe statistical unusualness, not confirmed safety hazards, faulty appliances, or electricity theft. The 5% model contamination parameter sets the approximate flag rate.</div>',unsafe_allow_html=True)

elif page=='Recommendations':
    st.markdown('<div class="section">Prioritized energy observations</div>',unsafe_allow_html=True)
    priority=st.selectbox('Filter by priority',['All','High','Medium','Low'])
    table=view if priority=='All' else view[view['Risk_Level']==priority]
    st.dataframe(table[['Datetime','Risk_Level','Global_active_power','Global_intensity','Recommendation']].sort_values('Datetime',ascending=False),use_container_width=True,hide_index=True)
    st.markdown('<div class="section">Patterns to investigate</div>',unsafe_allow_html=True)
    high_power=int((view['Global_active_power']>3).sum())
    overnight=int(((view['Hour']<6)&(view['Global_active_power']>2)).sum())
    high_current=int((view['Global_intensity']>15).sum())
    p,q,r=st.columns(3)
    p.metric('High power hours (>3 kW)',f'{high_power:,}')
    q.metric('Overnight elevated hours',f'{overnight:,}')
    r.metric('High current hours (>15 A)',f'{high_current:,}')
    st.markdown('<div class="note">Recommendations use illustrative rule thresholds, not device-specific diagnostics. Review thresholds against local usage patterns before operational use.</div>',unsafe_allow_html=True)

else:
    st.markdown('<div class="section">Explore & export readings</div>',unsafe_allow_html=True)
    st.dataframe(view.drop(columns=['Hour','Day']),use_container_width=True,hide_index=True,height=500)
    st.markdown('**Data requirements for new uploads**')
    st.caption('Datetime (or Date + Time), Global_active_power, Global_reactive_power, Voltage, Global_intensity, Sub_metering_1, Sub_metering_2, Sub_metering_3. The model file must be present to score new readings.')

csv=view.drop(columns=['Hour','Day']).to_csv(index=False).encode('utf-8')
st.download_button('⬇ Download analyzed CSV',csv,'energywatch_analyzed_report.csv','text/csv',use_container_width=True)
st.caption('Dataset context: UCI Individual Household Electric Power Consumption (France, 2006–2010). Average active power is in kW; do not interpret it as measured energy in kWh. This app is an exploratory analytics demonstration.')
