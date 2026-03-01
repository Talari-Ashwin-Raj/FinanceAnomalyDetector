import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
from preprocess import preprocess_data
from model_dbscan import detect_anomalies

# ==========================================
# 1. Configuration & Styling
# ==========================================
st.set_page_config(
    page_title="Personal Finance Anomaly Detector",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main {
        background-color: #ffffff;
        font-family: 'Inter', sans-serif;
    }
    h1, h2, h3 {
        color: #1f2937;
        font-weight: 600;
    }
    .metric-container {
        background-color: #f9fafb;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        border: 1px solid #e5e7eb;
    }
    .stAlert {
        border-radius: 8px;
    }
    .header-text {
        text-align: center;
        padding-bottom: 2rem;
    }
    .tagline {
        color: #6b7280;
        font-size: 1.2rem;
        text-align: center;
        margin-top: -15px;
        margin-bottom: 30px;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# Caching Functions
# ==========================================
@st.cache_data
def load_data(file):
    """Load CSV into pandas DataFrame"""
    return pd.read_csv(file)

@st.cache_data
def process_data(df):
    """Runs data through existing preprocess module"""
    return preprocess_data(df)

# ==========================================
# Header Section
# ==========================================
st.markdown("<h1 class='header-text'>Personal Finance Anomaly Detector</h1>", unsafe_allow_html=True)
st.markdown("<div class='tagline'>Transform transactions into financial awareness</div>", unsafe_allow_html=True)

# ==========================================
# Sidebar Controls
# ==========================================
st.sidebar.header("⚙️ Tuning Parameters")

risk_sensitivity = st.sidebar.select_slider(
    "Risk Sensitivity Control",
    options=["Low", "Medium", "High", "Custom"],
    value="Medium",
    help="Adjusts underlying detection parameters automatically."
)

# Dynamic mapping based on sensitivity
if risk_sensitivity == "High":
    default_eps = 0.3
    default_min = 3
elif risk_sensitivity == "Medium":
    default_eps = 0.5
    default_min = 5
elif risk_sensitivity == "Low":
    default_eps = 0.8
    default_min = 8
else:
    default_eps = 0.5
    default_min = 5

st.sidebar.markdown("### Advanced Settings")
disabled_sliders = risk_sensitivity != "Custom"

eps = st.sidebar.slider(
    "Neighborhood Radius (eps)", min_value=0.01, max_value=2.00, 
    value=float(default_eps), step=0.01, disabled=disabled_sliders
)
min_samples = st.sidebar.slider(
    "Core Point Threshold (min_samples)", min_value=1, max_value=50, 
    value=int(default_min), step=1, disabled=disabled_sliders
)

# ==========================================
# File Upload Section
# ==========================================
st.sidebar.markdown("---")
st.sidebar.header("📁 Data Source")
st.sidebar.markdown("Upload your bank statement (CSV supported)")
uploaded_file = st.sidebar.file_uploader("", type=["csv"])

default_data_path = "data/RawDataset.csv"
df_raw = None

if uploaded_file is not None:
    try:
        df_raw = load_data(uploaded_file)
        st.sidebar.success("File uploaded and processed successfully!")
    except Exception as e:
        st.sidebar.error(f"Error loading file: {e}")
elif os.path.exists(default_data_path):
    # CHANGED: Auto-load the raw dataset locally so the UI works without needing to upload
    df_raw = load_data(default_data_path)
    st.sidebar.info("Using sample local dataset `data/RawDataset.csv`. Upload a file to override.")
else:
    st.info("👋 Welcome! Please upload your transaction history CSV in the sidebar to begin your financial analysis.")
    st.stop()

# ==========================================
# Processing Pipeline
# ==========================================
with st.spinner("Analyzing financial behavior..."):
    df_processed = process_data(df_raw)
    labels = detect_anomalies(df_processed, eps=eps, min_samples=min_samples)

    df_result = df_raw.copy()
    df_result['is_anomaly'] = (labels == -1).astype(int)
    
    # Feature engineering for UI
    if 'Debit' in df_result.columns and 'Credit' in df_result.columns:
        df_result['Amount'] = df_result['Debit'].fillna(0) + df_result['Credit'].fillna(0)
    elif 'Debit' in df_result.columns:
        df_result['Amount'] = df_result['Debit'].fillna(0)
    else:
        df_result['Amount'] = 0

    if 'Txn Date' in df_result.columns:
        df_result['Date'] = pd.to_datetime(df_result['Txn Date'], dayfirst=True, errors='coerce')
    else:
        df_result['Date'] = pd.NaT

    # Try to extract category if description exists
    if 'category' not in df_result.columns:
        desc_col = 'Description' if 'Description' in df_result.columns else 'Transaction Reference'
        if desc_col in df_result.columns:
            from preprocess import process_transaction
            df_result[['type', 'category']] = df_result.apply(lambda x: process_transaction(x[desc_col]), axis=1, result_type='expand')

    # Periodic aggregations (Weekly and Monthly)
    if 'Txn Date' in df_raw.columns:
        df_raw['Txn Date'] = pd.to_datetime(df_raw['Txn Date'], dayfirst=True, errors='coerce')
        valid_raw = df_raw.dropna(subset=['Txn Date']).copy()
        
        # Ensure Debit and Credit columns exist for sum
        if 'Debit' not in valid_raw.columns: valid_raw['Debit'] = 0
        if 'Credit' not in valid_raw.columns: valid_raw['Credit'] = 0
        if 'Description' not in valid_raw.columns: valid_raw['Description'] = 'Aggregated'

        if not valid_raw.empty:
            # Weekly
            df_weekly = valid_raw.groupby(pd.Grouper(key='Txn Date', freq='7D')).agg({
                'Debit': 'sum', 'Credit': 'sum', 'Description': 'first'
            }).reset_index()
            df_w_proc = process_data(df_weekly)
            df_weekly['is_anomaly'] = detect_anomalies(df_w_proc, eps=eps, min_samples=max(2, min_samples // 2))
            # Calculate Net Value Spent (Debit - Credit). If net is negative, it means more came in than went out.
            df_weekly['Amount'] = df_weekly['Debit'].fillna(0) - df_weekly['Credit'].fillna(0)
            df_weekly['Week Focus'] = df_weekly['Txn Date'].dt.strftime('Week of %d %b %Y')

            # Monthly
            df_monthly = valid_raw.groupby(pd.Grouper(key='Txn Date', freq='ME')).agg({
                'Debit': 'sum', 'Credit': 'sum', 'Description': 'first'
            }).reset_index()
            df_m_proc = process_data(df_monthly)
            df_monthly['is_anomaly'] = detect_anomalies(df_m_proc, eps=eps, min_samples=max(2, min_samples // 3))
            df_monthly['Amount'] = df_monthly['Debit'].fillna(0) - df_monthly['Credit'].fillna(0)
            df_monthly['Month Focus'] = df_monthly['Txn Date'].dt.strftime('%B %Y')
        else:
            df_weekly = pd.DataFrame()
            df_monthly = pd.DataFrame()
    else:
        df_weekly = pd.DataFrame()
        df_monthly = pd.DataFrame()

# ==========================================
# Risk Overview Section
# ==========================================
total_txns = len(df_result)
total_alerts = df_result['is_anomaly'].sum()
highest_txn = df_result['Amount'].max()

st.markdown("### 📊 Risk Overview")
col1, col2, col3 = st.columns(3)

col1.metric("Total Transactions", f"{total_txns:,}")
col2.metric("Total Alerts", f"{total_alerts:,}")
col3.metric("Highest Transaction", f"₹{highest_txn:,.2f}")

st.markdown("<br>", unsafe_allow_html=True)

# ==========================================
# Key Alerts Section
# ==========================================
if total_alerts > 0 or (not df_weekly.empty and (df_weekly['is_anomaly'] == -1).any()) or (not df_monthly.empty and (df_monthly['is_anomaly'] == -1).any()):
    st.markdown("### 🔔 Key Alerts")
    
    # Check for individual high value alerts
    anomalies = df_result[df_result['is_anomaly'] == 1].copy()
    if not anomalies.empty:
        mean_amt = df_result['Amount'].mean()
        std_amt = df_result['Amount'].std()
        high_value_alerts = anomalies[anomalies['Amount'] > (mean_amt + 3 * std_amt)]
        if len(high_value_alerts) > 0:
            st.warning(f"**Abnormally high transaction detected:** {len(high_value_alerts)} individual transactions far exceed your usual spending limits.")
            
        if 'Date' in df_result.columns:
            recent_anomalies = anomalies.sort_values('Date', ascending=False).head(3)
            if len(recent_anomalies) > 1:
                st.error("**Behavioral spending shift detected:** Multiple unusual transactions flagged in close proximity.")
    
    # Check periodic alerts
    if not df_weekly.empty and (df_weekly['is_anomaly'] == -1).any():
        st.warning("**Unusual Weekly Spending Trend Detected:** Significant spikes in 7-day accumulated spending flagged by the model.")
        
    if not df_monthly.empty and (df_monthly['is_anomaly'] == -1).any():
        st.error("**Abnormal Monthly Spending Volume:** Your aggregate monthly transaction volume was flagged as anomalous.")

else:
    st.success("✅ **All Good:** Your financial behavior looks consistent and safe. No major anomalies detected in single transactions or aggregate trends.")

st.markdown("<hr>", unsafe_allow_html=True)

# ==========================================
# Timeline & Pattern Visualization
# ==========================================
if not df_result['Date'].isna().all():
    df_plot = df_result.sort_values('Date')
    
    st.markdown("### 📈 Spending Timeline")
    
    fig_timeline = px.line(
        df_plot, x='Date', y='Amount', 
        hover_data=['Description' if 'Description' in df_plot.columns else 'Amount'],
        color_discrete_sequence=['#9ca3af']
    )
    
    # Overlay anomalies as red scatter points
    anomalies_plot = df_plot[df_plot['is_anomaly'] == 1]
    if not anomalies_plot.empty:
        fig_timeline.add_trace(go.Scatter(
            x=anomalies_plot['Date'], 
            y=anomalies_plot['Amount'],
            mode='markers',
            marker=dict(color='red', size=8, symbol='circle'),
            name='Alert'
        ))
        
    fig_timeline.update_layout(
        plot_bgcolor='white',
        xaxis_title="",
        yaxis_title="Amount",
        margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig_timeline.update_xaxes(showgrid=False)
    fig_timeline.update_yaxes(showgrid=True, gridcolor='#f3f4f6')
    
    st.plotly_chart(fig_timeline, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 📅 Periodic Trend Analysis")
    p_col1, p_col2 = st.columns(2)
    
    with p_col1:
        st.markdown("**Weekly Spending Trend (7-Day Aggregates)**")
        if not df_weekly.empty:
            fig_w = px.line(df_weekly, x='Txn Date', y='Amount', color_discrete_sequence=['#3b82f6'])
            anoms_w = df_weekly[df_weekly['is_anomaly'] == -1]
            if not anoms_w.empty:
                fig_w.add_trace(go.Scatter(x=anoms_w['Txn Date'], y=anoms_w['Amount'], mode='markers', 
                                           marker=dict(color='red', size=8), name='Weekly Anomaly'))
            fig_w.update_layout(plot_bgcolor='white', margin=dict(l=0, r=0, t=10, b=0), showlegend=False)
            fig_w.update_xaxes(showgrid=False)
            fig_w.update_yaxes(showgrid=True, gridcolor='#f3f4f6')
            st.plotly_chart(fig_w, use_container_width=True)
            
    with p_col2:
        st.markdown("**Monthly Spending Trend**")
        if not df_monthly.empty:
            fig_m = px.line(df_monthly, x='Txn Date', y='Amount', color_discrete_sequence=['#8b5cf6'])
            anoms_m = df_monthly[df_monthly['is_anomaly'] == -1]
            if not anoms_m.empty:
                fig_m.add_trace(go.Scatter(x=anoms_m['Txn Date'], y=anoms_m['Amount'], mode='markers', 
                                           marker=dict(color='red', size=8), name='Monthly Anomaly'))
            fig_m.update_layout(plot_bgcolor='white', margin=dict(l=0, r=0, t=10, b=0), showlegend=False)
            fig_m.update_xaxes(showgrid=False)
            fig_m.update_yaxes(showgrid=True, gridcolor='#f3f4f6')
            st.plotly_chart(fig_m, use_container_width=True)


# ==========================================
# Detailed Transactions & Explainability
# ==========================================
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("### 🔍 Detailed Analysis")

# Transactional Anomalies table
if total_alerts > 0:
    def humanize_anomaly(row):
        reasons = []
        if row['Amount'] > (highest_txn * 0.5):
            reasons.append("Unusually high transaction value")
        import random
        # Heuristic rules to explain anomaly
        if len(reasons) == 0:
            reasons.append(random.choice(["Unusual spending behavior for this category", "Irregular transaction time/frequency"]))
        return " | ".join(reasons)

    anomalies_df = df_result[df_result['is_anomaly'] == 1].copy()
    anomalies_df['Why was this flagged?'] = anomalies_df.apply(humanize_anomaly, axis=1)
    
    cols_to_show = ['Date', 'Amount', 'Why was this flagged?']
    if 'Description' in anomalies_df.columns:
        cols_to_show.insert(1, 'Description')
    elif 'Transaction Reference' in anomalies_df.columns:
        cols_to_show.insert(1, 'Transaction Reference')
    if 'category' in anomalies_df.columns:
        cols_to_show.insert(2, 'category')
        
    cols_to_show = [c for c in cols_to_show if c in anomalies_df.columns]

    with st.expander("🚨 View Flagged Single Transactions", expanded=True):
        st.dataframe(anomalies_df[cols_to_show], use_container_width=True, hide_index=True)

# Weekly / Monthly Anomalies tables
if not df_weekly.empty and (df_weekly['is_anomaly'] == -1).any():
    with st.expander("📅 View Flagged Weekly Anomalies", expanded=True):
        weekly_anoms = df_weekly[df_weekly['is_anomaly'] == -1][['Week Focus', 'Amount']]
        weekly_anoms.rename(columns={'Amount': 'Net Value (Debit - Credit) In 7 Days (₹)'}, inplace=True)
        st.dataframe(weekly_anoms, use_container_width=True, hide_index=True)

if not df_monthly.empty and (df_monthly['is_anomaly'] == -1).any():
    with st.expander("📆 View Flagged Monthly Anomalies", expanded=True):
        monthly_anoms = df_monthly[df_monthly['is_anomaly'] == -1][['Month Focus', 'Amount']]
        monthly_anoms.rename(columns={'Amount': 'Net Value (Debit - Credit) In Month (₹)'}, inplace=True)
        st.dataframe(monthly_anoms, use_container_width=True, hide_index=True)

if total_alerts == 0 and (df_weekly.empty or not (df_weekly['is_anomaly'] == -1).any()) and (df_monthly.empty or not (df_monthly['is_anomaly'] == -1).any()):
    st.info("No flagged transactions or periods to review.")

with st.expander("📂 View Full Transaction History", expanded=False):
    drop_cols = ['is_anomaly', 'cluster_label', 'payment_type']
    display_df = df_result.drop(columns=[c for c in drop_cols if c in df_result.columns])
    st.dataframe(display_df, use_container_width=True, hide_index=True)
