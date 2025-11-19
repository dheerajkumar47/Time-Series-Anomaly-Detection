# Copyright (c) 2025 Dheeraj Kumar
# Licensed under the MIT License. See LICENSE file for details.

import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import os
from datetime import datetime

# Import custom modules
from frontend_utils import Visualizer, format_metrics_table, format_profiling_table

# API Configuration
API_URL = os.getenv("API_URL", "http://localhost:8000")

# Page configuration
st.set_page_config(
    page_title="Time-Series Anomaly Detection System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        font-size: 2.5rem;
        font-weight: 700;
        color: #0E1117;
        text-align: center;
        margin-bottom: 1rem;
        padding-bottom: 1rem;
        border-bottom: 2px solid #F0F2F6;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #FFFFFF;
        border: 1px solid #E0E0E0;
        border-radius: 8px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #F0F2F6;
        border-radius: 4px 4px 0 0;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #FFFFFF;
        border-bottom: 2px solid #FF4B4B;
    }
</style>
""", unsafe_allow_html=True)


def reset_results_and_splits():
    """Utility: clear train/test/results/profiling when dataset changes."""
    st.session_state.train_data = None
    st.session_state.test_data = None
    st.session_state.results = {}
    st.session_state.profiling_results = {}


def main():
    # Main Header
    st.markdown('<div class="main-header">📊 Time-Series Anomaly Detection System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Detect anomalies in financial, IoT, and server data using AI</div>', unsafe_allow_html=True)

    # Sidebar for global controls
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/2779/2779775.png", width=50)
        st.title("Control Panel")
        st.markdown("---")
        data_source = st.selectbox(
            "Select Data Source",
            ["Generate Synthetic Data", "Upload CSV File"],
            key="data_source"
        )
        st.markdown("---")
        st.info("Use the tabs on the right to navigate through the workflow.")

    # Initialize session state
    if "data" not in st.session_state:
        st.session_state.data = None
    if "train_data" not in st.session_state:
        st.session_state.train_data = None
    if "test_data" not in st.session_state:
        st.session_state.test_data = None
    if "results" not in st.session_state:
        st.session_state.results = {}
    if "profiling_results" not in st.session_state:
        st.session_state.profiling_results = {}

    # Create Tabs
    tab1, tab2, tab3 = st.tabs(["📂 Data & Overview", "⚙️ Model Training", "📈 Analysis & Insights"])

    # ---------------- TAB 1: Data Loading & Overview ---------------- #
    with tab1:
        st.subheader("1. Load & Visualize Data")
        
        if data_source == "Generate Synthetic Data":
            col1, col2, col3 = st.columns(3)
            with col1:
                data_type = st.selectbox("Data Type", ["Financial Transactions", "Server Metrics", "Sensor Data"])
            with col2:
                n_points = st.slider("Data Points", 500, 5000, 1500)
            with col3:
                anomaly_rate = st.slider("Anomaly Rate", 0.01, 0.20, 0.05)

            if st.button("🚀 Generate Data", type="primary", use_container_width=True):
                with st.spinner("Generating synthetic data..."):
                    try:
                        payload = {"data_type": data_type, "n_points": n_points, "anomaly_rate": anomaly_rate}
                        response = requests.post(f"{API_URL}/data/generate", json=payload)
                        response.raise_for_status()
                        
                        # Fetch data
                        data_response = requests.get(f"{API_URL}/data/test")
                        data_response.raise_for_status()
                        df = pd.DataFrame(data_response.json())
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                        st.session_state.data = df
                        reset_results_and_splits()
                        st.toast("Data generated successfully!", icon="✅")
                    except Exception as e:
                        st.error(f"API Error: {e}")

        else:  # Upload CSV
            uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
            with st.expander("CSV Format Options"):
                col1, col2 = st.columns(2)
                timestamp_col = col1.text_input("Timestamp Column", "timestamp")
                value_col = col2.text_input("Value Column", "value")
                timestamp_unit = st.selectbox("Timestamp Unit", ["String", "Seconds", "Milliseconds"])

            if uploaded_file and st.button("📤 Upload & Process", type="primary"):
                with st.spinner("Uploading..."):
                    try:
                        files = {"file": uploaded_file.getvalue()}
                        params = {"timestamp_col": timestamp_col, "value_col": value_col, "timestamp_unit": timestamp_unit}
                        requests.post(f"{API_URL}/data/upload", files={"file": uploaded_file}, params=params).raise_for_status()
                        
                        data_response = requests.get(f"{API_URL}/data/test")
                        df = pd.DataFrame(data_response.json())
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                        st.session_state.data = df
                        reset_results_and_splits()
                        st.toast("File uploaded successfully!", icon="✅")
                    except Exception as e:
                        st.error(f"Upload Error: {e}")

        # Data Overview
        if st.session_state.data is not None:
            st.markdown("### Data Summary")
            m1, m2, m3 = st.columns(3)
            m1.metric("Total Points", len(st.session_state.data))
            m2.metric("Date Range", f"{st.session_state.data['timestamp'].min().date()} - {st.session_state.data['timestamp'].max().date()}")
            if "is_anomaly" in st.session_state.data.columns:
                m3.metric("Anomaly Rate", f"{st.session_state.data['is_anomaly'].mean():.2%}")
            
            st.plotly_chart(Visualizer.plot_time_series_with_anomalies(st.session_state.data), use_container_width=True)

    # ---------------- TAB 2: Model Configuration ---------------- #
    with tab2:
        st.subheader("2. Configure & Train Models")
        
        c1, c2 = st.columns(2)
        with c1:
            with st.container(border=True):
                st.markdown("#### 📈 Prophet (Forecasting)")
                use_prophet = st.toggle("Enable Prophet", value=True)
                if use_prophet:
                    seasonality = st.multiselect("Seasonality", ["Yearly", "Weekly", "Daily"], default=["Yearly", "Weekly", "Daily"])
                    mode = st.selectbox("Mode", ["additive", "multiplicative"])

        with c2:
            with st.container(border=True):
                st.markdown("#### 🌲 Isolation Forest (Unsupervised)")
                use_if = st.toggle("Enable Isolation Forest", value=True)
                if use_if:
                    contamination = st.slider("Contamination", 0.01, 0.2, 0.05)
                    window = st.number_input("Window Size (Hours)", 12, 168, 24)

        if st.button("⚡ Train Models", type="primary", use_container_width=True):
            if not (use_prophet or use_if):
                st.error("Select at least one model.")
            else:
                with st.spinner("Training models... this may take a moment"):
                    try:
                        payload = {
                            "use_prophet": use_prophet,
                            "use_isolation_forest": use_if,
                            "yearly_seasonality": "Yearly" in seasonality if use_prophet else False,
                            "weekly_seasonality": "Weekly" in seasonality if use_prophet else False,
                            "daily_seasonality": "Daily" in seasonality if use_prophet else False,
                            "seasonality_mode": mode if use_prophet else "additive",
                            "contamination": contamination if use_if else 0.05,
                            "window_size": window if use_if else 24
                        }
                        requests.post(f"{API_URL}/models/train", json=payload).raise_for_status()
                        
                        # Fetch all results
                        st.session_state.results = requests.get(f"{API_URL}/models/results").json()
                        st.session_state.profiling_results = requests.get(f"{API_URL}/models/profiling").json()
                        st.toast("Training complete!", icon="🎉")
                        st.balloons()
                    except Exception as e:
                        st.error(f"Training Error: {e}")

    # ---------------- TAB 3: Analysis & Insights ---------------- #
    with tab3:
        st.subheader("3. Results & Performance")
        
        if not st.session_state.results:
            st.info("👈 Please train models in the 'Model Training' tab first.")
        else:
            # Predictions
            st.markdown("#### 🔍 Model Predictions")
            st.plotly_chart(Visualizer.plot_model_comparison(st.session_state.data, st.session_state.results), use_container_width=True)
            
            # Metrics
            st.markdown("#### 📊 Performance Metrics")
            metrics_res = requests.get(f"{API_URL}/models/metrics")
            if metrics_res.status_code == 200:
                metrics = metrics_res.json()
                st.dataframe(format_metrics_table(metrics), use_container_width=True)
                st.plotly_chart(Visualizer.plot_performance_metrics(metrics), use_container_width=True)
            
            # Profiling
            with st.expander("⏱️ System Profiling (Time & Memory)"):
                if st.session_state.profiling_results:
                    prof_df = format_profiling_table(st.session_state.profiling_results)
                    st.dataframe(prof_df, use_container_width=True)
                    st.plotly_chart(Visualizer.plot_profiling_results(st.session_state.profiling_results), use_container_width=True)
            
            # Export
            st.markdown("#### 📥 Export")
            if st.button("Download Predictions CSV"):
                try:
                    test_export = st.session_state.data.copy()
                    for model_name, res in st.session_state.results.items():
                        anomalies = res.get("anomalies")
                        scores = res.get("anomaly_scores")
                        if anomalies:
                            test_export[f"{model_name}_anomaly"] = anomalies
                        if scores:
                            test_export[f"{model_name}_score"] = scores
                    
                    csv = test_export.to_csv(index=False)
                    st.download_button(
                        label="📥 Download CSV", data=csv,
                        file_name=f"anomaly_predictions_{datetime.now().strftime('%Y%m%d_%H%M')}.csv", mime="text/csv"
                    )
                except Exception as e:
                    st.error(f"Export error: {e}")

if __name__ == "__main__":
    main()
