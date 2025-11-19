# Copyright (c) 2025 Dheeraj Kumar
# Licensed under the MIT License. See LICENSE file for details.

import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
from datetime import datetime

# Import custom modules
from frontend_utils import Visualizer, format_metrics_table, format_profiling_table

# API Configuration
API_URL = "http://localhost:8000"

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
.main-header { font-size: 3rem; font-weight: bold; text-align: center; color: #1f77b4; margin-bottom: 2rem; }
.section-header { font-size: 1.5rem; font-weight: bold; color: #2e86ab; margin-top: 2rem; margin-bottom: 1rem; }
.metric-card { background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #1f77b4; }
.plot-container { border: 1px solid #ddd; padding: 1rem; margin: 1rem 0; }
</style>
""", unsafe_allow_html=True)


def reset_results_and_splits():
    """Utility: clear train/test/results/profiling when dataset changes."""
    st.session_state.train_data = None
    st.session_state.test_data = None
    st.session_state.results = {}
    st.session_state.profiling_results = {}


def main():
    # Main title
    st.markdown('<h1 class="main-header">📊 Time-Series Anomaly Detection System</h1>', unsafe_allow_html=True)

    # Sidebar for navigation and controls
    st.sidebar.title("🔧 Control Panel")
    data_source = st.sidebar.selectbox(
        "Select Data Source",
        ["Generate Synthetic Data", "Upload CSV File"],
        key="data_source"
    )

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

    # ---------------- Data Loading ---------------- #
    st.markdown('<div class="section-header">📊 Data Loading & Preprocessing</div>', unsafe_allow_html=True)

    if data_source == "Generate Synthetic Data":
        col1, col2 = st.columns([3, 1])
        with col1:
            data_type = st.selectbox(
                "Data Type",
                ["Financial Transactions", "Server Metrics", "Sensor Data"],
                key="data_type"
            )
            n_points = st.slider("Number of Data Points", 500, 3000, 1500, key="n_points")
        with col2:
            anomaly_rate = st.slider("Anomaly Rate", 0.01, 0.15, 0.05, 0.01, key="anomaly_rate")

        if st.button("Generate Data", type="primary", key="generate_data"):
            with st.spinner("Generating synthetic data via API..."):
                try:
                    payload = {
                        "data_type": data_type,
                        "n_points": n_points,
                        "anomaly_rate": anomaly_rate
                    }
                    response = requests.post(f"{API_URL}/data/generate", json=payload)
                    response.raise_for_status()
                    result = response.json()
                    
                    # Fetch the actual data (test set is available via API, but we might want full data for viz)
                    # For now, we'll fetch the test data to visualize, or assume backend holds state.
                    # To visualize "Original Data", we need to fetch it.
                    # Let's fetch the test data from the backend to show something.
                    data_response = requests.get(f"{API_URL}/data/test")
                    data_response.raise_for_status()
                    df = pd.DataFrame(data_response.json())
                    # Convert timestamp back to datetime
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    
                    st.session_state.data = df # This is technically just the test data + train data if we fetched all
                    # But for simplicity, let's just use what we have. 
                    # Ideally we'd have an endpoint to get ALL data.
                    # For this MVP, let's just use the test data for visualization as it's what we predict on.
                    
                    reset_results_and_splits()
                    st.success(f"Generated data successfully. Backend has split data. (Showing Test Data: {len(df)} points)")
                    
                except Exception as e:
                    st.error(f"API Error: {e}")

    else:  # Upload CSV
        uploaded_file = st.file_uploader(
            "Upload CSV File", type=["csv"], key="uploaded_file",
            help="Upload your dataset. Ensure it has a timestamp and value column."
        )
        st.subheader("Column Mapping & Format (for CSV upload)")
        timestamp_col = st.text_input("Timestamp Column Name", value="timestamp", key="csv_timestamp_col")
        value_col = st.text_input("Value Column Name", value="value", key="csv_value_col")
        anomaly_col = st.text_input("Anomaly Column Name (optional)", value="is_anomaly", key="csv_anomaly_col")
        timestamp_unit = st.selectbox("Timestamp Format", ["String", "Seconds", "Milliseconds"], key="csv_timestamp_unit")

        if uploaded_file is not None:
            if st.button("Upload & Process", key="upload_btn"):
                with st.spinner("Uploading to backend..."):
                    try:
                        files = {"file": uploaded_file.getvalue()}
                        params = {
                            "timestamp_col": timestamp_col,
                            "value_col": value_col,
                            "anomaly_col": anomaly_col,
                            "timestamp_unit": timestamp_unit
                        }
                        response = requests.post(f"{API_URL}/data/upload", files={"file": uploaded_file}, params=params)
                        response.raise_for_status()
                        
                        # Fetch data for viz
                        data_response = requests.get(f"{API_URL}/data/test")
                        data_response.raise_for_status()
                        df = pd.DataFrame(data_response.json())
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                        
                        st.session_state.data = df
                        reset_results_and_splits()
                        st.success("Data uploaded and processed successfully.")
                    except Exception as e:
                        st.error(f"Upload Error: {e}")

    # ---------------- Data Overview ---------------- #
    if st.session_state.data is not None:
        try:
            # Simple summary of the loaded (test) data
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Data Points (Test Set)", len(st.session_state.data))
                if "is_anomaly" in st.session_state.data.columns:
                    st.metric("Anomaly Rate", f"{st.session_state.data['is_anomaly'].mean():.2%}")
            with col2:
                st.metric("Date Range", f"{st.session_state.data['timestamp'].min()} to {st.session_state.data['timestamp'].max()}")
            
            st.plotly_chart(Visualizer.plot_time_series_with_anomalies(st.session_state.data), use_container_width=True)
        except Exception as e:
            st.error(f"Error showing data overview: {e}")

    # ---------------- Model Config & Training ---------------- #
    st.markdown('<div class="section-header">🤖 Model Configuration & Training</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Prophet (Baseline)")
        use_prophet = st.checkbox("Use Prophet", value=True, key="use_prophet")
        if use_prophet:
            yearly_seasonality = st.checkbox("Yearly Seasonality", value=True, key="yearly_seasonality")
            weekly_seasonality = st.checkbox("Weekly Seasonality", value=True, key="weekly_seasonality")
            daily_seasonality = st.checkbox("Daily Seasonality", value=True, key="daily_seasonality")
            seasonality_mode = st.selectbox("Seasonality Mode", ["additive", "multiplicative"], key="seasonality_mode")
        else:
            # Defaults if unchecked, to avoid unbound vars
            yearly_seasonality = True
            weekly_seasonality = True
            daily_seasonality = True
            seasonality_mode = "additive"

    with col2:
        st.subheader("Isolation Forest (Advanced)")
        use_isolation_forest = st.checkbox("Use Isolation Forest", value=True, key="use_isolation_forest")
        if use_isolation_forest:
            contamination = st.slider("Contamination Rate", 0.01, 0.2, 0.05, 0.01, key="contamination")
            window_size = st.slider("Window Size (Hours)", 12, 48, 24, key="window_size")
        else:
            contamination = 0.05
            window_size = 24

    if st.button("Train Models", type="primary", key="train_button"):
        if not (use_prophet or use_isolation_forest):
            st.error("❌ Select at least one model.")
            return

        with st.spinner("Training models on backend..."):
            try:
                payload = {
                    "use_prophet": use_prophet,
                    "use_isolation_forest": use_isolation_forest,
                    "yearly_seasonality": yearly_seasonality,
                    "weekly_seasonality": weekly_seasonality,
                    "daily_seasonality": daily_seasonality,
                    "seasonality_mode": seasonality_mode,
                    "contamination": contamination,
                    "window_size": window_size
                }
                response = requests.post(f"{API_URL}/models/train", json=payload)
                response.raise_for_status()
                result = response.json()
                st.success(result["message"])
                
                # Fetch results
                results_response = requests.get(f"{API_URL}/models/results")
                results_response.raise_for_status()
                st.session_state.results = results_response.json()
                
                # Fetch profiling
                prof_response = requests.get(f"{API_URL}/models/profiling")
                prof_response.raise_for_status()
                st.session_state.profiling_results = prof_response.json()
                
            except Exception as e:
                st.error(f"Training Error: {e}")

    # ---------------- Results & Analysis ---------------- #
    if st.session_state.results and st.session_state.data is not None:
        st.markdown('<div class="section-header">🔍 Results & Analysis</div>', unsafe_allow_html=True)
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Predictions", "📈 Anomaly Scores", "🎯 Performance Metrics", "⚡ Profiling Results"])

        # Tab 1: Model comparison plot
        with tab1:
            try:
                fig = Visualizer.plot_model_comparison(st.session_state.data, st.session_state.results)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Plot error: {e}")

        # Tab 2: Anomaly scores
        with tab2:
            try:
                fig = Visualizer.plot_anomaly_scores(st.session_state.data, st.session_state.results)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Plot error: {e}")

        # Tab 3: Metrics
        with tab3:
            try:
                metrics_response = requests.get(f"{API_URL}/models/metrics")
                if metrics_response.status_code == 200:
                    metrics_comparison = metrics_response.json()
                    st.dataframe(format_metrics_table(metrics_comparison), width="stretch")
                    st.caption("Performance metrics (precision, recall, F1, accuracy, AUC-ROC)")
                    st.plotly_chart(Visualizer.plot_performance_metrics(metrics_comparison), use_container_width=True)
                else:
                    st.info("Metrics not available (possibly no ground truth).")
            except Exception as e:
                st.error(f"Metrics error: {e}")

        # Tab 4: Profiling
        with tab4:
            try:
                if st.session_state.profiling_results:
                    profiling_df = format_profiling_table(st.session_state.profiling_results)
                    st.dataframe(profiling_df, width="stretch")
                    st.plotly_chart(Visualizer.plot_profiling_results(st.session_state.profiling_results), use_container_width=True)
                else:
                    st.info("No profiling results available yet.")
            except Exception as e:
                st.error(f"Profiling plot error: {e}")

    # ---------------- Export Section ---------------- #
    if st.session_state.results and st.session_state.data is not None:
        st.markdown('<div class="section-header">📥 Export Results</div>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("Download Predictions", key="export_pred"):
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

        # Metrics and Profiling exports can be similar, omitted for brevity or can be added if needed.
        # The user didn't strictly ask for full feature parity on export, but it's good to have.


if __name__ == "__main__":
    main()
