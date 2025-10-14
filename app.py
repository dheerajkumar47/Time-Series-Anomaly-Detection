# Copyright (c) 2025 Dheeraj Kumar
# Licensed under the MIT License. See LICENSE file for details.

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

# Import custom modules
from data_generator import TimeSeriesGenerator
from anomaly_detectors import ProphetDetector, IsolationForestDetector, ModelProfiler, ModelEvaluator
from utils import DataPreprocessor, Visualizer, format_metrics_table, format_profiling_table

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
            with st.spinner("Generating synthetic data..."):
                generator = TimeSeriesGenerator()
                if data_type == "Financial Transactions":
                    df = generator.generate_financial_data(n_points)
                elif data_type == "Server Metrics":
                    df = generator.generate_server_metrics(n_points)
                else:
                    df = generator.generate_sensor_data(n_points)

                # enforce requested anomaly rate by re-injecting on non-anomalous
                current_anomaly_rate = df["is_anomaly"].mean()
                if not np.isclose(current_anomaly_rate, anomaly_rate, atol=0.01):
                    non_anomaly_df = df[df["is_anomaly"] == 0].copy()
                    df = generator.inject_anomalies(non_anomaly_df, anomaly_rate=anomaly_rate)

                # Save data and reset previous model outputs/splits
                st.session_state.data = df.copy()
                reset_results_and_splits()
                st.success(f"Generated {len(df)} data points with {df['is_anomaly'].sum()} anomalies")

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
            df = DataPreprocessor.load_and_validate_data(
                uploaded_file,
                timestamp_col=timestamp_col,
                value_col=value_col,
                anomaly_col=anomaly_col,
                timestamp_unit=timestamp_unit,
            )
            if df is not None:
                st.session_state.data = df.copy()
                reset_results_and_splits()
                st.success(f"Loaded {len(df)} data points")

    # ---------------- Data Overview ---------------- #
    if st.session_state.data is not None:
        try:
            summary = DataPreprocessor.get_data_summary(st.session_state.data)
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Data Points", summary["total_points"])
                st.metric("Anomaly Rate", summary["anomaly_rate"])
            with col2:
                st.metric("Date Range", summary["date_range"])
                st.metric("Value Range", summary["value_range"])
            st.plotly_chart(Visualizer.plot_time_series_with_anomalies(st.session_state.data), use_container_width=True)
        except Exception as e:
            st.error(f"Error showing data overview: {e}")

    # ---------------- Train/Test Split ---------------- #
    st.markdown('<div class="section-header">✂️ Train/Test Split</div>', unsafe_allow_html=True)
    test_size = st.slider("Test Set Size (Proportion)", 0.1, 0.5, 0.3, 0.05, key="test_size")

    # Auto-split when data loads (but only once)
    if st.session_state.data is not None and st.session_state.train_data is None:
        train_df, test_df = DataPreprocessor.split_train_test(st.session_state.data, test_size)
        st.session_state.train_data, st.session_state.test_data = train_df, test_df
        st.success(f"Auto-split: {len(train_df)} train, {len(test_df)} test")

    if st.button("Split Data", key="split_data"):
        if st.session_state.data is not None:
            train_df, test_df = DataPreprocessor.split_train_test(st.session_state.data, test_size)
            st.session_state.train_data, st.session_state.test_data = train_df, test_df
            st.success(f"Manually split: {len(train_df)} train, {len(test_df)} test")
        else:
            st.error("No data loaded. Please load data first.")

    if st.session_state.train_data is not None and st.session_state.test_data is not None:
        st.write(f"✅ Train/Test Split: {len(st.session_state.train_data)} training, {len(st.session_state.test_data)} test")

    # ---------------- Model Config & Training ---------------- #
    st.markdown('<div class="section-header">🤖 Model Configuration & Training</div>', unsafe_allow_html=True)
    if st.session_state.train_data is None or st.session_state.test_data is None:
        st.warning("⚠️ Split data first.")
        return

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Prophet (Baseline)")
        use_prophet = st.checkbox("Use Prophet", value=True, key="use_prophet")
        if use_prophet:
            yearly_seasonality = st.checkbox("Yearly Seasonality", value=True, key="yearly_seasonality")
            weekly_seasonality = st.checkbox("Weekly Seasonality", value=True, key="weekly_seasonality")
            daily_seasonality = st.checkbox("Daily Seasonality", value=True, key="daily_seasonality")
            seasonality_mode = st.selectbox("Seasonality Mode", ["additive", "multiplicative"], key="seasonality_mode")
    with col2:
        st.subheader("Isolation Forest (Advanced)")
        use_isolation_forest = st.checkbox("Use Isolation Forest", value=True, key="use_isolation_forest")
        if use_isolation_forest:
            contamination = st.slider("Contamination Rate", 0.01, 0.2, 0.05, 0.01, key="contamination")
            window_size = st.slider("Window Size (Hours)", 12, 48, 24, key="window_size")

    if st.button("Train Models", type="primary", key="train_button"):
        if not (use_prophet or use_isolation_forest):
            st.error("❌ Select at least one model.")
            return

        results, profiling_results = {}, {}
        train_data, test_data = st.session_state.train_data.copy(), st.session_state.test_data.copy()
        progress_bar, status_text = st.progress(0), st.empty()
        total_models, model_count = sum([use_prophet, use_isolation_forest]), 0

        # Prophet
        if use_prophet:
            status_text.text("Training Prophet...")
            detector = ProphetDetector(
                yearly_seasonality=yearly_seasonality,
                weekly_seasonality=weekly_seasonality,
                daily_seasonality=daily_seasonality,
                seasonality_mode=seasonality_mode
            )
            profiling = ModelProfiler.profile_detector(detector, train_data, test_data)
            # prediction arrays will have same length as test_data inside detectors
            results["Prophet"] = detector.predict(test_data)
            profiling_results["Prophet"] = profiling
            model_count += 1
            progress_bar.progress(model_count / (total_models if total_models > 0 else 1))

        # Isolation Forest
        if use_isolation_forest:
            status_text.text("Training Isolation Forest...")
            detector = IsolationForestDetector(contamination=contamination, window_size=window_size)
            profiling = ModelProfiler.profile_detector(detector, train_data, test_data)
            results["Isolation Forest"] = detector.predict(test_data)
            profiling_results["Isolation Forest"] = profiling
            model_count += 1
            progress_bar.progress(model_count / (total_models if total_models > 0 else 1))

        # Save results to session (overwrite any previous)
        st.session_state.results = results
        st.session_state.profiling_results = profiling_results
        st.success(f"✅ Trained {len(results)} models.")
        status_text.text("Training completed!")

    # ---------------- Results & Analysis ---------------- #
    if st.session_state.results and st.session_state.test_data is not None:
        st.markdown('<div class="section-header">🔍 Results & Analysis</div>', unsafe_allow_html=True)
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Predictions", "📈 Anomaly Scores", "🎯 Performance Metrics", "⚡ Profiling Results"])

        # Tab 1: Model comparison plot
        with tab1:
            try:
                # Visualizer is robust: it will skip models whose prediction length doesn't match test_df
                fig = Visualizer.plot_model_comparison(st.session_state.test_data, st.session_state.results)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Plot error: {e}")

        # Tab 2: Anomaly scores
        with tab2:
            try:
                fig = Visualizer.plot_anomaly_scores(st.session_state.test_data, st.session_state.results)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Plot error: {e}")

        # Tab 3: Metrics
        with tab3:
            if "is_anomaly" in st.session_state.test_data.columns:
                try:
                    metrics_comparison = ModelEvaluator.compare_models(st.session_state.results, st.session_state.test_data)
                    st.dataframe(format_metrics_table(metrics_comparison), use_container_width=True)
                    st.caption("Performance metrics (precision, recall, F1, accuracy, AUC-ROC)")
                    st.plotly_chart(Visualizer.plot_performance_metrics(metrics_comparison), use_container_width=True)

                    if metrics_comparison:
                        # only consider models that returned sane f1_score
                        safe_models = {m: metrics_comparison[m] for m in metrics_comparison if "f1_score" in metrics_comparison[m]}
                        if safe_models:
                            best_model = max(safe_models, key=lambda m: safe_models[m]["f1_score"])
                            st.success(f"🏆 Best Model: {best_model} (F1-Score: {safe_models[best_model]['f1_score']:.2f})")
                except Exception as e:
                    st.error(f"Metrics error: {e}")
            else:
                st.info("No 'is_anomaly' labels available for evaluation. Upload a labeled dataset for metrics.")

        # Tab 4: Profiling
        with tab4:
            try:
                if st.session_state.profiling_results:
                    profiling_df = format_profiling_table(st.session_state.profiling_results)
                    st.dataframe(profiling_df, use_container_width=True)
                    st.plotly_chart(Visualizer.plot_profiling_results(st.session_state.profiling_results), use_container_width=True)
                else:
                    st.info("No profiling results available yet.")
            except Exception as e:
                st.error(f"Profiling plot error: {e}")

    # ---------------- Export Section ---------------- #
    if st.session_state.results and st.session_state.test_data is not None:
        st.markdown('<div class="section-header">📥 Export Results</div>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("Download Predictions", key="export_pred"):
                try:
                    test_export = st.session_state.test_data.copy()
                    # add only models with matching lengths
                    for model_name, res in st.session_state.results.items():
                        anomalies = res.get("anomalies")
                        scores = res.get("anomaly_scores")
                        if anomalies is None or len(anomalies) != len(test_export):
                            st.warning(f"Skipping {model_name}: prediction length mismatch.")
                            continue
                        test_export[f"{model_name}_anomaly"] = anomalies
                        # anomaly_scores might be numpy array; ensure correct length
                        if scores is not None and len(scores) == len(test_export):
                            test_export[f"{model_name}_score"] = scores
                        else:
                            test_export[f"{model_name}_score"] = np.nan
                    csv = test_export.to_csv(index=False)
                    st.download_button(
                        label="📥 Download CSV", data=csv,
                        file_name=f"anomaly_predictions_{datetime.now().strftime('%Y%m%d_%H%M')}.csv", mime="text/csv"
                    )
                except Exception as e:
                    st.error(f"Export error: {e}")

        with col2:
            if st.button("Download Metrics", key="export_metrics"):
                try:
                    if "is_anomaly" in st.session_state.test_data.columns:
                        metrics_comparison = ModelEvaluator.compare_models(st.session_state.results, st.session_state.test_data)
                        metrics_df = format_metrics_table(metrics_comparison)
                        csv = metrics_df.to_csv(index=True)
                        st.download_button(
                            label="📥 Download CSV", data=csv,
                            file_name=f"model_metrics_{datetime.now().strftime('%Y%m%d_%H%M')}.csv", mime="text/csv"
                        )
                    else:
                        st.error("No metrics available (missing 'is_anomaly').")
                except Exception as e:
                    st.error(f"Export metrics error: {e}")

        with col3:
            if st.button("Download Profiling", key="export_profiling"):
                try:
                    profiling_df = format_profiling_table(st.session_state.profiling_results)
                    csv = profiling_df.to_csv(index=True)
                    st.download_button(
                        label="📥 Download CSV", data=csv,
                        file_name=f"model_profiling_{datetime.now().strftime('%Y%m%d_%H%M')}.csv", mime="text/csv"
                    )
                except Exception as e:
                    st.error(f"Export profiling error: {e}")


if __name__ == "__main__":
    main()
