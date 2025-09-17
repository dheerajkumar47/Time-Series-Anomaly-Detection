import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st


class DataPreprocessor:
    """Utilities for data preprocessing and validation"""

    @staticmethod
    def load_and_validate_data(
        uploaded_file,
        timestamp_col="timestamp",
        value_col="value",
        anomaly_col="is_anomaly",
        timestamp_unit="String",
        max_points=500000,
    ):
        """Load CSV data, validate columns, convert timestamp, handle anomalies, and limit large datasets."""
        try:
            df = pd.read_csv(uploaded_file)

            # Validate and rename timestamp column
            if timestamp_col not in df.columns:
                st.error(f"❌ Error: Column '{timestamp_col}' not found in the dataset.")
                return None
            df.rename(columns={timestamp_col: "timestamp"}, inplace=True)

            # Validate and rename value column
            if value_col not in df.columns:
                st.error(f"❌ Error: Column '{value_col}' not found in the dataset.")
                return None
            df.rename(columns={value_col: "value"}, inplace=True)

            # Validate and rename anomaly column (optional)
            if anomaly_col in df.columns:
                df.rename(columns={anomaly_col: "is_anomaly"}, inplace=True)
                if not pd.api.types.is_integer_dtype(df["is_anomaly"]):
                    df["is_anomaly"] = df["is_anomaly"].astype(int)
            else:
                df["is_anomaly"] = 0
                st.warning(
                    f"⚠️ No anomaly column '{anomaly_col}' found. Assuming all data points are normal."
                )

                # Optional: inject synthetic anomalies if dataset has none
                if df["is_anomaly"].sum() == 0 and len(df) > 100:
                    anomaly_idx = df.sample(frac=0.01, random_state=42).index
                    df.loc[anomaly_idx, "is_anomaly"] = 1
                    st.info("🔄 Synthetic anomalies injected (1% of dataset).")

            # Convert timestamp column
            try:
                if timestamp_unit in ["Seconds", "Milliseconds"]:
                    if not pd.api.types.is_numeric_dtype(df["timestamp"]):
                        st.error(
                            f"❌ 'timestamp' column is not numeric but unit '{timestamp_unit}' is selected. Use 'String' instead."
                        )
                        return None

                    pd_unit = "s" if timestamp_unit == "Seconds" else "ms"
                    df["timestamp"] = pd.to_datetime(
                        df["timestamp"], unit=pd_unit, origin="unix", utc=True, errors="coerce"
                    )
                else:  # treat as string datetime
                    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)

            except Exception as e:
                st.error(f"❌ Timestamp conversion failed: {str(e)}. Check column values/unit.")
                return None

            # Verify timestamp conversion
            if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
                st.error("❌ 'timestamp' column is not datetime-like. Ensure valid date format.")
                return None

            # Strip timezone if present
            if df["timestamp"].dt.tz is not None:
                df["timestamp"] = df["timestamp"].dt.tz_convert("UTC").dt.tz_localize(None)
                st.warning("🔄 Timezone-aware timestamps converted to UTC and stripped.")

            # Sort data
            df = df.sort_values("timestamp").reset_index(drop=True)

            # Check missing values
            if df["timestamp"].isnull().any():
                st.error("❌ Missing values detected in 'timestamp' column.")
                return None
            if df["value"].isnull().any():
                st.error("❌ Missing values detected in 'value' column.")
                return None

            # Downsample very large datasets
            if len(df) > max_points:
                st.warning(
                    f"⚠️ Dataset too large ({len(df)} rows). Using first {max_points} points for training."
                )
                df = df.head(max_points).copy()

            return df

        except Exception as e:
            st.error(f"❌ General data error: {str(e)}")
            return None

    @staticmethod
    def split_train_test(df, test_size=0.3):
        """Split time-series data into train/test sets by time (not random)."""
        split_idx = int(len(df) * (1 - test_size))
        return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()

    @staticmethod
    def get_data_summary(df):
        """Generate dataset summary stats."""
        return {
            "total_points": len(df),
            "date_range": f"{df['timestamp'].min().strftime('%Y-%m-%d %H:%M:%S')} to {df['timestamp'].max().strftime('%Y-%m-%d %H:%M:%S')}",
            "value_range": f"{df['value'].min():.2f} to {df['value'].max():.2f}",
            "anomaly_rate": f"{df['is_anomaly'].mean():.2%}",
            "missing_values": df[["timestamp", "value"]].isnull().sum().sum(),
        }


# ---------------- Visualization Utils ---------------- #

class Visualizer:
    """Visualization utilities for anomaly detection"""

    @staticmethod
    def plot_time_series_with_anomalies(df, title="Time Series with Anomalies"):
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["timestamp"], y=df["value"],
            mode="lines", name="Normal Data",
            line=dict(color="blue", width=1)
        ))
        anomaly_mask = df["is_anomaly"] == 1
        if anomaly_mask.any():
            fig.add_trace(go.Scatter(
                x=df[anomaly_mask]["timestamp"], y=df[anomaly_mask]["value"],
                mode="markers", name="True Anomalies",
                marker=dict(color="red", size=8, symbol="x")
            ))
        fig.update_layout(title=title, xaxis_title="Timestamp", yaxis_title="Value", height=500, hovermode="x unified")
        return fig

    @staticmethod
    def plot_model_comparison(test_df, results_dict, title="Model Predictions Comparison"):
        """Compare model predictions vs. true anomalies."""
        fig = make_subplots(
            rows=len(results_dict) + 1, cols=1,
            subplot_titles=["Original Data"] + list(results_dict.keys()),
            shared_xaxes=True, vertical_spacing=0.05
        )

        # Original data
        fig.add_trace(go.Scatter(
            x=test_df["timestamp"], y=test_df["value"],
            mode="lines", name="Original Data",
            line=dict(color="gray", width=1)
        ), row=1, col=1)

        # True anomalies
        anomaly_mask = test_df["is_anomaly"] == 1
        if anomaly_mask.any():
            fig.add_trace(go.Scatter(
                x=test_df[anomaly_mask]["timestamp"], y=test_df[anomaly_mask]["value"],
                mode="markers", name="True Anomalies",
                marker=dict(color="red", size=8, symbol="x")
            ), row=1, col=1)

        # Model predictions
        for i, (model_name, results) in enumerate(results_dict.items(), start=2):
            fig.add_trace(go.Scatter(
                x=test_df["timestamp"], y=test_df["value"],
                mode="lines", name=f"{model_name} - Data",
                line=dict(color="lightblue", width=1)
            ), row=i, col=1)

            pred_mask = results["anomalies"] == 1
            if pred_mask.any():
                fig.add_trace(go.Scatter(
                    x=test_df[pred_mask]["timestamp"], y=test_df[pred_mask]["value"],
                    mode="markers", name=f"{model_name} - Predicted",
                    marker=dict(color="orange", size=6, symbol="circle")
                ), row=i, col=1)

        fig.update_layout(title=title, height=500 * len(results_dict), showlegend=False)
        return fig

    @staticmethod
    def plot_anomaly_scores(test_df, results_dict, title="Anomaly Scores Over Time"):
        fig = go.Figure()
        colors = ["#FF5733", "#33FF57", "#3357FF", "#FF33F6"]
        for i, (model_name, results) in enumerate(results_dict.items()):
            scores = results["anomaly_scores"]
            fig.add_trace(go.Scatter(
                x=test_df["timestamp"], y=scores,
                mode="lines", name=model_name,
                line=dict(color=colors[i % len(colors)], width=2)
            ))
        fig.update_layout(title=title, xaxis_title="Timestamp", yaxis_title="Anomaly Score", hovermode="x unified", height=500)
        return fig

    @staticmethod
    def plot_performance_metrics(comparison_results):
        metrics_df = pd.DataFrame(comparison_results).T
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=["Precision", "Recall", "F1 Score", "Accuracy"],
            specs=[[{"type": "bar"}, {"type": "bar"}], [{"type": "bar"}, {"type": "bar"}]]
        )
        for idx, metric in enumerate(["precision", "recall", "f1_score", "accuracy"]):
            row, col = (idx // 2) + 1, (idx % 2) + 1
            fig.add_trace(go.Bar(x=metrics_df.index, y=metrics_df[metric], name=metric.title()), row=row, col=col)
        fig.update_layout(title="Model Performance Metrics", height=600, showlegend=False)
        return fig

    @staticmethod
    def plot_profiling_results(profiling_results):
        profiling_df = pd.DataFrame(profiling_results).T
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=["Training Time (s)", "Prediction Time (s)", "Memory Usage (MB)", "Throughput (points/s)"],
            specs=[[{"type": "bar"}, {"type": "bar"}], [{"type": "bar"}, {"type": "bar"}]]
        )
        fig.add_trace(go.Bar(x=profiling_df.index, y=profiling_df["training_time"]), row=1, col=1)
        fig.add_trace(go.Bar(x=profiling_df.index, y=profiling_df["prediction_time"]), row=1, col=2)
        fig.add_trace(go.Bar(x=profiling_df.index, y=profiling_df["memory_training_delta_mb"]), row=2, col=1)
        fig.add_trace(go.Bar(x=profiling_df.index, y=profiling_df["throughput"]), row=2, col=2)
        fig.update_layout(title="Model Profiling Metrics", height=600, showlegend=False)
        return fig


def format_metrics_table(comparison_results):
    return pd.DataFrame(comparison_results).T.round(4)


def format_profiling_table(profiling_results):
    profiling_df = pd.DataFrame(profiling_results).T
    columns_to_show = ["training_time", "prediction_time", "memory_training_delta_mb", "throughput"]
    return profiling_df[columns_to_show].round(4).rename(columns={
        "training_time": "Training Time (s)",
        "prediction_time": "Prediction Time (s)",
        "memory_training_delta_mb": "Memory Delta (MB)",
        "throughput": "Throughput (points/s)"
    })
