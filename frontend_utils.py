import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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

            # Handle results format (API returns lists, local returns dict/arrays)
            # API returns: {'anomalies': [...], 'anomaly_scores': [...]}
            # Local returns: {'anomalies': np.array, ...}
            # We'll assume input is consistent or handle both.
            anomalies = results.get("anomalies")
            if anomalies is None: 
                continue
                
            # Convert to Series/Array if list
            if isinstance(anomalies, list):
                anomalies = pd.Series(anomalies)
            
            # Align with test_df index if needed, but assuming length matches
            if len(anomalies) == len(test_df):
                pred_mask = anomalies == 1
                if pred_mask.any():
                    # Need to index test_df carefully. If test_df is DataFrame, boolean indexing works.
                    # If pred_mask is list, need to convert.
                    subset = test_df[list(pred_mask)]
                    fig.add_trace(go.Scatter(
                        x=subset["timestamp"], y=subset["value"],
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
            scores = results.get("anomaly_scores")
            if scores is None: continue
            
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
            if metric in metrics_df.columns:
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
        if "training_time" in profiling_df.columns:
            fig.add_trace(go.Bar(x=profiling_df.index, y=profiling_df["training_time"]), row=1, col=1)
        if "prediction_time" in profiling_df.columns:
            fig.add_trace(go.Bar(x=profiling_df.index, y=profiling_df["prediction_time"]), row=1, col=2)
        if "memory_training_delta_mb" in profiling_df.columns:
            fig.add_trace(go.Bar(x=profiling_df.index, y=profiling_df["memory_training_delta_mb"]), row=2, col=1)
        if "throughput" in profiling_df.columns:
            fig.add_trace(go.Bar(x=profiling_df.index, y=profiling_df["throughput"]), row=2, col=2)
        fig.update_layout(title="Model Profiling Metrics", height=600, showlegend=False)
        return fig


def format_metrics_table(comparison_results):
    return pd.DataFrame(comparison_results).T.round(4)


def format_profiling_table(profiling_results):
    profiling_df = pd.DataFrame(profiling_results).T
    columns_to_show = ["training_time", "prediction_time", "memory_training_delta_mb", "throughput"]
    # Filter columns that exist
    existing_cols = [c for c in columns_to_show if c in profiling_df.columns]
    return profiling_df[existing_cols].round(4).rename(columns={
        "training_time": "Training Time (s)",
        "prediction_time": "Prediction Time (s)",
        "memory_training_delta_mb": "Memory Delta (MB)",
        "throughput": "Throughput (points/s)"
    })
