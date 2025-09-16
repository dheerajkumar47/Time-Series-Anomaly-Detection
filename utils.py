import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

class DataPreprocessor:
    """Utilities for data preprocessing and validation"""
    
    @staticmethod
    def load_and_validate_data(uploaded_file, timestamp_col='timestamp', value_col='value', anomaly_col='is_anomaly'):
        """Load CSV data and validate against expected columns with dynamic mapping."""
        try:
            df = pd.read_csv(uploaded_file)
            
            # Validate and rename timestamp column
            if timestamp_col not in df.columns:
                st.error(f"❌ Error: Column '{timestamp_col}' not found in the dataset.")
                return None
            df.rename(columns={timestamp_col: 'timestamp'}, inplace=True)
            
            # Validate and rename value column
            if value_col not in df.columns:
                st.error(f"❌ Error: Column '{value_col}' not found in the dataset.")
                return None
            df.rename(columns={value_col: 'value'}, inplace=True)
            
            # Validate and rename anomaly column (optional)
            if anomaly_col in df.columns:
                df.rename(columns={anomaly_col: 'is_anomaly'}, inplace=True)
            else:
                # Add default 'is_anomaly' column if missing
                df['is_anomaly'] = 0
                st.warning(f"⚠️ No anomaly column '{anomaly_col}' found. Assuming all data points are normal.")
            
            # Convert 'timestamp' to datetime (handle potential invalid formats)
            try:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            except Exception as e:
                st.error(f"❌ Error converting 'timestamp' column to datetime: {str(e)}")
                return None
            
            # Sort data by timestamp (critical for time-series processing)
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            # Basic data validation (ensure no missing values in key columns)
            if df['timestamp'].isnull().any():
                st.error("❌ 'timestamp' column contains missing values.")
                return None
            if df['value'].isnull().any():
                st.error("❌ 'value' column contains missing values.")
                return None
            
            return df
        
        except Exception as e:
            st.error(f"❌ General data loading error: {str(e)}")
            return None
    
    @staticmethod
    def split_train_test(df, test_size=0.3):
        """Split time-series data into train/test sets by time (not random)."""
        split_idx = int(len(df) * (1 - test_size))
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()
        return train_df, test_df
    
    @staticmethod
    def get_data_summary(df):
        """Generate summary statistics for the dataset."""
        return {
            'total_points': len(df),
            'date_range': f"{df['timestamp'].min().strftime('%Y-%m-%d %H:%M:%S')} to {df['timestamp'].max().strftime('%Y-%m-%d %H:%M:%S')}",
            'value_range': f"{df['value'].min():.2f} to {df['value'].max():.2f}",
            'anomaly_rate': f"{df['is_anomaly'].mean():.2%}",
            'missing_values': df[['timestamp', 'value']].isnull().sum().sum()  # Focus on key columns
        }

class Visualizer:
    """Visualization utilities for anomaly detection"""
    
    @staticmethod
    def plot_time_series_with_anomalies(df, title="Time Series with Anomalies"):
        """Plot time series with highlighted true anomalies."""
        fig = go.Figure()
        
        # Normal points (blue line)
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['value'],
            mode='lines',
            name='Normal Data',
            line=dict(color='blue', width=1)
        ))
        
        # True anomalies (red markers)
        anomaly_mask = df['is_anomaly'] == 1
        if anomaly_mask.any():
            fig.add_trace(go.Scatter(
                x=df[anomaly_mask]['timestamp'],
                y=df[anomaly_mask]['value'],
                mode='markers',
                name='True Anomalies',
                marker=dict(color='red', size=8, symbol='x')
            ))
        
        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title='Timestamp',
            yaxis_title='Value',
            hovermode='x unified',
            height=500
        )
        return fig
    
    # Remaining Visualizer methods (plot_model_comparison, plot_anomaly_scores, etc.) remain unchanged...
    @staticmethod
    def plot_model_comparison(test_df, results_dict, title="Model Predictions Comparison"):
        """Compare model predictions against true data and true anomalies."""
        fig = make_subplots(
            rows=len(results_dict) + 1,
            cols=1,
            subplot_titles=['Original Data'] + list(results_dict.keys()),
            shared_xaxes=True,
            vertical_spacing=0.05
        )
        
        # Original data (row 1)
        fig.add_trace(go.Scatter(
            x=test_df['timestamp'],
            y=test_df['value'],
            mode='lines',
            name='Original Data',
            line=dict(color='gray', width=1)
        ), row=1, col=1)
        
        # True anomalies (row 1)
        anomaly_mask = test_df['is_anomaly'] == 1
        if anomaly_mask.any():
            fig.add_trace(go.Scatter(
                x=test_df[anomaly_mask]['timestamp'],
                y=test_df[anomaly_mask]['value'],
                mode='markers',
                name='True Anomalies',
                marker=dict(color='red', size=8, symbol='x')
            ), row=1, col=1)
        
        # Model predictions (subsequent rows)
        for i, (model_name, results) in enumerate(results_dict.items(), start=2):
            # Original data line (light blue)
            fig.add_trace(go.Scatter(
                x=test_df['timestamp'],
                y=test_df['value'],
                mode='lines',
                name=f'{model_name} - Data',
                line=dict(color='lightblue', width=1)
            ), row=i, col=1)
            
            # Predicted anomalies (orange markers)
            pred_mask = results['anomalies'] == 1
            if pred_mask.any():
                fig.add_trace(go.Scatter(
                    x=test_df[pred_mask]['timestamp'],
                    y=test_df[pred_mask]['value'],
                    mode='markers',
                    name=f'{model_name} - Predicted',
                    marker=dict(color='orange', size=6, symbol='circle')
                ), row=i, col=1)
        
        # Update layout
        fig.update_layout(
            title=title,
            height=500 * len(results_dict),
            showlegend=False
        )
        return fig
    
    @staticmethod
    def plot_anomaly_scores(test_df, results_dict, title="Anomaly Scores Over Time"):
        """Plot anomaly scores from all models."""
        fig = go.Figure()
        
        # Assign distinct colors to each model
        colors = ['#FF5733', '#33FF57', '#3357FF', '#FF33F6']
        
        for i, (model_name, results) in enumerate(results_dict.items()):
            scores = results['anomaly_scores']
            fig.add_trace(go.Scatter(
                x=test_df['timestamp'],
                y=scores,
                mode='lines',
                name=model_name,
                line=dict(color=colors[i % len(colors)], width=2)
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Timestamp',
            yaxis_title='Anomaly Score',
            hovermode='x unified',
            height=500
        )
        return fig
    
    @staticmethod
    def plot_performance_metrics(comparison_results):
        """Bar plot comparing model metrics (precision, recall, F1, accuracy)."""
        metrics_df = pd.DataFrame(comparison_results).T
        
        # Create a figure with subplots for each metric
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Precision', 'Recall', 'F1 Score', 'Accuracy'],
            specs=[[{"type": "bar"}, {"type": "bar"}],
                  [{"type": "bar"}, {"type": "bar"}]]
        )
        
        # Plot each metric
        for idx, metric in enumerate(['precision', 'recall', 'f1_score', 'accuracy']):
            row = (idx // 2) + 1
            col = (idx % 2) + 1
            
            fig.add_trace(go.Bar(
                x=metrics_df.index,
                y=metrics_df[metric],
                name=metric.replace('_', ' ').title()
            ), row=row, col=col)
        
        # Update layout
        fig.update_layout(
            title='Model Performance Metrics',
            height=600,
            showlegend=False
        )
        return fig
    
    @staticmethod
    def plot_profiling_results(profiling_results):
        """Plot model profiling metrics (training time, memory usage, etc.)."""
        profiling_df = pd.DataFrame(profiling_results).T
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Training Time (s)', 'Prediction Time (s)', 'Memory Usage (MB)', 'Throughput (points/s)'],
            specs=[[{"type": "bar"}, {"type": "bar"}],
                  [{"type": "bar"}, {"type": "bar"}]]
        )
        
        # Training time
        fig.add_trace(go.Bar(
            x=profiling_df.index,
            y=profiling_df['training_time']
        ), row=1, col=1)
        
        # Prediction time
        fig.add_trace(go.Bar(
            x=profiling_df.index,
            y=profiling_df['prediction_time']
        ), row=1, col=2)
        
        # Memory delta (after training)
        fig.add_trace(go.Bar(
            x=profiling_df.index,
            y=profiling_df['memory_training_delta_mb']  # Use updated column name
        ), row=2, col=1)
        
        # Throughput
        fig.add_trace(go.Bar(
            x=profiling_df.index,
            y=profiling_df['throughput']
        ), row=2, col=2)
        
        fig.update_layout(
            title='Model Profiling Metrics',
            height=600,
            showlegend=False
        )
        return fig

def format_metrics_table(comparison_results):
    """Convert metric comparison results to a Streamlit-friendly DataFrame."""
    metrics_df = pd.DataFrame(comparison_results).T.round(4)
    return metrics_df

def format_profiling_table(profiling_results):
    """Convert profiling results to a Streamlit-friendly DataFrame."""
    profiling_df = pd.DataFrame(profiling_results).T
    columns_to_show = ['training_time', 'prediction_time', 'memory_training_delta_mb', 'throughput']
    formatted_df = profiling_df[columns_to_show].round(4).rename(columns={
        'training_time': 'Training Time (s)',
        'prediction_time': 'Prediction Time (s)',
        'memory_training_delta_mb': 'Memory Delta (MB)',  # Updated column name
        'throughput': 'Throughput (points/s)'
    })
    return formatted_df