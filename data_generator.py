import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

class TimeSeriesGenerator:
    """Generate synthetic time-series data with controlled anomalies."""
    
    def __init__(self, start_date='2023-01-01', freq='h'):  # 'h' instead of 'H' to avoid warning
        self.start_date = start_date
        self.freq = freq
    
    def generate_base_series(self, n_points=1000, trend=0.01, noise_level=0.1):
        """Generate a base time series with trend, daily/weekly seasonality, and noise."""
        dates = pd.date_range(start=self.start_date, periods=n_points, freq=self.freq)  # Uses 'h'
        
        # Trend component
        trend_component = np.arange(n_points) * trend
        
        # Daily seasonality (period=24 hours)
        daily_season = 10 * np.sin(2 * np.pi * np.arange(n_points) / 24)
        
        # Weekly seasonality (period=24*7 hours)
        weekly_season = 5 * np.sin(2 * np.pi * np.arange(n_points) / (24 * 7))
        
        # Random noise
        noise = np.random.normal(0, noise_level, n_points)
        
        # Combine components
        values = 50 + trend_component + daily_season + weekly_season + noise
        
        return pd.DataFrame({
            'timestamp': dates,
            'value': values,
            'is_anomaly': 0  # Initially no anomalies
        })
    
    def inject_anomalies(self, df, anomaly_rate=0.05, anomaly_types=['spike']):
        """Inject anomalies into the DataFrame based on specified rate and types."""
        df = df.copy().reset_index(drop=True)
        n_total = len(df)
        n_anomalies = max(1, int(n_total * anomaly_rate))  # At least 1 anomaly
        
        # Select random indices for anomalies (without replacement)
        anomaly_indices = random.sample(range(n_total), min(n_anomalies, n_total))
        
        for idx in anomaly_indices:
            value = df.loc[idx, 'value']
            df.loc[idx, 'is_anomaly'] = 1  # Mark as anomaly
            
            # Apply anomaly type
            anomaly_type = random.choice(anomaly_types)
            if anomaly_type == 'spike':
                df.loc[idx, 'value'] *= random.uniform(3, 5)  # 3-5x spike
            elif anomaly_type == 'drop':
                df.loc[idx, 'value'] *= random.uniform(0.2, 0.5)  # 0.2-0.5x drop
            elif anomaly_type == 'shift':
                # Shift affects 5-20 consecutive points
                shift_length = random.randint(5, 20)
                end_idx = min(idx + shift_length, n_total)
                shift_magnitude = value * random.uniform(0.5, 1.5)  # Moderate shift
                for shift_idx in range(idx, end_idx):
                    df.loc[shift_idx, 'value'] += shift_magnitude
                    df.loc[shift_idx, 'is_anomaly'] = 1  # Mark all shifted points as anomalies
        
        return df
    
    def generate_financial_data(self, n_points=2000):
        """Generate financial transaction-like data (positive values, spike anomalies)."""
        df = self.generate_base_series(n_points, trend=0.005, noise_level=0.2)
        df['value'] = np.abs(df['value']) * 100  # Scale to positive transaction amounts
        return self.inject_anomalies(df, anomaly_rate=0.03, anomaly_types=['spike'])
    
    def generate_server_metrics(self, n_points=1500):
        """Generate server CPU/memory usage data (0-100%, spike/shift anomalies)."""
        df = self.generate_base_series(n_points, trend=0.001, noise_level=0.05)
        df['value'] = np.clip(df['value'], 0, 100)  # Clamp to 0-100%
        return self.inject_anomalies(df, anomaly_rate=0.04, anomaly_types=['spike', 'shift'])
    
    def generate_sensor_data(self, n_points=1800):
        """Generate manufacturing sensor data (spike/drop/shift anomalies)."""
        df = self.generate_base_series(n_points, trend=0.002, noise_level=0.15)
        return self.inject_anomalies(df, anomaly_rate=0.06, anomaly_types=['spike', 'drop', 'shift'])