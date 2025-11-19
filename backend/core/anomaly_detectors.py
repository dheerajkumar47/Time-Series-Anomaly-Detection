import numpy as np
import pandas as pd
from prophet import Prophet
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
import time
import psutil
import os

class ProphetDetector:
    """Prophet-based anomaly detector for time series."""
    
    def __init__(self, yearly_seasonality=True, weekly_seasonality=True, 
                 daily_seasonality=True, seasonality_mode='additive'):
        self.model = Prophet(
            yearly_seasonality=yearly_seasonality,
            weekly_seasonality=weekly_seasonality,
            daily_seasonality=daily_seasonality,
            seasonality_mode=seasonality_mode,
            interval_width=0.95  # 95% confidence interval
        )
        self.is_fitted = False
        self.scaler = StandardScaler()  # For scaling (optional, but consistent)
    
    def fit(self, df):
        """Fit the Prophet model to training data."""
        # Prepare data (Prophet requires 'ds' and 'y' columns)
        prophet_df = df[['timestamp', 'value']].rename(columns={'timestamp': 'ds', 'value': 'y'})
        self.model.fit(prophet_df)
        self.is_fitted = True
        return self
    
    def predict(self, df):
        """Predict anomalies and return scores/predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction.")
        
        # Prepare input for prediction (same as training)
        prophet_df = df[['timestamp', 'value']].rename(columns={'timestamp': 'ds', 'value': 'y'}).reset_index(drop=True)
        
        # Generate forecast
        forecast = self.model.predict(prophet_df)
        
        # Calculate residuals (actual - predicted)
        residuals = prophet_df['y'].values - forecast['yhat'].values
        
        # Anomaly scores (absolute residual)
        anomaly_scores = np.abs(residuals)
        
        # Detect anomalies (outside confidence interval)
        lower_bound = forecast['yhat_lower'].values
        upper_bound = forecast['yhat_upper'].values
        anomalies = ((prophet_df['y'].values < lower_bound) | (prophet_df['y'].values > upper_bound)).astype(int)
        
        return {
            'anomaly_scores': anomaly_scores,
            'anomalies': anomalies,
            'forecast': forecast,
            'residuals': residuals
        }

class IsolationForestDetector:
    """Isolation Forest detector for time series anomalies."""
    
    def __init__(self, contamination=0.05, window_size=24):
        self.contamination = contamination  # Expected proportion of anomalies
        self.window_size = window_size        # Sliding window for features
        self.model = IsolationForest(
            contamination=contamination,
            random_state=42
            # Removed 'Behaviour' parameter (deprecated and invalid)
        )
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def _create_window_features(self, values):
        """Extract sliding window features (mean, std, min, max, etc.)."""
        features = []
        for i in range(self.window_size, len(values)):
            window = values[i - self.window_size : i]
            current_value = values[i]
            features.append([
                np.mean(window),          # Window mean
                np.std(window),          # Window standard deviation
                np.min(window),          # Window min
                np.max(window),          # Window max
                current_value - np.mean(window),  # Current - window mean
                current_value - window[-1]         # Current - previous value (trend)
            ])
        return np.array(features)
    
    def fit(self, df):
        """Fit the Isolation Forest model using windowed features."""
        values = df['value'].values
        
        # Create features from sliding window
        features = self._create_window_features(values)
        if len(features) == 0:
            raise ValueError(f"Not enough data points for window size {self.window_size}")
        
        # Scale features (Isolation Forest is sensitive to scale)
        features_scaled = self.scaler.fit_transform(features)
        
        # Fit the model
        self.model.fit(features_scaled)
        self.is_fitted = True
        return self
    
    def predict(self, df):
        """Predict anomalies on new data and return padded results."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction.")
        
        values = df['value'].values
        features = self._create_window_features(values)
        
        if len(features) == 0:
            # No features (insufficient data): return all 0s
            padded_scores = np.zeros(len(values))
            padded_anomalies = np.zeros(len(values))
        else:
            # Scale features
            features_scaled = self.scaler.transform(features)
            
            # Predict anomalies (scores and labels)
            anomaly_scores = -self.model.decision_function(features_scaled)  # Higher = more anomalous
            anomalies = (self.model.predict(features_scaled) == -1).astype(int)
            
            # Pad with 0s for the initial window (no features available)
            padded_scores = np.concatenate([np.zeros(self.window_size), anomaly_scores])
            padded_anomalies = np.concatenate([np.zeros(self.window_size), anomalies])
        
        return {
            'anomaly_scores': padded_scores,
            'anomalies': padded_anomalies,
            'residuals': None  # Not applicable for Isolation Forest
        }

class ModelProfiler:
    """Profile model performance (training/prediction time, memory usage)."""
    
    @staticmethod
    def profile_detector(detector, train_df, test_df):
        """Profile a detector's training and prediction performance."""
        process = psutil.Process(os.getpid())  # Track current process
        
        # Memory before training
        memory_before = process.memory_info().rss / (1024 ** 2)  # Convert bytes to MB
        
        # Training time
        start_time = time.time()
        detector.fit(train_df)
        training_time = time.time() - start_time
        
        # Memory after training
        memory_after_training = process.memory_info().rss / (1024 ** 2)
        
        # Prediction time
        start_time = time.time()
        results = detector.predict(test_df)
        prediction_time = time.time() - start_time
        
        # Memory after prediction
        memory_after_prediction = process.memory_info().rss / (1024 ** 2)
        
        # Throughput (points processed per second)
        throughput = len(test_df) / prediction_time if prediction_time > 0 else 0
        
        return {
            'training_time': training_time,
            'prediction_time': prediction_time,
            'memory_before_mb': memory_before,
            'memory_after_training_mb': memory_after_training,
            'memory_after_prediction_mb': memory_after_prediction,
            'memory_training_delta_mb': memory_after_training - memory_before,
            'memory_prediction_delta_mb': memory_after_prediction - memory_after_training,
            'throughput': throughput
        }

class ModelEvaluator:
    """Evaluate anomaly detection models using standard metrics."""
    
    @staticmethod
    def evaluate_predictions(y_true, y_pred, y_scores=None):
        """Compute precision, recall, F1, accuracy, and AUC-ROC (if scores exist)."""
        # Basic metrics (precision, recall, F1, accuracy)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
        accuracy = np.mean(y_true == y_pred)
        
        # AUC-ROC (requires anomaly scores)
        auc_roc = 0.0
        if y_scores is not None:
            try:
                auc_roc = roc_auc_score(y_true, y_scores)
            except ValueError:
                auc_roc = 0.0  # Handle edge cases (e.g., all labels same)
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'accuracy': accuracy,
            'auc_roc': auc_roc
        }
    
    @staticmethod
    def compare_models(results_dict, test_df):
        """Compare multiple models by extracting metrics from results."""
        comparison = {}
        y_true = test_df['is_anomaly'].values
        
        for model_name, results in results_dict.items():
            y_pred = results['anomalies']
            y_scores = results['anomaly_scores']
            
            # Ensure arrays are aligned (trim if necessary)
            min_len = min(len(y_true), len(y_pred), len(y_scores))
            y_true_trimmed = y_true[:min_len]
            y_pred_trimmed = y_pred[:min_len]
            y_scores_trimmed = y_scores[:min_len]
            
            # Evaluate and store metrics
            metrics = ModelEvaluator.evaluate_predictions(y_true_trimmed, y_pred_trimmed, y_scores_trimmed)
            comparison[model_name] = metrics
        
        return comparison