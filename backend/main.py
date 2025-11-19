from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import io
from typing import Dict, Optional

from backend.schemas import (
    DataGenerationRequest, TrainRequest, PredictionResponse, 
    MetricsResponse, ProfilingResponse
)
from backend.core.data_generator import TimeSeriesGenerator
from backend.core.anomaly_detectors import (
    ProphetDetector, IsolationForestDetector, ModelProfiler, ModelEvaluator
)
from backend.core.utils import DataPreprocessor

app = FastAPI(title="Anomaly Detection API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory state storage
class SessionState:
    def __init__(self):
        self.train_data: Optional[pd.DataFrame] = None
        self.test_data: Optional[pd.DataFrame] = None
        self.models: Dict[str, Any] = {}
        self.results: Dict[str, Any] = {}
        self.profiling_results: Dict[str, Any] = {}

state = SessionState()

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/data/generate")
def generate_data(request: DataGenerationRequest):
    try:
        generator = TimeSeriesGenerator()
        if request.data_type == "Financial Transactions":
            df = generator.generate_financial_data(request.n_points)
        elif request.data_type == "Server Metrics":
            df = generator.generate_server_metrics(request.n_points)
        else:
            df = generator.generate_sensor_data(request.n_points)

        # Enforce anomaly rate
        current_anomaly_rate = df["is_anomaly"].mean()
        if not np.isclose(current_anomaly_rate, request.anomaly_rate, atol=0.01):
            non_anomaly_df = df[df["is_anomaly"] == 0].copy()
            df = generator.inject_anomalies(non_anomaly_df, anomaly_rate=request.anomaly_rate)

        # Split data automatically (70/30)
        train_df, test_df = DataPreprocessor.split_train_test(df, test_size=0.3)
        
        state.train_data = train_df
        state.test_data = test_df
        state.models = {}
        state.results = {}
        state.profiling_results = {}
        
        return {
            "message": "Data generated successfully",
            "total_points": len(df),
            "anomalies": int(df["is_anomaly"].sum()),
            "train_size": len(train_df),
            "test_size": len(test_df)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/data/upload")
async def upload_data(
    file: UploadFile = File(...), 
    timestamp_col: str = "timestamp", 
    value_col: str = "value", 
    anomaly_col: str = "is_anomaly",
    timestamp_unit: str = "String"
):
    try:
        contents = await file.read()
        df = DataPreprocessor.load_and_validate_data(
            io.BytesIO(contents),
            timestamp_col=timestamp_col,
            value_col=value_col,
            anomaly_col=anomaly_col,
            timestamp_unit=timestamp_unit
        )
        
        train_df, test_df = DataPreprocessor.split_train_test(df, test_size=0.3)
        state.train_data = train_df
        state.test_data = test_df
        state.models = {}
        state.results = {}
        state.profiling_results = {}
        
        return {
            "message": "Data loaded successfully",
            "total_points": len(df),
            "train_size": len(train_df),
            "test_size": len(test_df)
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/data/test")
def get_test_data():
    if state.test_data is None:
        raise HTTPException(status_code=404, detail="No data loaded")
    # Convert timestamps to string for JSON serialization
    df = state.test_data.copy()
    df['timestamp'] = df['timestamp'].astype(str)
    return df.to_dict(orient="records")

@app.post("/models/train")
def train_models(request: TrainRequest):
    if state.train_data is None or state.test_data is None:
        raise HTTPException(status_code=400, detail="No data loaded. Generate or upload data first.")
    
    results = {}
    profiling_results = {}
    
    try:
        # Prophet
        if request.use_prophet:
            detector = ProphetDetector(
                yearly_seasonality=request.yearly_seasonality,
                weekly_seasonality=request.weekly_seasonality,
                daily_seasonality=request.daily_seasonality,
                seasonality_mode=request.seasonality_mode
            )
            profiling = ModelProfiler.profile_detector(detector, state.train_data, state.test_data)
            results["Prophet"] = detector.predict(state.test_data)
            profiling_results["Prophet"] = profiling
            state.models["Prophet"] = detector

        # Isolation Forest
        if request.use_isolation_forest:
            detector = IsolationForestDetector(
                contamination=request.contamination, 
                window_size=request.window_size
            )
            profiling = ModelProfiler.profile_detector(detector, state.train_data, state.test_data)
            results["Isolation Forest"] = detector.predict(state.test_data)
            profiling_results["Isolation Forest"] = profiling
            state.models["Isolation Forest"] = detector

        state.results = results
        state.profiling_results = profiling_results
        
        return {"message": f"Trained {len(results)} models", "models": list(results.keys())}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models/results")
def get_results():
    if not state.results:
        raise HTTPException(status_code=404, detail="No results available")
    
    # Format results for frontend
    formatted_results = {}
    for model_name, res in state.results.items():
        formatted_results[model_name] = {
            "anomalies": res["anomalies"].tolist(),
            "anomaly_scores": res["anomaly_scores"].tolist()
        }
    return formatted_results

import math

def sanitize_floats(data):
    """Recursively replace NaN/Infinity with None for JSON compliance."""
    if isinstance(data, dict):
        return {k: sanitize_floats(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [sanitize_floats(v) for v in data]
    elif isinstance(data, float):
        if math.isnan(data) or math.isinf(data):
            return None
    return data

@app.get("/models/metrics")
def get_metrics():
    if not state.results or state.test_data is None:
        raise HTTPException(status_code=404, detail="No results or data available")
    
    if "is_anomaly" not in state.test_data.columns:
        raise HTTPException(status_code=400, detail="No ground truth labels available")
        
    metrics = ModelEvaluator.compare_models(state.results, state.test_data)
    return sanitize_floats(metrics)

@app.get("/models/profiling")
def get_profiling():
    if not state.profiling_results:
        raise HTTPException(status_code=404, detail="No profiling results available")
    return state.profiling_results
