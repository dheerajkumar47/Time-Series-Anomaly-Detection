from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Union

class DataGenerationRequest(BaseModel):
    data_type: str  # "Financial Transactions", "Server Metrics", "Sensor Data"
    n_points: int = 1500
    anomaly_rate: float = 0.05

class TrainRequest(BaseModel):
    # For simplicity, we'll assume data is already loaded/generated in the backend session
    # or passed (but passing large data in JSON is bad).
    # Since the plan assumed in-memory state, we'll rely on session state in the backend service.
    # However, FastAPI is stateless by default. We need a way to store data.
    # For this MVP, we will use a global variable in main.py or a simple singleton.
    use_prophet: bool = True
    use_isolation_forest: bool = True
    yearly_seasonality: bool = True
    weekly_seasonality: bool = True
    daily_seasonality: bool = True
    seasonality_mode: str = "additive"
    contamination: float = 0.05
    window_size: int = 24

class PredictionResponse(BaseModel):
    model_name: str
    anomalies: List[int]
    scores: List[float]

class MetricsResponse(BaseModel):
    precision: float
    recall: float
    f1_score: float
    accuracy: float
    auc_roc: float

class ProfilingResponse(BaseModel):
    training_time: float
    prediction_time: float
    memory_training_delta_mb: float
    throughput: float
