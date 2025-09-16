# Time-Series Anomaly Detection System - TODO

## Project Overview
Build a comprehensive anomaly detection system for time-series data using Prophet (baseline) and Isolation Forest (advanced) models. Deliver an interactive Streamlit dashboard with data generation, preprocessing, model training, visualization, evaluation, and export capabilities.

## Key Tasks (Current Status)
- [x] Define folder structure and essential files.
- [x] Implement synthetic data generation (`data_generator.py`).
- [x] Develop anomaly detection models (Prophet, Isolation Forest) (`anomaly_detectors.py`).
- [x] Create data preprocessing and visualization utilities (`utils.py`).
- [x] Build the Streamlit dashboard (`app.py`) with:
  - [x] Data source selection (synthetic/upload).
  - [x] Auto-data generation/loading on startup.
  - [x] Train/test split controls.
  - [x] Model configuration and training.
  - [x] Results visualization (predictions, scores, metrics).
  - [x] Export functionality (predictions, metrics, profiling).
- [ ] (Optional) Add dtaianomaly integration for more advanced time-series detectors.
- [ ] (Optional) Implement online learning (auto-retrain on new data).
- [ ] (Optional) Add alerting system (e.g., Slack/email notifications for anomalies).
- [ ] (Optional) Optimize for production (e.g., Docker containerization, API endpoints).

## Next Steps
1. Test the dashboard with `sample_data.csv` and synthetic datasets.
2. Validate model performance and profiling results.
3. Address any UI/UX improvements (e.g., error messages, loading spinners).
4. Expand with optional features (e.g., dtaianomaly integration) as needed.