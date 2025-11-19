[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](./LICENSE)
![Author: Dheeraj Kumar](https://img.shields.io/badge/Author-Dheeraj_Kumar-blue.svg)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688.svg)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B.svg)
![Machine Learning](https://img.shields.io/badge/ML-Isolation%20Forest-orange.svg)
![Forecasting](https://img.shields.io/badge/Forecasting-Prophet-yellow.svg)
![Status](https://img.shields.io/badge/Project_Status-Active-brightgreen.svg)

---

# 📊 Time-Series Anomaly Detection System
### _An Interactive, Scalable, and Explainable AI System for Detecting Anomalies in Time-Series Data_

---

## 🧠 Overview
This project is a modern, **decoupled AI system** designed to **detect anomalies in time-series datasets** with high precision. It features a robust **FastAPI backend** for data processing and model training, paired with an interactive **Streamlit frontend** for visualization and control.

Leveraging **Prophet (Meta)** for trend forecasting and **Isolation Forest** for unsupervised anomaly detection, this tool is built for **researchers, data scientists, and engineers** working in **finance, IoT, and system monitoring**.

### 🌟 Why This Project?
- **Decoupled Architecture**: Scalable backend API separated from the UI.
- **Explainable AI**: Visualizes *why* a point is anomalous (deviation vs. isolation).
- **Research-Ready**: Includes performance metrics (F1-Score, AUC-ROC) and resource profiling.

---

## ✨ Key Features
✅ **Dual-Model Engine**: Combines Prophet (Seasonality/Trend) and Isolation Forest (Outlier Detection).
✅ **FastAPI Backend**: RESTful API for data generation, training, and inference.
✅ **Interactive Dashboard**: Real-time Plotly charts for time-series and anomaly scores.
✅ **Synthetic Data Generator**: Create realistic datasets for Finance, Server Metrics, and IoT Sensors.
✅ **Performance Metrics**: Auto-calculated Precision, Recall, F1-Score, and Accuracy.
✅ **Resource Profiling**: Track training time, memory usage, and throughput.
✅ **Export Capabilities**: Download predictions and reports as CSV.

---

## 🧩 Tech Stack

| Component | Technology | Description |
|-----------|------------|-------------|
| **Backend** | **FastAPI** | High-performance API for ML logic |
| **Frontend** | **Streamlit** | Interactive web dashboard |
| **ML Models** | **Prophet**, **Scikit-learn** | Forecasting & Anomaly Detection |
| **Data Processing** | **Pandas**, **NumPy** | Data manipulation & vectorization |
| **Visualization** | **Plotly** | Interactive charts |
| **Validation** | **Pydantic** | Data validation & schema management |

---

## 🚀 Getting Started

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/dheerajkumar47/Time-Series-Anomaly-Detection.git
cd Time-Series-Anomaly-Detection
```

### 2️⃣ Create Virtual Environment
```bash
python -m venv .venv
# Activate (Windows)
.venv\Scripts\activate
# Activate (Mac/Linux)
source .venv/bin/activate
```
│   ├── core/               # Core ML & Data Logic
│   │   ├── anomaly_detectors.py
│   │   ├── data_generator.py
│   │   └── utils.py
│   ├── main.py             # API Entry Point
│   └── schemas.py          # Pydantic Models
├── app.py                  # Streamlit Frontend
├── frontend_utils.py       # UI Helper Functions
├── requirements.txt        # Dependencies
└── README.md               # Documentation
```

---

## 🧮 Core Models

### 🔹 Prophet (Meta)
- **Type**: Forecasting / Regression
- **Use Case**: Detects anomalies by checking if values fall outside the predicted confidence interval.
- **Best For**: Data with strong seasonal patterns (e.g., daily sales, server traffic).

### 🔹 Isolation Forest
- **Type**: Unsupervised Learning
- **Use Case**: Isolates anomalies by randomly partitioning data points. Outliers require fewer partitions.
- **Best For**: High-dimensional data or irregular anomalies (e.g., fraud detection).

---

## 🌍 Real-World Applications
- **💰 Finance**: Fraud detection in credit card transactions.
- **📡 IoT**: Identifying sensor malfunctions or drift.
- **🖥️ DevOps**: Monitoring server CPU/Memory spikes.
- **🏥 Healthcare**: Detecting irregularities in patient vitals.

---

## 🔮 Future Scope
- [ ] **Deep Learning**: Integration with LSTM/Autoencoders.
- [ ] **AutoML**: Automatic model selection and hyperparameter tuning.
- [ ] **Streaming**: Real-time data ingestion via Kafka/WebSocket.
- [ ] **Deployment**: Docker containerization and Cloud deployment (AWS/GCP).

---

## 🤝 Contributing
Contributions are welcome! Please fork the repository and submit a pull request.
For major changes, please open an issue first to discuss what you would like to change.

---

## 🏅 Author
**Dheeraj Kumar**
- 📍 B.S. Software Engineering, Iqra University (Class of 2025)
- 📧 [dheerajkumar47@gmail.com](mailto:dheerajkumar47@gmail.com)
- 🔗 [LinkedIn](https://www.linkedin.com/in/dheeraj-kumar-b21a741a2/) | [GitHub](https://github.com/dheerajkumar47)

---

_If you find this project useful, please give it a ⭐ on GitHub!_
