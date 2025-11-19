# 📈 Time-Series Anomaly Detection System | AI-Powered Analytics

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](./LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED.svg)](https://www.docker.com/)
[![Status](https://img.shields.io/badge/Project_Status-Active-brightgreen.svg)]()

> **A production-ready, Dockerized Machine Learning application for detecting anomalies in time-series data using Prophet and Isolation Forest.**

---

## 📖 Table of Contents
- [Overview](#-overview)
- [Key Features](#-key-features)
- [Tech Stack](#-tech-stack)
- [Installation & Setup](#-installation--setup)
    - [Using Docker (Recommended)](#option-1-run-with-docker-recommended)
    - [Manual Installation](#option-2-manual-installation)
- [How It Works](#-how-it-works)
- [API Documentation](#-api-documentation)
- [Future Roadmap](#-future-roadmap)
- [Contributing](#-contributing)
- [Author](#-author)

---

## 🧠 Overview

The **Time-Series Anomaly Detection System** is a robust, full-stack AI application designed to identify irregularities in sequential data. Whether you are monitoring **server metrics (CPU/Memory)**, **financial transactions**, or **IoT sensor readings**, this system provides a powerful, explainable, and interactive interface to detect outliers.

Unlike simple scripts, this project features a **decoupled architecture**:
*   **Backend**: A high-performance **FastAPI** server that handles data processing, model training, and inference.
*   **Frontend**: A modern **Streamlit** dashboard for data visualization and user interaction.

This project is perfect for developers, data scientists, and researchers looking for a **scalable template** for anomaly detection tasks.

---

## ✨ Key Features

*   **🚀 Dual-Model Engine**:
    *   **Prophet (by Meta)**: Best for forecasting and detecting trend/seasonality breaks.
    *   **Isolation Forest**: Unsupervised learning algorithm ideal for high-dimensional outlier detection.
*   **🐳 Fully Dockerized**: One-command deployment using `docker-compose`.
*   **📊 Interactive Dashboard**:
    *   Real-time **Plotly** charts.
    *   Tabbed interface for Data, Training, and Analysis.
    *   Custom CSS styling for a professional look.
*   **📉 Synthetic Data Generator**: Built-in tool to generate realistic test datasets (Financial, Server, Sensor).
*   **📈 Performance Metrics**: Automatically calculates **Precision**, **Recall**, **F1-Score**, and **AUC-ROC**.
*   **⚡ Resource Profiling**: Tracks training time and memory usage for optimization.
*   **🔌 RESTful API**: Well-documented endpoints for integration with other systems.

---

## 🧩 Tech Stack

| Component | Technology | Purpose |
| :--- | :--- | :--- |
| **Language** | **Python 3.10+** | Core programming language |
| **Backend** | **FastAPI** | High-speed web framework for building APIs |
| **Frontend** | **Streamlit** | Rapid UI development for Machine Learning |
| **Containerization** | **Docker & Docker Compose** | Consistent deployment across environments |
| **ML Libraries** | **Prophet, Scikit-learn** | Time-series forecasting and anomaly detection |
| **Data Processing** | **Pandas, NumPy** | Data manipulation and vectorization |
| **Visualization** | **Plotly** | Interactive, zoomable graphing |
| **Validation** | **Pydantic** | Data validation and settings management |

---

## 🛠 Installation & Setup

### Option 1: Run with Docker (Recommended)
This is the easiest way to get started. No need to install Python or libraries manually.

1.  **Prerequisites**: Install [Docker Desktop](https://www.docker.com/products/docker-desktop/).
2.  **Clone the repo**:
    ```bash
    git clone https://github.com/dheerajkumar47/Time-Series-Anomaly-Detection.git
    cd Time-Series-Anomaly-Detection
    ```
3.  **Run the app**:
    ```bash
    docker-compose up --build
    ```
4.  **Access the Dashboard**: Open [http://localhost:8501](http://localhost:8501) in your browser.

### Option 2: Manual Installation
If you prefer running it locally without Docker:

**1. Backend Setup**
```bash
cd backend
pip install -r ../requirements.txt
uvicorn main:app --reload --port 8000
```

**2. Frontend Setup** (Open a new terminal)
```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 🔍 How It Works

1.  **Data Ingestion**: Upload a CSV file or generate synthetic data via the UI.
2.  **Preprocessing**: The backend validates the schema and handles missing values.
3.  **Model Training**:
    *   **Prophet**: Learns the seasonality (daily, weekly, yearly) and trend.
    *   **Isolation Forest**: Randomly partitions data to isolate anomalies.
4.  **Inference**: The models predict anomalies on the test set.
5.  **Visualization**: The frontend fetches results and displays interactive plots and metrics.

---

## 📚 API Documentation

Once the backend is running, you can access the interactive API docs (Swagger UI) at:
👉 **[http://localhost:8000/docs](http://localhost:8000/docs)**

**Key Endpoints:**
*   `POST /data/generate`: Generate synthetic time-series data.
*   `POST /models/train`: Train anomaly detection models.
*   `GET /models/results`: Get prediction results.
*   `GET /models/metrics`: Get evaluation metrics (F1, Accuracy, etc.).

---

## 🔮 Future Roadmap

*   [ ] **Database Integration**: Add PostgreSQL/SQLite for persistent storage.
*   [ ] **Authentication**: Implement JWT (JSON Web Tokens) for secure API access.
*   [ ] **Deep Learning**: Integrate LSTM and Autoencoders for complex pattern recognition.
*   [ ] **Cloud Deployment**: CI/CD pipelines for AWS/GCP deployment.

---

## 🤝 Contributing

Contributions are always welcome!
1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/AmazingFeature`).
3.  Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4.  Push to the branch (`git push origin feature/AmazingFeature`).
5.  Open a Pull Request.

---

## 🏅 Author

**Dheeraj Kumar**
*   🎓 **Software Engineering Student** at Iqra University
*   📧 **Email**: [dheerajkumar47@gmail.com](mailto:dheerajkumar47@gmail.com)
*   🔗 **LinkedIn**: [Dheeraj Kumar](https://www.linkedin.com/in/dheeraj-kumar-b21a741a2/)
*   🐙 **GitHub**: [@dheerajkumar47](https://github.com/dheerajkumar47)

---

### ⭐ Show your support
Give a ⭐️ if this project helped you!
