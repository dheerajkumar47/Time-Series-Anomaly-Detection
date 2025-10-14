
<<<<<<< HEAD
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](./LICENSE)
![Author: Dheeraj Kumar](https://img.shields.io/badge/Author-Dheeraj_Kumar-blue.svg)

**📊 Time-Series Anomaly Detection System**
=======
# 📊 Time-Series Anomaly Detection System
>>>>>>> eedf936dc42901a11bae209c74f309c1c1a3757b

A user-friendly **web application** for detecting anomalies in time-series data.
It provides an **interactive dashboard** to load data, configure models, visualize anomalies, and analyze performance.
This system is designed for **students, researchers, and engineers** who need to identify unusual behavior in financial transactions, server metrics, sensor data, or custom datasets.

---

## 🎯 Purpose & Motivation

* **Why this project?**
  Time-series data is everywhere: stock prices, server CPU usage, IoT sensors, health monitoring. Detecting anomalies is crucial for spotting fraud, preventing system failures, and ensuring data integrity.

* **The problem:** Traditional anomaly detection methods are either too rigid or too complex for quick prototyping.

* **Our solution:** An **intuitive Streamlit-based app** that combines **forecasting (Prophet)** and **machine learning (Isolation Forest)** in a single dashboard, making anomaly detection **accessible, explainable, and interactive.**

---

## ✨ Features

✅ **Interactive Dashboard** – Built with Streamlit for smooth exploration.
✅ **Flexible Data Input** – Upload CSV datasets or generate synthetic time-series data.
✅ **Multiple Models** – Prophet (trend-based) + Isolation Forest (outlier-based).
✅ **Dynamic Train/Test Splitting** – Auto/manual splitting with adjustable ratios.
✅ **Real-Time Visualizations** – Plot original series, detected anomalies, and scores.
✅ **Performance Metrics** – Precision, Recall, F1, Accuracy, ROC-AUC.
✅ **Profiling Tools** – Compare models on training time, prediction time, memory usage.
✅ **Export Options** – Download predictions, metrics, and profiling results in CSV format.

---

## 🚀 Getting Started

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/dheerajkumar47/Time-Series-Anomaly-Detection.git
cd Time-Series-Anomaly-Detection
```

### 2️⃣ Create & Activate Virtual Environment

```bash
# Create venv
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (Mac/Linux)
source .venv/bin/activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Run the Application

```bash
streamlit run app.py
```

Your default browser will open the dashboard automatically. 🎉

---

## 📂 Project Structure

```
├── app.py                  # Main Streamlit app (UI + workflow)
├── data_generator.py       # Synthetic data generation (financial, server, sensor)
├── anomaly_detectors.py    # Prophet & Isolation Forest + profiling/evaluation
├── utils.py                # Data preprocessing & visualization utilities
├── requirements.txt        # Python dependencies
└── README.md               # Documentation (this file)
```

---

## 💡 Core Concepts & Models

### 🔹 Prophet

* Developed by Facebook for forecasting.
* Captures **trends + seasonality** in time-series.
* Used as a **baseline** to detect anomalies as deviations from predicted values.

### 🔹 Isolation Forest

* Tree-based machine learning algorithm for outlier detection.
* Works by **isolating rare points** in fewer random splits.
* Great for **non-seasonal, high-dimensional anomaly detection.**

---

## 📸 Demo (Optional)

You can add screenshots/gifs here:

* Dashboard home
* Synthetic data generation
* Prophet vs Isolation Forest results
* Metrics comparison

---

## 🤝 Contributing

Contributions are welcome! 🚀

* Open an **issue** for bug reports or feature requests.
* Submit a **pull request** to improve the project.

---

## 🙏 Acknowledgements

* [Streamlit](https://streamlit.io/) – for building interactive apps.
* [Prophet](https://facebook.github.io/prophet/) – for time-series forecasting.
* [scikit-learn](https://scikit-learn.org/) – for Isolation Forest and metrics.
* [Plotly](https://plotly.com/python/) – for interactive visualizations.

---

👉 This project was built to make **time-series anomaly detection practical and explainable**, bridging the gap between research and real-world applications.

---

<<<<<<< HEAD
🙏 Acknowledgements
A special thanks to the creators of the Prophet, dtaianomaly, and Streamlit libraries for providing the powerful tools that made this project possible.

## License and Attribution
This repository is developed and maintained by **Dheeraj Kumar**.  
Licensed under the [MIT License](./LICENSE). Please give proper credit if you use or modify this project.
=======
>>>>>>> eedf936dc42901a11bae209c74f309c1c1a3757b
