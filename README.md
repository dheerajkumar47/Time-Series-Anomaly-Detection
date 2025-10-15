[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](./LICENSE)
![Author: Dheeraj Kumar](https://img.shields.io/badge/Author-Dheeraj_Kumar-blue.svg)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![Streamlit](https://img.shields.io/badge/Framework-Streamlit-red.svg)
![Machine Learning](https://img.shields.io/badge/ML-Isolation%20Forest-orange.svg)
![Forecasting](https://img.shields.io/badge/Forecasting-Prophet-yellow.svg)
![Status](https://img.shields.io/badge/Project_Status-Active-brightgreen.svg)
![Platform](https://img.shields.io/badge/Platform-Web_App-blueviolet.svg)

---

# 📊 Time-Series Anomaly Detection System  
### _An Interactive, Explainable, and Research-Ready Dashboard for Detecting Anomalies in Time-Series Data_

---

## 🧠 Overview  

This project is a **Streamlit-based AI system** designed to **detect anomalies in time-series datasets** using **Prophet (forecasting)** and **Isolation Forest (machine learning)**.  

It offers an **intuitive, no-code web dashboard** where users can:
- Upload or generate data,  
- Apply multiple models,  
- Visualize anomalies in real time,  
- Compare performance metrics, and  
- Export detailed analysis results.  

The system is ideal for **researchers, students, and engineers** working in areas like **finance, IoT, network security, and predictive analytics**.  

---

## 🎯 Motivation  

Every dataset tells a story — but sometimes, the anomalies are the most important parts.  

From **financial frauds** and **server failures** to **sensor faults** and **health monitoring**, anomalies can reveal early signs of problems.  
However, most existing tools are either too rigid, require heavy coding, or lack visual explainability.  

This project aims to **simplify anomaly detection** — combining **forecasting + machine learning** into one interactive platform.

---

## ✨ Key Features  

✅ **Streamlit-Powered Interactive UI** – Real-time results, no coding needed.  
✅ **Multiple Model Support** – Prophet (trend-based) + Isolation Forest (statistical outlier).  
✅ **Synthetic Data Generator** – Create time-series for testing or benchmarking.  
✅ **Flexible Input Options** – Upload CSV datasets or use demo data.  
✅ **Dynamic Train/Test Splitting** – Manual or automatic ratio adjustment.  
✅ **Advanced Metrics Dashboard** – Accuracy, Precision, Recall, F1, ROC-AUC.  
✅ **Performance Comparison** – See which model performs better and faster.  
✅ **Plotly Visualizations** – Clean, zoomable, and interactive anomaly charts.  
✅ **Exportable Results** – Save predictions and metrics for academic reporting.  
✅ **Research-Ready Code** – Modular structure for customization and paper replication.  

---

## 🧩 Tech Stack  

| Category | Technology |
|-----------|-------------|
| **Frontend / UI** | Streamlit |
| **Machine Learning** | Scikit-learn (Isolation Forest) |
| **Forecasting** | Prophet |
| **Data Handling** | Pandas, NumPy |
| **Visualization** | Plotly |
| **Language** | Python 3.10+ |
| **Version Control** | Git & GitHub |

---

## 🚀 Getting Started  

### 1️⃣ Clone the Repository  
```bash
git clone https://github.com/dheerajkumar47/Time-Series-Anomaly-Detection.git
cd Time-Series-Anomaly-Detection

### 2️⃣ Create Virtual Environment
```bash
python -m venv .venv
# Activate (Windows)
.venv\Scripts\activate
# Activate (Mac/Linux)
source .venv/bin/activate

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt

### 4️⃣ Run the Application
```bash
streamlit run app.py

Once executed, the dashboard automatically opens in your browser. 🎉

### Project Structure

├── app.py                  # Main Streamlit app (UI + logic)
├── data_generator.py       # Synthetic data creation (financial, IoT, system)
├── anomaly_detectors.py    # Prophet & Isolation Forest models
├── utils.py                # Helper functions (plots, evaluation, pre-processing)
├── requirements.txt        # Dependency list
├── LICENSE                 # MIT License
└── README.md               # Documentation (this file)

### 🧮 Core Models
🔹** Prophet**

Developed by Meta (Facebook).

Forecasts trends + seasonality using additive models.

Detects anomalies by comparing predictions vs. observed data.

Ideal for business, environmental, and sensor data.

🔹** Isolation Forest**

Tree-based unsupervised algorithm for outlier detection.

Identifies points that are “isolated” from the rest of the dataset.

Works well on non-periodic or high-dimensional data.

**📸 Demo Screenshots**
<img width="1920" height="1403" alt="screencapture-time-series-anomaly-detection-6sdvf5vvcvzngp2fj48nvv-streamlit-app-2025-10-16-01_02_49" src="https://github.com/user-attachments/assets/a8069819-7387-4f00-81f4-52f3fcf7005b" />
<img width="1920" height="3550" alt="screencapture-time-series-anomaly-detection-6sdvf5vvcvzngp2fj48nvv-streamlit-app-2025-10-16-01_03_29" src="https://github.com/user-attachments/assets/dc282184-6d7d-41a1-87f1-0d08edb06d76" />



**🧪 Evaluation Metrics**

The app automatically calculates:

Accuracy
Precision
Recall
F1 Score
ROC-AUC
Execution Time
Memory Usage

**🌍 Real-World Applications**

✅ Finance: Fraud and transaction anomaly detection.
✅ IoT Systems: Sensor drift and malfunction alerts.
✅ Cybersecurity: Server activity and network traffic anomalies.
✅ Healthcare: Abnormal pattern detection in vitals or wearable data.
✅ Industry 4.0: Predictive maintenance and process monitoring.

**🔮 Future Scope**

**🚧 Planned improvements:**

Integration with LSTM (Deep Learning) for sequence prediction.

Support for auto-ML model selection.

Addition of real-time streaming data analysis.

Integration with cloud dashboards (AWS, GCP).

Export PDF reports of anomaly findings for research papers.

**🤝 Contributing**

Contributions are welcome!

Open an issue for bug reports or feature requests.

Submit a pull request for improvements.

**🏅 Author & Attribution**

Developed and maintained by **Dheeraj Kumar**
📍 B.S. Software Engineering, Iqra University (Class of 2025)
🎓 Focus Areas: AI • Machine Learning • Data Analysis • Research Systems

If you use this repository, please credit the author by citing or linking back.
Licensed under the MIT License
.

**💬 About the Developer**

I’m Dheeraj Kumar — a Software Engineering graduate passionate about AI, ML, and research-driven software systems.
This project represents my goal to combine academic learning with real-world AI applications — creating tools that are not just smart but understandable and usable.

📫 Contact:

LinkedIn
    https://www.linkedin.com/in/dheeraj-kumar-b21a741a2/
GitHub

📧 dheerajkumar47@gmail.com

**🙏 Acknowledgements**

Special thanks to the developers and communities behind:

Streamlit
Prophet
Scikit-learn
Plotly

**⭐ If you find this project helpful, please give it a star on GitHub — it motivates and supports future open-source research!**



