
**📊 Time-Series Anomaly Detection System**

A powerful and user-friendly web application for detecting anomalies in time-series data. This system provides an intuitive dashboard for data loading, model configuration, and performance analysis, allowing users to identify unusual patterns in various datasets, from financial transactions to server metrics.

✨ Features
📈 Interactive Dashboard: A comprehensive Streamlit dashboard for a seamless user experience.


📂 Flexible Data Loading: Easily upload your own CSV files or generate synthetic datasets with customizable parameters.


🤖 Multiple Model Support: Compare the performance of specialized anomaly detection algorithms like Prophet (for trend-based anomalies) and Isolation Forest (a versatile generalist model).



✂️ Dynamic Data Splitting: Automatically or manually split your dataset into training and testing sets based on a user-defined proportion.


📊 Real-Time Visualization: View original data, predicted anomalies, and model performance metrics through interactive plots.



⏱️ Performance Profiling: Analyze and compare model performance with metrics like training time, prediction time, and memory consumption.


🚀 How to Run the Application
This project is built with Python and Streamlit. Follow these simple steps to get the application running on your local machine.

Step 1: Clone the Repository
Bash

git clone https://github.com/dheerajkumar47/Time-Series-Anomaly-Detection.git)
cd your-repository-name
(Replace with your actual GitHub repository URL and folder name)

Step 2: Set Up a Virtual Environment (Recommended)
It's a best practice to use a virtual environment to manage project dependencies.

Bash

python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
Step 3: Install Dependencies
Install all the required libraries using the 

requirements.txt file.

Bash

pip install -r requirements.txt
Step 4: Run the App
Launch the Streamlit dashboard from your terminal.

Bash

streamlit run app.py
Your default web browser should open automatically with the application running.

📂 Project Structure
├── app.py                     # Main Streamlit application file
├── data_generator.py          # Module for generating synthetic time-series data
├── anomaly_detectors.py       # Contains the anomaly detection models (Prophet, Isolation Forest) and profiler
├── utils.py                   # Utility functions for data preprocessing and visualization
├── requirements.txt           # List of project dependencies
└── README.md                  # This file
💡 Core Concepts & Models
This project utilizes a specialized approach to time-series anomaly detection by focusing on domain-specific frameworks.


Prophet: A forecasting tool developed by Facebook, it excels at modeling time-series data with strong seasonal components and trends. It's used as a baseline model in this project to quickly identify anomalies based on deviations from predicted forecasts.


Isolation Forest: A versatile and robust anomaly detection algorithm that works by building a forest of random isolation trees. It's highly effective at identifying outliers by isolating them from the rest of the data.

📸 Demo
🤝 Contributing
Contributions are welcome! If you have suggestions for new features or find a bug, please open an issue or submit a pull request.


🙏 Acknowledgements
A special thanks to the creators of the Prophet, dtaianomaly, and Streamlit libraries for providing the powerful tools that made this project possible.
