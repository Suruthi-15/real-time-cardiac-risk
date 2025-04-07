# 💓 Cardiac Risk Prediction

This project is a web-based application built with **Streamlit** to monitor heart rate and predict **cardiac risk** in real-time using **Machine Learning**.

## 🚀 Features

- Upload clustered health data (CSV).
- Live heart rate input via slider.
- Predicts cardiac risk using a trained **Random Forest** model.
- Visualizations: Heart Rate vs Cluster (Altair Chart).
- Simple, interactive UI powered by Streamlit.

## 🧠 ML Model

- **Algorithm**: Random Forest Classifier
- **Input Feature**: Heart Rate (BPM)
- **Output**: Risk Label (High Risk / Low Risk)
- Data is pre-clustered using K-Means before prediction.

## 🖥️ App Preview

Live App: [Streamlit Cloud App](https://your-app-link.streamlit.app)  
(*Replace with your actual URL*)

## 📁 Files

- `app.py`: Main Streamlit application
- `clustered_data.csv`: Sample input data with heart rate and risk labels
- `README.md`: Project overview

## 🛠️ How to Run Locally

```bash
pip install streamlit pandas scikit-learn altair
streamlit run app.py
