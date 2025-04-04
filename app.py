import streamlit as st
import pandas as pd

# Load your data
df = pd.read_csv("clustered data.csv")  # Upload this file into Colab

st.title("Real-Time Cardiac Risk Monitoring")
st.write("This app simulates heart rate monitoring and predicts cardiac risk.")

# Show a sample
st.subheader("Sample Clustered Data (First 10 Rows)")
st.dataframe(df.head(10))

# Live Input
heart_rate = st.slider("Enter Heart Rate (BPM)", min_value=50, max_value=200, value=80)

# Prediction logic
def predict_risk(hr):
    return "High Risk" if hr > 100 else "Low Risk"

# Display Prediction
predicted_risk = predict_risk(heart_rate)
st.subheader("🩺 Live Heart Rate & Risk Prediction")
st.write(f"**Heart Rate (BPM):** {heart_rate}")
st.write(f"**Predicted Risk:** :red[{predicted_risk}]")
!streamlit run app.py & npx localtunnel --port 8501
