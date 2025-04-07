import streamlit as st
import pandas as pd

# Page title
st.title("💓 Real-Time Cardiac Risk Monitoring")
st.write("This app uses your pre-clustered data to simulate cardiac risk based on heart rate.")

# Load data from local file (must be in the repo)
try:
    df = pd.read_csv("clustered_data.csv")

    # Show sample
    st.subheader("📊 Sample Clustered Data (First 10 Rows)")
    st.dataframe(df.head(10))

    # Heart rate input
    heart_rate = st.slider("💓 Enter Heart Rate (BPM)", min_value=50, max_value=200, value=80)

    # Prediction logic
    def predict_risk(hr):
        return "High Risk" if hr > 100 else "Low Risk"

    predicted_risk = predict_risk(heart_rate)

    # Display result
    st.subheader("🩺 Risk Prediction Result")
    st.write(f"**Heart Rate:** {heart_rate} BPM")
    st.write(f"**Predicted Risk:** :red[{predicted_risk}]")

except FileNotFoundError:
    st.error("❌ clustered_data.csv not found. Please upload it to your GitHub repo.")

