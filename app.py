import streamlit as st
import pandas as pd

# Page title
st.title(" Real-Time Cardiac Risk Monitoring") 
st.write("This app simulates heart rate monitoring and predicts cardiac risk based on input.")

# Upload clustered data
uploaded_file = st.file_uploader(" Upload your clustered CSV data", type="csv") 

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Show a sample
    st.subheader(" Sample Clustered Data (First 10 Rows)") 
    st.dataframe(df.head(10))

    # Live heart rate input
    heart_rate = st.slider(" Enter Heart Rate (BPM)", min_value=50, max_value=200, value=80) 

    # Prediction logic
    def predict_risk(hr):
        return "High Risk" if hr > 100 else "Low Risk"

    predicted_risk = predict_risk(heart_rate)

    # Display prediction
    st.subheader(" Live Heart Rate & Risk Prediction") 
    st.write(f"**Heart Rate (BPM):** {heart_rate}")
    st.write(f"**Predicted Risk:** :red[{predicted_risk}]")
else:
    st.warning(" Please upload a valid CSV file to proceed.")
