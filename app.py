import streamlit as st
import pandas as pd
import altair as alt
import joblib
import os

# Set page configuration
st.set_page_config(page_title="Real-Time Cardiac Risk Prediction", layout="centered")

# Title and Introduction
st.title(" Cardiac Risk Prediction")
st.markdown("""
This project is a web-based application built with **Streamlit** to monitor heart rate and predict **cardiac risk** in real-time using **Machine Learning**.
""")

# Features Section
st.markdown("##  Features")
st.markdown("""
-  Upload clustered health data (CSV)
-  Live heart rate input via slider
-  Predict cardiac risk using a trained **Random Forest** model
-  Visualizations: Heart Rate vs Cluster (Altair)
-  Simple, interactive UI powered by Streamlit
""")

# Upload CSV file
st.markdown("##  Upload Clustered Data")
uploaded_file = st.file_uploader("Upload your clustered CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Sample Clustered Data (First 10 Rows)")
    st.dataframe(df.head(10))

    # Visualization - Heart Rate vs Cluster
    if "Heart Rate" in df.columns and "Cluster" in df.columns:
        st.subheader(" Heart Rate vs Cluster Visualization")
        chart = alt.Chart(df.head(50)).mark_circle(size=60).encode(
            x="Heart Rate",
            y="Cluster",
            color="Cluster:N",
            tooltip=["Heart Rate", "Cluster"]
        ).interactive()
        st.altair_chart(chart, use_container_width=True)
    else:
        st.warning("Columns 'Heart Rate' and 'Cluster' not found in uploaded CSV.")

# Load ML model
st.markdown("##  ML Model Prediction")
model_path = "random_forest_model.pkl"

if os.path.exists(model_path):
    model = joblib.load(model_path)

    # Input heart rate
    heart_rate = st.slider("Select Heart Rate", min_value=40, max_value=180, value=75, step=1)
    user_input = pd.DataFrame({"Heart Rate": [heart_rate]})

    # Predict
    prediction = model.predict(user_input)[0]
    risk_label = "High Risk" if prediction == 1 else "Low Risk"

    # Output
    st.subheader Prediction Result")
    st.write(f"**Predicted Cardiac Risk Level:** `{risk_label}`")

else:
    st.error("⚠ Random Forest model file not found. Please ensure 'random_forest_model.pkl' is in the app directory.")
