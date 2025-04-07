import streamlit as st
import pandas as pd
import altair as alt

# Set page configuration
st.set_page_config(page_title="Real-Time Cardiac Risk Prediction", layout="centered")

# Title and Introduction
st.title(" Real-Time Cardiac Risk Prediction")
st.markdown("""
This is a web-based app built with **Streamlit** to monitor heart rate and predict **cardiac risk** in real-time.
""")

# Features Section
st.markdown("##  Features")
st.markdown("""
- Upload clustered health data (CSV)
- Live heart rate input via slider
- Predict cardiac risk using simple logic (threshold-based)
- Visualization: Heart Rate vs Cluster (Altair chart)
""")

# Upload CSV file
st.markdown("##  Upload Clustered Data")
uploaded_file = st.file_uploader("Upload your clustered CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader(" Sample Clustered Data (First 10 Rows)")
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
        st.warning("⚠ Columns 'Heart Rate' and 'Cluster' not found in uploaded CSV.")

# Live heart rate input and prediction
st.markdown("##  Live Heart Rate Monitoring")
heart_rate = st.slider("Select Heart Rate (BPM)", min_value=40, max_value=180, value=75)

# Risk logic based on threshold
def predict_risk(hr):
    return "High Risk" if hr > 100 else "Low Risk"

risk_label = predict_risk(heart_rate)

# Show result
st.subheader("🩺 Prediction Result")
st.write(f"**Heart Rate:** {heart_rate} BPM")
st.write(f"**Predicted Cardiac Risk:** :red[{risk_label}]")

