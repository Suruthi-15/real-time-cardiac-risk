import streamlit as st
import pandas as pd
import altair as alt

# Title and description
st.set_page_config(page_title="Cardiac Risk Monitor", page_icon="")
st.title(" Real-Time Cardiac Risk Monitoring")
st.write("This app uses your pre-clustered data to simulate cardiac risk based on heart rate.")

try:
    # Load initial data
    df = pd.read_csv("clustered_data.csv")

    st.subheader(" Sample Clustered Data (First 10 Rows)")
    st.dataframe(df.head(10))

    # Heart rate input
    heart_rate = st.slider(" Enter Heart Rate (BPM)", min_value=50, max_value=200, value=80)

    # Predict risk based on rule
    if heart_rate < 100:
        predicted_cluster = 1
        risk_level = "High Risk"
    else:
        predicted_cluster = 0
        risk_level = "Low Risk"

    st.subheader(" Risk Prediction Result")
    st.write(f"**Heart Rate:** {heart_rate} BPM")
    st.markdown(f"###  Prediction: **{risk_level}** (Cluster {predicted_cluster})")

    # Chart: Heart Rate vs. Cluster
    st.subheader(" Heart Rate vs Cluster (First 50 Rows)")
    chart = alt.Chart(df.head(50)).mark_bar().encode(
        x='Heart_Rate',
        y='Cluster:O',
        color='Cluster:N',
        tooltip=['Heart_Rate', 'Cluster']
    ).interactive()
    st.altair_chart(chart, use_container_width=True)

    # File upload section
    st.subheader(" Upload New Heart Rate Data")
    uploaded_file = st.file_uploader("Upload a CSV file with a 'Heart_Rate' column")

    if uploaded_file:
        new_data = pd.read_csv(uploaded_file)
        st.write(" Uploaded Data Preview:")
        st.dataframe(new_data.head())

        # Predict for uploaded data
        new_data["Predicted_Cluster"] = new_data["Heart_Rate"].apply(lambda x: 1 if x < 100 else 0)
        new_data["Risk_Level"] = new_data["Heart_Rate"].apply(lambda x: "High Risk" if x < 100 else "Low Risk")

        st.subheader(" Prediction Results")
        st.dataframe(new_data)

    # Styling & final notes
    st.markdown("##  Cardiac Risk Prediction Notes")
    st.markdown(":red[⚠️ High Risk means immediate medical attention may be required.]")
    st.markdown(":green[✅ Low Risk is considered safe but stay monitored.]")

except FileNotFoundError:
    st.error(" clustered_data.csv not found. Please upload it to your GitHub repo.")
