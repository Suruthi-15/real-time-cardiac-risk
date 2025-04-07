import streamlit as st
import pandas as pd
import altair as alt

# Page title
st.title(" Real-Time Cardiac Risk Monitoring")
st.write("This app uses heart rate data to predict cardiac risk.")

# Load base clustered data (optional preview)
try:
    df = pd.read_csv("clustered_data.csv")

    st.subheader(" Sample Clustered Data (First 10 Rows)")
    st.dataframe(df.head(10))

    # Bar chart: Heart Rate vs. Cluster
    st.subheader(" Heart Rate vs Cluster (First 50 Rows)")
    chart = alt.Chart(df.head(50)).mark_bar().encode(
        x='Heart_Rate',
        y='Cluster:O',
        color='Cluster:N',
        tooltip=['Heart_Rate', 'Cluster']
    ).interactive()
    st.altair_chart(chart, use_container_width=True)

except FileNotFoundError:
    st.warning(" clustered_data.csv not found, skipping sample preview.")

# Upload new data
st.subheader(" Upload New Heart Rate Data")
uploaded_file = st.file_uploader("Upload a CSV file with a 'Heart_Rate' column")

if uploaded_file:
    new_data = pd.read_csv(uploaded_file)

    st.write(" Uploaded Data Preview:")
    st.dataframe(new_data.head())

    # Predict based on heart rate
    new_data["Predicted_Cluster"] = new_data["Heart_Rate"].apply(lambda x: 1 if x < 100 else 0)
    new_data["Risk_Level"] = new_data["Heart_Rate"].apply(lambda x: "High Risk" if x < 100 else "Low Risk")

    # Show prediction results
    st.subheader(" Prediction Results")
    st.dataframe(new_data)

    # Download button
    csv = new_data.to_csv(index=False).encode('utf-8')
    st.download_button(" Download Predictions as CSV", data=csv, file_name="predicted_results.csv", mime="text/csv")

    # Real-time visualization
    st.subheader(" Risk Levels by Heart Rate")
    chart = alt.Chart(new_data).mark_bar().encode(
        x='Heart_Rate:Q',
        color='Risk_Level:N',
        tooltip=['Heart_Rate', 'Risk_Level']
    ).interactive()
    st.altair_chart(chart, use_container_width=True)

# Notes
st.markdown("##  Cardiac Risk Prediction Notes")
st.markdown(":red[⚠ High Risk means immediate medical attention may be required.]")
st.markdown(":green[ Low Risk is considered safe but stay monitored.]")
