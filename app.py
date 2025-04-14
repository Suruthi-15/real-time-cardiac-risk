import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib

# Set page configuration
st.set_page_config(page_title="Cardiac Risk Predictor", layout="centered")

# Title
st.title("🫀 Cardiac Risk Prediction Using Clustering")

# Load data and model
@st.cache_data
def load_data():
    df = pd.read_csv("numeric_dataset.csv")
    return df

df = load_data()

# Selected features used for training
selected_features = ['Age', 'Weight', 'Height', 'Medical_Conditions', 'Medication',
                     'Smoker', 'Alcohol_Consumption', 'ECG', 'Calories_Intake', 'Water_Intake',
                     'Stress_Level', 'Mood', 'Muscle_Mass', 'Health_Score',
                     'Heart_Rate', 'Blood_Oxygen_Level', 'Body_Fat_Percentage']

# Extract feature data
X = df[selected_features]

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train KMeans model
optimal_k = 2  # Update this if you found a different optimal_k
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42)
kmeans.fit(X_scaled)

# Risk label mapping
risk_labels = {0: "Low Risk", 1: "High Risk"}

# Sidebar input mode
input_mode = st.sidebar.radio("Choose input mode:", ("Upload CSV File", "Enter Manually"))

if input_mode == "Upload CSV File":
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

    if uploaded_file:
        user_data = pd.read_csv(uploaded_file)

        # Ensure required columns are present
        if all(col in user_data.columns for col in selected_features):
            X_user = user_data[selected_features]
            X_user_scaled = scaler.transform(X_user)
            labels = kmeans.predict(X_user_scaled)

            # Map labels to risk
            risks = [risk_labels[label] for label in labels]
            user_data["Predicted_Cluster"] = labels
            user_data["Predicted_Risk"] = risks

            st.success("Cluster Prediction Completed:")
            st.write(user_data[["Predicted_Cluster", "Predicted_Risk"]])
        else:
            st.error("Uploaded CSV is missing one or more required features.")

else:
    st.subheader("Enter Health Data Manually")

    # Create input widgets for all selected features
    input_data = {}
    for feature in selected_features:
        input_data[feature] = st.number_input(f"{feature}", value=0.0)

    if st.button("Predict Risk"):
        single_df = pd.DataFrame([input_data])
        single_scaled = scaler.transform(single_df)
        label = kmeans.predict(single_scaled)[0]
        risk = risk_labels[label]

        st.success(f"Predicted Cluster: {label}")
        st.info(f"Predicted Cardiac Risk: **{risk}**")
