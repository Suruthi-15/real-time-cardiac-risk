import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest


# Set page configuration
st.set_page_config(page_title="Cardiac Risk Predictor", layout="centered")

# Title
st.title("🫀 Cardiac Risk Prediction Using Clustering (Low, Moderate, High)")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("numeric_dataset.csv")
    return df

df = load_data()

# Selected features used for clustering
selected_features = ['Age', 'Weight', 'Height', 'Medical_Conditions', 'Medication',
                     'Smoker', 'Alcohol_Consumption', 'ECG', 'Calories_Intake', 'Water_Intake',
                     'Stress_Level', 'Mood', 'Muscle_Mass', 'Health_Score',
                     'Heart_Rate', 'Blood_Oxygen_Level', 'Body_Fat_Percentage']

# Extract feature data
X = df[selected_features]

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train KMeans model with 3 clusters
optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42)
kmeans.fit(X_scaled)

# Determine cluster-to-risk mapping based on cluster centers
cluster_risks = kmeans.cluster_centers_.mean(axis=1)
sorted_indices = np.argsort(cluster_risks)
risk_labels = {sorted_indices[0]: "Low Risk", 
               sorted_indices[1]: "Moderate Risk", 
               sorted_indices[2]: "High Risk"}

# Sidebar input mode
input_mode = st.sidebar.radio("Choose input mode:", ("Upload CSV File", "Enter Manually"))

if input_mode == "Upload CSV File":
    uploaded_file = st.file_uploader("Upload a CSV file with health data", type=["csv"])
    
    if uploaded_file:
        user_data = pd.read_csv(uploaded_file)

        if all(col in user_data.columns for col in selected_features):
            X_user = user_data[selected_features]
            X_user_scaled = scaler.transform(X_user)
            labels = kmeans.predict(X_user_scaled)
            risks = [risk_labels[label] for label in labels]
            user_data["Predicted_Cluster"] = labels
            user_data["Predicted_Risk"] = risks

            st.success("Prediction Completed:")
            st.dataframe(user_data[["Predicted_Cluster", "Predicted_Risk"]])
        else:
            st.error("Uploaded file missing required columns.")

else:
    st.subheader("Enter Health Data Manually")
    input_data = {}
    
    for feature in selected_features:
        input_data[feature] = st.number_input(f"{feature}", value=0.0)

    if st.button("Predict Risk"):
        input_df = pd.DataFrame([input_data])
        input_scaled = scaler.transform(input_df)
        label = kmeans.predict(input_scaled)[0]
        risk = risk_labels[label]

        st.success(f"Predicted Cluster: {label}")
        st.info(f"Predicted Cardiac Risk: **{risk}**")
       



