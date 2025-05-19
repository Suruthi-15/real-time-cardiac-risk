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
        
from sklearn.ensemble import IsolationForest

# Divider
st.markdown("---")
st.header("🔍 Anomaly Detection on Health Data")

st.markdown("Upload a **CSV file** to detect abnormal health patterns (e.g., abnormal heart rate, age, weight, etc.).")

# File uploader for anomaly detection
anomaly_file = st.file_uploader("Upload CSV for Anomaly Detection", type=["csv"], key="anomaly_file")

if anomaly_file:
    try:
        anomaly_df = pd.read_csv(anomaly_file)
        st.subheader("Uploaded Data for Anomaly Detection")
        st.dataframe(anomaly_df.head())

        # Select features for anomaly detection
        st.markdown("### Select Features for Anomaly Detection")
        anomaly_features = st.multiselect("Choose numeric features", anomaly_df.columns.tolist(), default=['Heart_Rate', 'Age', 'Weight'])

        if len(anomaly_features) >= 2:
            X_anomaly = anomaly_df[anomaly_features].dropna()
            scaler_anomaly = StandardScaler()
            X_anomaly_scaled = scaler_anomaly.fit_transform(X_anomaly)

            # Isolation Forest
            iso_model = IsolationForest(contamination=0.05, random_state=42)
            anomaly_labels = iso_model.fit_predict(X_anomaly_scaled)

            # Attach labels
            result_df = anomaly_df.loc[X_anomaly.index].copy()
            result_df["Anomaly_Label"] = np.where(anomaly_labels == -1, "Anomaly", "Normal")

            st.markdown("### 🔬 Anomaly Detection Results")
            st.dataframe(result_df[anomaly_features + ["Anomaly_Label"]])

            # Show only anomalies
            st.markdown("### ⚠ Detected Anomalies")
            anomalies_only = result_df[result_df["Anomaly_Label"] == "Anomaly"]
            if not anomalies_only.empty:
                st.dataframe(anomalies_only[anomaly_features + ["Anomaly_Label"]])
            else:
                st.success("No anomalies detected.")
        else:
            st.warning("Please select at least two features.")
    except Exception as e:
        st.error(f"Error processing file: {e}")

       



