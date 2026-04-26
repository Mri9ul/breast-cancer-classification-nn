import streamlit as st
import pandas as pd
import numpy as np
import pickle
from tensorflow import keras

# Load model and scaler
model = keras.models.load_model("breast_cancer_nn_model.keras")

with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

st.set_page_config(
    page_title="Breast Cancer Classification",
    page_icon="🧬"
)

st.title("🧬 Breast Cancer Classification using Neural Network")

st.warning(
    "This app is for educational purposes only and should not be used for real medical diagnosis."
)

st.write(
    "Upload a CSV file containing the 30 diagnostic features. "
    "The model will classify each record as Benign or Malignant."
)

# Expected feature columns
expected_columns = [
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
    "compactness_mean", "concavity_mean", "concave points_mean", "symmetry_mean",
    "fractal_dimension_mean", "radius_se", "texture_se", "perimeter_se", "area_se",
    "smoothness_se", "compactness_se", "concavity_se", "concave points_se",
    "symmetry_se", "fractal_dimension_se", "radius_worst", "texture_worst",
    "perimeter_worst", "area_worst", "smoothness_worst", "compactness_worst",
    "concavity_worst", "concave points_worst", "symmetry_worst",
    "fractal_dimension_worst"
]

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Data Preview")
    st.dataframe(data.head())

    # Remove unnecessary columns if present
    if "id" in data.columns:
        data = data.drop(columns=["id"])

    if "diagnosis" in data.columns:
        data = data.drop(columns=["diagnosis"])

    # Check missing columns
    missing_columns = [col for col in expected_columns if col not in data.columns]

    if missing_columns:
        st.error("The uploaded file is missing required columns:")
        st.write(missing_columns)
    else:
        # Keep only expected columns in correct order
        input_data = data[expected_columns]

        # Scale data
        input_scaled = scaler.transform(input_data)

        # Predict
        predictions = model.predict(input_scaled)
        predicted_labels = np.argmax(predictions, axis=1)

        # Convert labels
        result_labels = [
            "Benign" if label == 1 else "Malignant"
            for label in predicted_labels
        ]

        prediction_confidence = np.max(predictions, axis=1)

        result_df = data.copy()
        result_df["Prediction"] = result_labels
        result_df["Confidence"] = prediction_confidence

        st.subheader("Prediction Results")
        st.dataframe(result_df)

        csv = result_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download Predictions as CSV",
            data=csv,
            file_name="breast_cancer_predictions.csv",
            mime="text/csv"
        )