import streamlit as st
import joblib
import numpy as np
import pandas as pd

# -----------------------------
# Load saved model + label encoder + feature order
# -----------------------------
model = joblib.load("ensemble_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")
feature_columns = joblib.load("feature_columns.pkl")

st.title("Drug Category Prediction App")
st.write("Enter the person's details below to predict the drug category.")

# -----------------------------
# User Input Fields
# -----------------------------
def get_user_input():
    Age = st.number_input("Age", min_value=10, max_value=100, value=25)

    Gender = st.selectbox("Gender", [0, 1])  
    Education = st.selectbox("Education Level", [0,1,2,3,4,5,6])
    Country = st.selectbox("Country Code", [0,1,2,3,4,5,6])
    Ethnicity = st.selectbox("Ethnicity Code", [0,1,2,3,4,5])

    Nscore = st.number_input("Nscore", value=10.0)
    Escore = st.number_input("Escore", value=10.0)
    Oscore = st.number_input("Oscore", value=10.0)
    Ascore = st.number_input("Ascore", value=10.0)
    Cscore = st.number_input("Cscore", value=10.0)

    Impulsive = st.number_input("Impulsive", value=5.0)
    SS = st.number_input("SS (Sensation Seeking)", value=5.0)

    # Return raw values
    return {
        "Age": Age,
        "Gender": Gender,
        "Education": Education,
        "Country": Country,
        "Ethnicity": Ethnicity,
        "Nscore": Nscore,
        "Escore": Escore,
        "Oscore": Oscore,
        "Ascore": Ascore,
        "Cscore": Cscore,
        "Impulsive": Impulsive,
        "SS": SS
    }

# Get user input
input_data = get_user_input()

# -----------------------------
# Feature Engineering (same as training)
# -----------------------------
def create_features(data):
    df = pd.DataFrame([data])

    df["Impulsive_SS"] = df["Impulsive"] * df["SS"]
    df["Nscore_Escore"] = df["Nscore"] * df["Escore"]
    df["Impulsive_squared"] = df["Impulsive"] ** 2
    df["SS_squared"] = df["SS"] ** 2

    # Reorder to match training
    df = df[feature_columns]

    return df

# Convert input to model-ready format
processed_data = create_features(input_data)

# -----------------------------
# Predict Button
# -----------------------------
if st.button("Predict Drug Category"):
    prediction_encoded = model.predict(processed_data)[0]
    prediction_label = label_encoder.inverse_transform([prediction_encoded])[0]

    st.success(f"Predicted Drug Category: **{prediction_label}**")
