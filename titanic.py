import streamlit as st
import numpy as np
import joblib

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def predict_proba(X, w, b):
    z_wb = np.dot(X, w) + b
    return sigmoid(z_wb)

# Load pre-trained model & scaler
w, b = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🚢 Titanic Survival Prediction (Custom Logistic Regression)")
st.write("Enter passenger details below to check survival probability:")

# Inputs
age = st.number_input("Age", min_value=0, max_value=100, value=25)

gender_label = st.selectbox("Gender", ["Male", "Female"])
gender = 1 if gender_label == "Male" else 0

ticket_label = st.selectbox("Ticket Class", ["1st Class", "2nd Class", "3rd Class"])
ticket_class = {"1st Class": 1, "2nd Class": 2, "3rd Class": 3}[ticket_label]

# Prediction
if st.button("Predict"):
    X = np.array([[age, gender, ticket_class]])
    X_scaled = scaler.transform(X)
    proba = float(predict_proba(X_scaled, w, b)[0])

    st.metric("Survival Probability", f"{proba*100:.2f}%")
    if proba >= 0.5:
        st.success("✅ Passenger is predicted to SURVIVE")
    else:
        st.error("❌ Passenger is predicted to NOT SURVIVE")
