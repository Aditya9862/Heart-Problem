# streamlit_app.py
import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(layout="centered", page_title="Heart Disease Predictor")

st.title("Heart Disease Predictor")
st.write("Enter patient data and get a prediction (binary classifier: 1 indicates presence of heart disease).")

# Load model
model = joblib.load('heart_model.joblib')

# Input widgets - adapt these to match your dataset columns
age = st.number_input("Age", min_value=1, max_value=120, value=55)
sex = st.selectbox("Sex (1 = male, 0 = female)", [1, 0], index=0)
cp = st.selectbox("Chest pain type (0-3)", [0,1,2,3], index=1)
trestbps = st.number_input("Resting blood pressure", value=130)
chol = st.number_input("Serum cholesterol (mg/dl)", value=250)
fbs = st.selectbox("Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)", [0,1], index=0)
restecg = st.selectbox("Resting ECG (0,1,2)", [0,1,2], index=0)
thalach = st.number_input("Max heart rate achieved", value=150)
exang = st.selectbox("Exercise-induced angina (1 = yes, 0 = no)", [0,1], index=0)
oldpeak = st.number_input("ST depression induced by exercise", value=1.0, step=0.1)
slope = st.selectbox("Slope of peak exercise ST segment (0-2)", [0,1,2], index=1)
ca = st.selectbox("Number of major vessels (0-3) colored by fluoroscopy", [0,1,2,3], index=0)
thal = st.selectbox("Thal (1 = normal; 2 = fixed defect; 3 = reversible defect)", [1,2,3], index=1)

# Prepare input
input_dict = {
    'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 'chol': chol,
    'fbs': fbs, 'restecg': restecg, 'thalach': thalach, 'exang': exang,
    'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal
}
input_df = pd.DataFrame([input_dict])

if st.button("Predict"):
    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0,1] if hasattr(model, "predict_proba") else None
    if pred == 1:
        st.error(f"Model predicts HEART DISEASE (probability: {prob:.2f})" if prob is not None else "Model predicts HEART DISEASE")
    else:
        st.success(f"Model predicts NO HEART DISEASE (probability: {prob:.2f})" if prob is not None else "Model predicts NO HEART DISEASE")

    st.write("Input used for prediction:")
    st.table(input_df.T)
