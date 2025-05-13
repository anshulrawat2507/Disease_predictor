import streamlit as st
import numpy as np
import joblib

st.set_page_config(page_title="Disease Prediction App")

st.title("🧠 Smart Disease Prediction System")

# Load models
diabetes_model = joblib.load("diabetes_model.pkl")
heart_model = joblib.load("heart_model.pkl")

# Select disease
disease = st.selectbox("Choose Disease to Predict", ["Diabetes", "Heart Attack"])

# ==============================
# 🚑 DIABETES PREDICTION FORM
# ==============================
if disease == "Diabetes":
    st.subheader("Enter Patient Details for Diabetes Prediction")

    preg = st.number_input("Pregnancies", min_value=0)
    glucose = st.number_input("Glucose", min_value=0)
    bp = st.number_input("Blood Pressure", min_value=0)
    skin = st.number_input("Skin Thickness", min_value=0)
    insulin = st.number_input("Insulin", min_value=0)
    bmi = st.number_input("BMI", min_value=0.0)
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0)
    age = st.number_input("Age", min_value=1)

    if st.button("Predict Diabetes"):
        user_data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
        prediction = diabetes_model.predict(user_data)
        st.success("Diabetic" if prediction[0] == 1 else "Not Diabetic")

# ==============================
# ❤️ HEART ATTACK PREDICTION FORM
# ==============================
elif disease == "Heart Attack":
    st.subheader("Enter Patient Details for Heart Attack Prediction")

    age = st.number_input("Age", min_value=20)
    diabetes = st.selectbox("Diabetes", [0, 1])
    hypertension = st.selectbox("Hypertension", [0, 1])
    obesity = st.selectbox("Obesity", [0, 1])
    smoking = st.selectbox("Smoking", [0, 1])
    alcohol = st.selectbox("Alcohol Consumption", [0, 1])
    activity = st.slider("Physical Activity (1=low, 10=high)", 1, 10)
    diet = st.slider("Diet Score (1=poor, 10=excellent)", 1, 10)
    chol = st.number_input("Cholesterol Level")
    trig = st.number_input("Triglyceride Level")
    ldl = st.number_input("LDL Level")
    hdl = st.number_input("HDL Level")
    sys_bp = st.number_input("Systolic BP")
    dia_bp = st.number_input("Diastolic BP")
    air = st.number_input("Air Pollution Exposure")
    family = st.selectbox("Family History", [0, 1])
    stress = st.slider("Stress Level (1=low, 10=high)", 1, 10)
    access = st.selectbox("Healthcare Access", [0, 1])
    hist = st.selectbox("Previous Heart Attack", [0, 1])
    response = st.number_input("Emergency Response Time")
    income = st.number_input("Annual Income")
    insurance = st.selectbox("Health Insurance", [0, 1])

    if st.button("Predict Heart Attack Risk"):
        input_data = np.array([[age, diabetes, hypertension, obesity, smoking,
                                alcohol, activity, diet, chol, trig, ldl, hdl,
                                sys_bp, dia_bp, air, family, stress, access,
                                hist, response, income, insurance]])
        prediction = heart_model.predict(input_data)
        st.success("At Risk of Heart Attack" if prediction[0] == 1 else "Not at Risk")
