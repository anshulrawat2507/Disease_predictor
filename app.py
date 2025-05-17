import streamlit as st
import numpy as np
import joblib

# Logo and project name
st.image("Healthi-fy.png", width=250)
st.markdown("<h2 style='text-align: center; color: #2F4F4F;'>HEALTHIFY: Smart Disease Prediction System</h2>", unsafe_allow_html=True)

# Custom CSS for beauty
st.markdown("""
    <style>
        .main {background-color: #f8f9fa;}
        .stButton button {
            background-color: #4CAF50;
            color: white;
            font-weight: bold;
            border-radius: 10px;
        }
        .stSelectbox, .stNumberInput {
            font-size: 16px !important;
        }
    </style>
""", unsafe_allow_html=True)


# Sidebar
st.sidebar.header("⚙️ Select Options")
disease = st.sidebar.selectbox("Choose Disease to Predict", ["Diabetes", "Heart Attack"])

# Load models
diabetes_model = joblib.load("diabetes_model.pkl")
heart_model = joblib.load("heart_model.pkl")

# 🧠 Diabetes Prediction
if disease == "Diabetes":
    st.subheader("🔬 Enter Patient Details for Diabetes")
    preg = st.number_input("Pregnancies", min_value=0)
    glucose = st.number_input("Glucose", min_value=0)
    bp = st.number_input("Blood Pressure", min_value=0)
    skin = st.number_input("Skin Thickness", min_value=0)
    insulin = st.number_input("Insulin", min_value=0)
    bmi = st.number_input("BMI", min_value=0.0)
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0)
    age = st.number_input("Age", min_value=1)

    if st.button("📈 Predict Diabetes"):
        user_data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
        prediction = diabetes_model.predict(user_data)
        proba = diabetes_model.predict_proba(user_data)[0][1]

        st.write("🎯 **Prediction Result**")
        if prediction[0] == 1:
            st.error("🩸 The patient is likely **Diabetic**")
        else:
            st.success("✅ The patient is likely **Not Diabetic**")
        st.info(f"🧪 Model confidence: `{proba:.2f}`")

# ❤️ Heart Attack Prediction
elif disease == "Heart Attack":
    st.subheader("🧬 Enter Patient Details for Heart Attack Risk")
    age = st.number_input("Age", min_value=20)
    diabetes = st.selectbox("Diabetes", [0, 1])
    hypertension = st.selectbox("Hypertension", [0, 1])
    obesity = st.selectbox("Obesity", [0, 1])
    smoking = st.selectbox("Smoking", [0, 1])
    alcohol = st.selectbox("Alcohol Consumption", [0, 1])
    activity = st.slider("Physical Activity (1 = low, 10 = high)", 1, 10)
    diet = st.slider("Diet Score (1 = poor, 10 = excellent)", 1, 10)
    chol = st.number_input("Cholesterol Level")
    trig = st.number_input("Triglyceride Level")
    ldl = st.number_input("LDL Level")
    hdl = st.number_input("HDL Level")
    sys_bp = st.number_input("Systolic BP")
    dia_bp = st.number_input("Diastolic BP")
    air = st.number_input("Air Pollution Exposure")
    family = st.selectbox("Family History", [0, 1])
    stress = st.slider("Stress Level (1 = low, 10 = high)", 1, 10)
    access = st.selectbox("Healthcare Access", [0, 1])
    hist = st.selectbox("Previous Heart Attack", [0, 1])
    response = st.number_input("Emergency Response Time")
    income = st.number_input("Annual Income")
    insurance = st.selectbox("Health Insurance", [0, 1])

    if st.button("📈 Predict Heart Attack Risk"):
        input_data = np.array([[age, diabetes, hypertension, obesity, smoking,
                                alcohol, activity, diet, chol, trig, ldl, hdl,
                                sys_bp, dia_bp, air, family, stress, access,
                                hist, response, income, insurance]])
        prediction = heart_model.predict(input_data)
        proba = heart_model.predict_proba(input_data)[0][1]

        st.write("🎯 **Prediction Result**")
        if prediction[0] == 1:
            st.error("💔 High Risk of Heart Attack")
        else:
            st.success("❤️ Low Risk of Heart Attack")
        st.info(f"🧪 Model confidence: `{proba:.2f}`")
