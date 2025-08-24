import streamlit as st
import joblib
import numpy as np
import pandas as pd


# Load Models
heart_model = joblib.load("heartdisease.joblib")
diabetes_model = joblib.load("diabetes.joblib")
parkinsons_model = joblib.load("parkinson.joblib")

# Feature Lists
heartdisease_features = [
    'age', 'sex', 'trestbps', 'chol', 'thalach', 'exang', 'oldpeak', 'ca',
    'cp_2.0', 'cp_3.0', 'cp_4.0',
    'restecg_1.0', 'restecg_2.0',
    'slope_2.0','thal_6.0', 'thal_7.0'
]

diabetes_features = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
]

parkinsons_features = [
    'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)',
    'MDVP:Shimmer', 'RPDE', 'DFA', 'spread1', 'spread2', 'D2'
]

# App Title
st.title("Disease Prediction System")
st.write("Predict **Heart Disease**, **Diabetes**, or **Parkinson's Disease**")

# Disease Selection
disease = st.sidebar.selectbox(
    "Select Disease to Predict",
    ("Heart Disease", "Diabetes", "Parkinson's")
)

# Input Forms Based on Selection

# HEART DISEASE
if disease == "Heart Disease":
    st.subheader("Heart Disease Prediction")

    age = st.number_input("Age", 1, 120, 50)
    sex = st.selectbox("Sex", ("Male", "Female"))
    trestbps = st.number_input("Resting Blood Pressure", 80, 200, 120)
    chol = st.number_input("Cholesterol", 100, 600, 200)
    thalach = st.number_input("Max Heart Rate", 60, 220, 150)
    exang = st.selectbox("Exercise Induced Angina", ("No", "Yes"))
    oldpeak = st.number_input("ST Depression (oldpeak)", 0.0, 10.0, 1.0)
    ca = st.number_input("Number of Major Vessels (0–3)", 0, 3, 0)

    # One-hot categorical features
    cp = st.selectbox("Chest Pain Type", ("1: Typical", "2: Atypical", "3: Non-anginal", "4: Asymptomatic"))
    restecg = st.selectbox("Resting ECG", ("0: Normal", "1: ST-T Abnormality", "2: Left Ventricular Hypertrophy"))
    slope = st.selectbox("Slope of ST Segment", ("1: Upsloping", "2: Flat", "3: Downsloping"))
    thal = st.selectbox("Thalassemia", ("3: Normal", "6: Fixed Defect", "7: Reversible Defect"))

    # Encoding
    sex = 1 if sex == "Male" else 0
    exang = 1 if exang == "Yes" else 0

    cp_2, cp_3, cp_4 = 0, 0, 0
    if "2" in cp: cp_2 = 1
    if "3" in cp: cp_3 = 1
    if "4" in cp: cp_4 = 1

    restecg_1, restecg_2 = 0, 0
    if "1" in restecg: restecg_1 = 1
    if "2" in restecg: restecg_2 = 1

    slope_2 = 1 if "2" in slope else 0
    thal_6, thal_7 = 0, 0
    if "6" in thal: thal_6 = 1
    if "7" in thal: thal_7 = 1

    if st.button("Predict Heart Disease"):
        features = pd.DataFrame([[
            age, sex, trestbps, chol, thalach, exang, oldpeak, ca,
            cp_2, cp_3, cp_4,
            restecg_1, restecg_2,
            slope_2,
            thal_6, thal_7
        ]], columns=heartdisease_features)

        prediction = heart_model.predict(features)
        result = "Positive (Heart Disease)" if prediction[0] == 1 else "Negative (No Heart Disease)"
        st.success(f"Prediction: {result}")


# DIABETES
elif disease == "Diabetes":
    st.subheader("Diabetes Prediction")

    pregnancies = st.number_input("Pregnancies", 0, 20, 1)
    glucose = st.number_input("Glucose Level", 0, 300, 120)
    bp = st.number_input("Blood Pressure", 0, 200, 80)
    skin = st.number_input("Skin Thickness", 0, 100, 20)
    insulin = st.number_input("Insulin Level", 0, 1000, 80)
    bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
    dpf = st.number_input("Diabetes Pedigree Function", 0.0, 5.0, 0.5)
    age = st.number_input("Age", 1, 120, 30)

    if st.button("Predict Diabetes"):
        # Apply log transform to all except glucose
        pregnancies_t = np.log1p(pregnancies)
        bp_t = np.log1p(bp)
        skin_t = np.log1p(skin)
        insulin_t = np.log1p(insulin)
        bmi_t = np.log1p(bmi)
        dpf_t = np.log1p(dpf)
        age_t = np.log1p(age)

        features = np.array([[pregnancies_t, glucose, bp_t, skin_t,
                              insulin_t, bmi_t, dpf_t, age_t]])

        prediction = diabetes_model.predict(features)
        result = "Positive (Diabetes)" if prediction[0] == 1 else "Negative (No Diabetes)"
        st.success(f"Prediction: {result}")


# PARKINSON’S
elif disease == "Parkinson's":
    st.subheader("Parkinson's Prediction")

    fo = st.number_input("MDVP:Fo(Hz)", 0.0, 500.0, 150.0)
    fhi = st.number_input("MDVP:Fhi(Hz)", 0.0, 500.0, 200.0)
    flo = st.number_input("MDVP:Flo(Hz)", 0.0, 500.0, 100.0)
    jitter = st.number_input("MDVP:Jitter(%)", 0.0, 1.0, 0.01)
    shimmer = st.number_input("MDVP:Shimmer", 0.0, 1.0, 0.01)
    rpde = st.number_input("RPDE", 0.0, 2.0, 0.5)
    dfa = st.number_input("DFA", 0.0, 2.0, 0.5)
    spread1 = st.number_input("Spread1", -10.0, 10.0, -4.0)
    spread2 = st.number_input("Spread2", 0.0, 10.0, 3.0)
    d2 = st.number_input("D2", 0.0, 5.0, 2.0)

    if st.button("Predict Parkinson's"):
        features = pd.DataFrame([[
            fo, fhi, flo, jitter, shimmer, rpde, dfa, spread1, spread2, d2
        ]], columns=parkinsons_features)

        prediction = parkinsons_model.predict(features)
        result = "Positive (Parkinson's)" if prediction[0] == 1 else "Negative (No Parkinson's)"
        st.success(f"Prediction: {result}")