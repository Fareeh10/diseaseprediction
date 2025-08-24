import streamlit as st
import joblib
import numpy as np
import pandas as pd

# --- CONFIG ---
st.set_page_config(page_title="Disease Prediction System", page_icon="", layout="wide")

# --- LOAD MODELS ---
heart_model = joblib.load("heartdisease.joblib")
diabetes_model = joblib.load("diabetes.joblib")
parkinsons_model = joblib.load("parkinson.joblib")

# --- FEATURE LISTS ---
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

# --- CUSTOM CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Poppins', sans-serif; }
    .landing-container {
        display: flex; flex-direction: column; align-items: center; justify-content: center;
        text-align: center; padding-top: 12%;
    }
    .headline { font-size: 3rem;line-height: 3.8rem;font-weight: 700;color: #FF4B4B; margin-bottom: 2rem; }
    .subheadline { font-size: 1.2rem; color: #fff; margin-bottom: 2rem; max-width: 600px; }
    
    .stButton>button {
        background-color: #FF4B4B;
        color: white !important;
        font-size: 1.1em;
        font-weight: 600;
        padding: 0.8em 2.5em;
        border-radius: 50px;
        border: none;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }

    .stButton>button:hover {
        background-color: #e84343;
        transform: translateY(-2px);
        color: white !important;       /* 👈 Force text to stay white */
        opacity: 1 !important;         /* 👈 Prevent Streamlit fade */
    }
    
    /* Force all h3 (subheaders) in Streamlit to stay white */
    div[data-testid="stMarkdownContainer"] h3 {
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# --- SESSION STATE ---
if "show_app" not in st.session_state:
    st.session_state.show_app = False

# --- LANDING PAGE ---
if not st.session_state.show_app:
    st.markdown("""
        <div class="landing-container">
            <div class="headline">Disease Prediction System</div>
            <div class="subheadline">
                Predict the likelihood of Heart Disease, Diabetes, or Parkinson's Disease.<br>
                Enter your health details and get instant ML-powered predictions.
            </div>
        </div>
    """, unsafe_allow_html=True)

    _, col2, _ = st.columns([1,1,1])
    with col2:
        if st.button("🔍 Start Prediction", use_container_width=True):
            st.session_state.show_app = True
            st.rerun()

# --- MAIN APP ---
else:
    if st.button("Back to Home"):
        st.session_state.show_app = False
        st.rerun()

    # Disease Selection
    disease = st.radio(
        "Select Disease to Predict",
        ("Heart Disease", "Diabetes", "Parkinson's"),
        horizontal=True   # makes it appear as buttons in a row (better for mobile)
    )

    # --- HEART DISEASE ---
    if disease == "Heart Disease":
        st.markdown("<h3>Heart Disease Prediction</h3>", unsafe_allow_html=True)

        age = st.number_input("Age", 1, 120, 50)
        sex = st.selectbox("Sex", ("Male", "Female"))
        trestbps = st.number_input("Resting Blood Pressure", 80, 200, 120)
        chol = st.number_input("Cholesterol", 100, 600, 200)
        thalach = st.number_input("Max Heart Rate", 60, 220, 150)
        exang = st.selectbox("Exercise Induced Angina", ("No", "Yes"))
        oldpeak = st.number_input("ST Depression (oldpeak)", 0.0, 10.0, 1.0)
        ca = st.number_input("Number of Major Vessels (0–3)", 0, 3, 0)

        cp = st.selectbox("Chest Pain Type", ("1: Typical", "2: Atypical", "3: Non-anginal", "4: Asymptomatic"))
        restecg = st.selectbox("Resting ECG", ("0: Normal", "1: ST-T Abnormality", "2: Left Ventricular Hypertrophy"))
        slope = st.selectbox("Slope of ST Segment", ("1: Upsloping", "2: Flat", "3: Downsloping"))
        thal = st.selectbox("Thalassemia", ("3: Normal", "6: Fixed Defect", "7: Reversible Defect"))

        # Encoding
        sex = 1 if sex == "Male" else 0
        exang = 1 if exang == "Yes" else 0
        cp_2, cp_3, cp_4 = 0,0,0
        if "2" in cp: cp_2 = 1
        if "3" in cp: cp_3 = 1
        if "4" in cp: cp_4 = 1
        restecg_1, restecg_2 = 0,0
        if "1" in restecg: restecg_1 = 1
        if "2" in restecg: restecg_2 = 1
        slope_2 = 1 if "2" in slope else 0
        thal_6, thal_7 = 0,0
        if "6" in thal: thal_6 = 1
        if "7" in thal: thal_7 = 1

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Predict Heart Disease"):
            features = pd.DataFrame([[
                age, sex, trestbps, chol, thalach, exang, oldpeak, ca,
                cp_2, cp_3, cp_4,
                restecg_1, restecg_2,
                slope_2,
                thal_6, thal_7
            ]], columns=heartdisease_features)

            prediction = heart_model.predict(features)
            result = "Positive (Heart Disease Risk)" if prediction[0] == 1 else "Negative (No Heart Disease Risk)"
            st.success(f"Prediction: {result}")

    # --- DIABETES ---
    elif disease == "Diabetes":
        st.markdown("<h3>Diabetes Prediction</h3>", unsafe_allow_html=True)

        pregnancies = st.number_input("Pregnancies", 0, 20, 1)
        glucose = st.number_input("Glucose Level", 0, 300, 120)
        bp = st.number_input("Blood Pressure", 0, 200, 80)
        skin = st.number_input("Skin Thickness", 0, 100, 20)
        insulin = st.number_input("Insulin Level", 0, 1000, 80)
        bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
        dpf = st.number_input("Diabetes Pedigree Function", 0.0, 5.0, 0.5)
        age = st.number_input("Age", 1, 120, 30)

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Predict Diabetes"):
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

    # --- PARKINSON'S ---
    elif disease == "Parkinson's":
        st.markdown("<h3>Parkinson's Prediction</h3>", unsafe_allow_html=True)

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

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Predict Parkinson's"):
            features = pd.DataFrame([[
                fo, fhi, flo, jitter, shimmer, rpde, dfa, spread1, spread2, d2
            ]], columns=parkinsons_features)

            prediction = parkinsons_model.predict(features)
            result = "Positive (Parkinson's)" if prediction[0] == 1 else "Negative (No Parkinson's)"
            st.success(f"Prediction: {result}")
