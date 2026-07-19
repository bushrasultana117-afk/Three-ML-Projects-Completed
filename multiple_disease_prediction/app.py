import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import os
import time

# Set page configuration
st.set_page_config(
    page_title="Health Assistant",
    layout="wide",
    page_icon="🧑‍⚕️"
)

# Custom CSS
st.markdown("""
<style>
/* Main background */
.stApp {
    background: linear-gradient(135deg, #E3F2FD, #FFFFFF);
}

/* Title */
h1 {
    color: #0F62FE;
    text-align: center;
    font-weight: bold;
}

/* Sub Headers */
h2, h3 {
    color: #1565C0;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(#0F172A, #1E293B);
}

section[data-testid="stSidebar"] * {
    color: white;
}

/* Buttons */
.stButton>button {
    width: 100%;
    background: #0F62FE;
    color: white;
    border-radius: 10px;
    height: 55px;
    font-size: 18px;
    font-weight: bold;
    border: none;
}

/* Button Hover */
.stButton>button:hover {
    background: #0043CE;
    transform: scale(1.02);
}

/* Input Boxes */
.stTextInput>div>div>input {
    border-radius: 10px;
    border: 2px solid #90CAF9;
}

/* Metric Cards */
.metric-box {
    background: white;
    padding: 15px;
    border-radius: 15px;
    box-shadow: 0px 5px 15px rgba(0,0,0,0.15);
}
</style>
""", unsafe_allow_html=True)

# Getting the working directory of the main.py
working_dir = os.path.dirname(os.path.abspath(__file__))

# Loading the Trained Models safely
try:
    diabetes_model = pickle.load(open(os.path.join(working_dir, 'models', 'Diabetics_model.sav'), 'rb'))
    heart_disease_model = pickle.load(open(os.path.join(working_dir, 'models', 'Heart_Disease_model.sav'), 'rb'))
    Parkinsons_model = pickle.load(open(os.path.join(working_dir, 'models', 'parkinsons_model.sav'), 'rb'))
except FileNotFoundError as e:
    st.error(f"Model file loading failed. Verify your 'models' folder structure. Error: {e}")

# Sidebar for Navigation
with st.sidebar:
    selected = option_menu(
        "🏥 AI Health Assistant",
        [
            '🩸Diabetes Prediction',
            '❤️Heart Disease Prediction',
            '🧠Parkinsons Prediction'
        ],
        icons=["activity", "heart-pulse", "person"],
        menu_icon="hospital",
        default_index=0
    )

# -------------------------------------------------------------
# Diabetes Prediction Page
# -------------------------------------------------------------
if selected == '🩸Diabetes Prediction':
    st.markdown("""
    <style>
    .stButton>button {
        background: #E63946;
    }
    .stButton>button:hover {
        background: #C1121F;
    }
    [data-testid="stTextInput"] input {
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

    # Local path works only on your PC; replace with images/diabetes_banner.jpg for production
    image_path = os.path.join(working_dir, "images", "diabetes_banner.jpg")

    if os.path.exists(image_path):
        st.image(image_path, width="stretch")

    st.markdown("<h1 style='text-align:center; color:#E63946; font-size:42px; font-weight:bold;'>🩸 Diabetes Prediction</h1>", unsafe_allow_html=True)
    st.write('Classifies individuals as diabetic or non-diabetic using machine learning and health-related attributes.')
    st.info("ℹ️ Enter all patient values carefully. The prediction is based on a trained Machine Learning model.")

    with st.expander("📋 Click here to enter Patient Details 👇", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            Pregnancies = st.text_input('Number of Pregnancies', value="1")
            SkinThickness = st.text_input('Skin Thickness Value', value="0")
        with col2:
            Glucose = st.text_input('Glucose Level', value="126")
            Insulin = st.text_input('Insulin Level', value="0")
        with col3:
            BloodPressure = st.text_input('Blood Pressure Value', value="60")
            BMI = st.text_input('BMI Value', value="30.1")
        
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function Value', value="0.349")
        Age = st.slider("Age", min_value=0, max_value=100, value=47)

    if st.button("🩸 Diabetes Test Result"):
        with st.spinner("🔍 Analyzing Patient Data..."):
            time.sleep(2)
            try:
                input_data = [
                    float(Pregnancies), float(Glucose), float(BloodPressure),
                    float(SkinThickness), float(Insulin), float(BMI),
                    float(DiabetesPedigreeFunction), float(Age)
                ]
                diab_prediction = diabetes_model.predict([input_data])
                probability = diabetes_model.predict_proba([input_data])[0]

                st.subheader("📊 Prediction Result")
                if diab_prediction[0] == 1:
                    st.error("🩸 The person is predicted to be Diabetic.")
                    st.metric(label="🎯 Prediction Confidence", value=f"{probability[1]:.2%}")
                else:
                    st.success("✅ The person is predicted to be Non-Diabetic.")
                    st.metric(label="🎯 Prediction Confidence", value=f"{probability[0]:.2%}")
                    st.balloons()
            except ValueError:
                st.warning("⚠️ Please fill out all fields with valid numbers before testing.")

# -------------------------------------------------------------
# Heart Disease Prediction Page
# -------------------------------------------------------------
elif selected == '❤️Heart Disease Prediction':
    st.markdown("""
    <style>
    .stButton>button {
        background: #E63946;
    }
    .stButton>button:hover {
        background: #C1121F;
    }
    </style>
    """, unsafe_allow_html=True)

    image_path = os.path.join(working_dir, "images", "heart_banner.jpg")

    if os.path.exists(image_path):
        st.image(image_path, width="stretch")

    st.markdown("<h1 style='text-align:center; color:#E63946; font-size:42px; font-weight:bold;'>❤️ Heart Disease Prediction</h1>", unsafe_allow_html=True)
    st.write('Predicts the risk of heart disease using clinical patient data and machine learning.')
    st.info("ℹ️ Enter all patient values carefully. The prediction is based on a trained Machine Learning model.")

    with st.expander("📋 Click here to enter Patient Details 👇", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            age = st.slider("Age", min_value=1, max_value=100, value=30)
            cp = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"])
            cp_mapping = {"Typical Angina": 0, "Atypical Angina": 1, "Non-anginal Pain": 2, "Asymptomatic": 3}
            cp = cp_mapping[cp]
            chol = st.number_input("🧪 Cholesterol (mg/dL)", min_value=100, max_value=600, value=200)
            restecg = st.number_input('Resting Electrocardiographic results', min_value=0, max_value=2, value=0)
            exang = st.selectbox("🏃 Exercise Induced Angina", ["No", "Yes"])
            exang = 0 if exang == "No" else 1
            slope = st.selectbox("📈 Slope of Peak Exercise ST Segment", ["Upsloping", "Flat", "Downsloping"])
            slope = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}[slope]
            thal = st.selectbox("🧬 Thalassemia", ["Normal", "Fixed Defect", "Reversible Defect"])
            thal = {"Normal": 0, "Fixed Defect": 1, "Reversible Defect": 2}[thal]

        with col2:
            sex = st.selectbox("Gender", ["Male", "Female"])
            sex = 1 if sex == "Male" else 0
            trestbps = st.number_input("🩸 Resting Blood Pressure (mmHg)", min_value=80, max_value=250, value=120)
            fbs = st.selectbox("Fasting Blood Sugar (> 120 mg/dl)", ["No (≤ 120 mg/dl)", "Yes (> 120 mg/dl)"])
            fbs = 0 if fbs == "No (≤ 120 mg/dl)" else 1
            thalach = st.number_input('Maximum Heart Rate achieved', min_value=60, max_value=220, value=150)
            oldpeak = st.number_input("📉 ST Depression Induced by Exercise", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
            ca = st.selectbox("🩺 Major Vessels Colored by Fluoroscopy", [0, 1, 2, 3, 4])

    if st.button("❤️ Heart Disease Test Result"):
        with st.spinner("🔍 Analyzing Patient Data..."):
            time.sleep(2)
            input_data = [
                float(age), float(sex), float(cp), float(trestbps), float(chol),
                float(fbs), float(restecg), float(thalach), float(exang),
                float(oldpeak), float(slope), float(ca), float(thal)
            ]
            heart_prediction = heart_disease_model.predict([input_data])
            probability = heart_disease_model.predict_proba([input_data])[0]

            st.subheader("📊 Prediction Result")
            if heart_prediction[0] == 1:
                st.error("❤️ The person is predicted to have Heart Disease.")
                st.metric(label="🎯 Prediction Confidence", value=f"{probability[1]:.2%}")
            else:
                st.success("✅ The person is predicted to be Healthy.")
                st.metric(label="🎯 Prediction Confidence", value=f"{probability[0]:.2%}")
                st.balloons()

# -------------------------------------------------------------
# Parkinson's Prediction Page
# -------------------------------------------------------------
elif selected == '🧠Parkinsons Prediction':
    st.markdown("""
    <style>
    .stButton>button {
        background: #6A1B9A;
    }
    .stButton>button:hover {
        background: #4A148C;
    }
    </style>
    """, unsafe_allow_html=True)
    
    image_path = os.path.join(working_dir, "images", "parkinsons_banner.jpg")

    if os.path.exists(image_path):
        st.image(image_path, width="stretch")
        

    st.markdown("<h1 style='text-align:center; color:#6A1B9A; font-size:42px; font-weight:bold;'>🧠 Parkinson's Prediction</h1>", unsafe_allow_html=True)
    st.write("Detects the likelihood of Parkinson's disease through machine learning analysis of medical vocal features.")
    st.info("ℹ️ Enter all patient values carefully. The prediction is based on a trained Machine Learning model.")
    
    with st.expander("📋 Click here to enter Voice Analysis Details 👇", expanded=False):
        st.subheader("🎤 Voice Frequency")
        st.divider()
        col1, col2, col3 = st.columns(3)
        with col1:
            fo = st.text_input('MDVP:Fo(Hz)', value="122.188")
        with col2:
            fhi = st.text_input('MDVP:Fhi(Hz)', value="128.611")
        with col3:
            flo = st.text_input('MDVP:Flo(Hz)', value="115.765")

        st.subheader("📊 Jitter Features")
        st.divider()
        col1, col2, col3 = st.columns(3)
        with col1:
            Jitter_percent = st.text_input('MDVP:Jitter(%)', value="0.00524")
            PPQ = st.text_input('MDVP:PPQ', value="0.00203")
        with col2:
            Jitter_Abs = st.text_input('MDVP:Jitter(Abs)', value="0.00004")
            DDP = st.text_input('Jitter:DDP', value="0.00507")
        with col3:
            RAP = st.text_input('MDVP:RAP', value="0.00004")

        st.subheader("📈 Shimmer Features")
        st.divider()
        col1, col2, col3 = st.columns(3)
        with col1:
            Shimmer = st.text_input('MDVP:Shimmer', value="0.01613")
            APQ5 = st.text_input('Shimmer:APQ5', value="0.00776")
        with col2:
            Shimmer_dB = st.text_input('MDVP:Shimmer(dB)', value="0.143")
            APQ = st.text_input('MDVP:APQ', value="0.01433")
        with col3:
            APQ3 = st.text_input('Shimmer:APQ3', value="0.00855")
            DDA = st.text_input('Shimmer:DDA', value="0.02566")

        st.subheader("🧠 Other Voice Features")
        st.divider()
        col1, col2, col3 = st.columns(3)
        with col1:
            NHR = st.text_input('NHR', value="0.00839")
            DFA = st.text_input('DFA', value="0.733659")
            D2 = st.text_input('D2', value="2.079922")
        with col2:
            HNR = st.text_input('HNR', value="23.162")
            spread1 = st.text_input('spread1', value="-6.439398")
            PPE = st.text_input('PPE', value="0.133867")
        with col3:
            RPDE = st.text_input('RPDE', value="0.579597")
            spread2 = st.text_input('spread2', value="0.266392")

    if st.button("🧠 Parkinson's Test Result"):
        with st.spinner("🔍 Analyzing Patient Data..."):
            time.sleep(2)
            try:
                user_input = [
                    float(fo), float(fhi), float(flo), float(Jitter_percent),
                    float(Jitter_Abs), float(RAP), float(PPQ), float(DDP),
                    float(Shimmer), float(Shimmer_dB), float(APQ3), float(APQ5),
                    float(APQ), float(DDA), float(NHR), float(HNR),
                    float(RPDE), float(DFA), float(spread1), float(spread2),
                    float(D2), float(PPE)
                ]

                parkinsons_prediction = Parkinsons_model.predict([user_input])
                st.subheader("📊 Prediction Result")

                if parkinsons_prediction[0] == 1:
                    st.error("🧠 The person is predicted to have Parkinson's Disease.")
                else:
                    st.success("✅ The person is predicted to be Healthy.")
                    st.balloons()
            except ValueError:
                st.warning("⚠️ Please enter valid numeric values in all fields.")

st.markdown("""
<style>
.footer {
    margin-top: 30px;
    padding: 18px;
    text-align: center;
    border-radius: 15px;
    background: linear-gradient(90deg, #0F62FE, #6C63FF);
    color: white;
    box-shadow: 0 6px 20px rgba(0,0,0,0.15);
}

.footer h3{
    margin:0;
    font-size:28px;
}

.footer p{
    margin:6px;
    font-size:16px;
}

.footer small{
    color:#E5E7EB;
}
</style>

<div class="footer">
    <h3>🏥 Multi Disease Prediction </h3>
    <p>Developed with by <b>Bushra Sultana</b></p>
    <small>© 2026 | Powered by Machine Learning • Streamlit • Python</small>
</div>
""", unsafe_allow_html=True)