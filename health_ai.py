# health_ai.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
import csv
from datetime import datetime
import ssl
import urllib.request

# Bypass SSL verification
ssl._create_default_https_context = ssl._create_unverified_context

# Configuration
HEART_DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
DIABETES_DATA_URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"

# Constants
EXERCISE_PLANS = {
    'Low': "30 mins daily walking/yoga",
    'Medium': "45 mins brisk walking + strength training 3x/week",
    'High': "60 mins moderate exercise 5x/week"
}

STRESS_TIPS = [
    "Deep breathing exercises",
    "Mindfulness meditation",
    "Regular sleep schedule",
    "Social activities"
]

def load_dataset(url):
    """Load datasets with SSL verification bypass"""
    with urllib.request.urlopen(url) as response:
        return pd.read_csv(response, header=None)

def train_and_save_models():
    """Train and save ML models"""
    try:
        # Heart Disease Model
        heart_columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
                        'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 
                        'ca', 'thal', 'num']
        heart_df = load_dataset(HEART_DATA_URL)
        heart_df.columns = heart_columns
        heart_df.replace('?', np.nan, inplace=True)
        heart_df = heart_df.apply(pd.to_numeric, errors='coerce')
        heart_df['target'] = heart_df['num'].apply(lambda x: 1 if x > 0 else 0)
        heart_features = heart_df[['age', 'sex', 'trestbps', 'chol', 'fbs', 'thalach', 'target']].dropna()

        X_heart = heart_features.drop('target', axis=1)
        y_heart = heart_features['target']
        X_train_heart, _, y_train_heart, _ = train_test_split(X_heart, y_heart, test_size=0.2)

        model_heart = RandomForestClassifier()
        model_heart.fit(X_train_heart, y_train_heart)
        joblib.dump(model_heart, "heart_model.joblib")

        # Diabetes Model
        diabetes_columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                          'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
        diabetes_df = load_dataset(DIABETES_DATA_URL)
        diabetes_df.columns = diabetes_columns
        diabetes_features = diabetes_df[['Glucose', 'BloodPressure', 'BMI', 'Age', 'Outcome']]
        diabetes_features.replace(0, np.nan, inplace=True)
        
        imputer = SimpleImputer(strategy='mean')
        diabetes_features[['Glucose', 'BloodPressure', 'BMI']] = imputer.fit_transform(
            diabetes_features[['Glucose', 'BloodPressure', 'BMI']])

        X_diabetes = diabetes_features.drop('Outcome', axis=1)
        y_diabetes = diabetes_features['Outcome']
        X_train_dia, _, y_train_dia, _ = train_test_split(X_diabetes, y_diabetes, test_size=0.2)

        model_diabetes = RandomForestClassifier()
        model_diabetes.fit(X_train_dia, y_train_dia)
        joblib.dump(model_diabetes, "diabetes_model.joblib")

    except Exception as e:
        st.error(f"Model training error: {str(e)}")

# Initialize models
try:
    model_heart = joblib.load("heart_model.joblib")
    model_diabetes = joblib.load("diabetes_model.joblib")
except FileNotFoundError:
    st.warning("Training models...")
    train_and_save_models()
    st.rerun()

# Streamlit UI
st.title("Indian Health Guardian 🇮🇳")
st.markdown("### AI-Powered Preventive Healthcare for India")

# Emergency Section
with st.expander("🚨 Emergency Health Assistance", expanded=True):
    st.write("""
    **National Emergency Numbers:**
    - 🚑 Ambulance: 108
    - 🚨 Police: 100
    - 🔥 Fire: 101
    """)
    st.write("24/7 Health Helpline: 104")

# Main Input Form
with st.form("health_form"):
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", 1, 120, 30)
        sex = st.selectbox("Gender", ["Male", "Female"])
        trestbps = st.number_input("Resting BP (mmHg)", 50, 200, 120)
        chol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200)
        
    with col2:
        glucose = st.number_input("Glucose (mg/dL)", 50, 300, 100)
        bmi = st.number_input("BMI", 10.0, 50.0, 22.0)
        thalach = st.number_input("Max Heart Rate", 60, 220, 150)
        stress = st.slider("Stress Level (1-10)", 1, 10, 3)
    
    submitted = st.form_submit_button("Analyze Health")

# Process Results
if submitted:
    # Predictions
    try:
        sex_enc = 1 if sex == "Male" else 0
        fbs = 1 if glucose > 120 else 0
        
        heart_proba = model_heart.predict_proba([[age, sex_enc, trestbps, chol, fbs, thalach]])[0][1]
        dia_proba = model_diabetes.predict_proba([[glucose, trestbps, bmi, age]])[0][1]

        # Risk Calculation
        def get_risk(prob):
            if prob < 0.3: return ("Low", "green")
            elif prob < 0.6: return ("Medium", "orange")
            else: return ("High", "red")

        heart_risk, h_color = get_risk(heart_proba)
        dia_risk, d_color = get_risk(dia_proba)

        # Display Results
        st.markdown("---")
        cols = st.columns(2)
        cols[0].markdown(f"### ❤️ Heart Risk: <span style='color:{h_color}'>{heart_risk}</span>",
                        unsafe_allow_html=True)
        cols[0].metric("Probability", f"{heart_proba*100:.1f}%")
        
        cols[1].markdown(f"### 🩸 Diabetes Risk: <span style='color:{d_color}'>{dia_risk}</span>",
                       unsafe_allow_html=True)
        cols[1].metric("Probability", f"{dia_proba*100:.1f}%")

        # Visualizations
        fig, ax = plt.subplots(figsize=(10,4))
        ax.barh(["Heart Disease", "Diabetes"], [heart_proba, dia_proba], 
               color=[h_color, d_color])
        ax.set_xlim(0, 1)
        st.pyplot(fig)

        # Wellness Plan
        st.markdown("---")
        st.subheader("📋 Personalized Wellness Plan")
        
        plan_cols = st.columns(2)
        with plan_cols[0]:
            st.markdown("### 🥗 Indian Diet Guide")
            if heart_risk in ["Medium", "High"]:
                st.write("""
                - Dal and whole grain roti
                - Fresh seasonal vegetables
                - Low-fat curd/yogurt
                - Nuts and seeds
                """)
            if dia_risk in ["Medium", "High"]:
                st.write("""
                - Brown rice instead of white
                - Bitter gourd (karela) dishes
                - Fenugreek (methi) preparations
                - Low glycemic fruits
                """)
        
        with plan_cols[1]:
            st.markdown("### 🏋️ Exercise Plan")
            st.write(EXERCISE_PLANS[max(heart_risk, dia_risk, key=lambda x: ["Low", "Medium", "High"].index(x))])
            st.write("""
            **Yoga Asanas:**
            - Bhujangasana (Cobra Pose)
            - Vajrasana (Diamond Pose)
            - Shavasana (Corpse Pose)
            """)

        # Community Support
        st.markdown("---")
        st.subheader("🤝 Indian Healthcare Support")
        
        tab1, tab2, tab3 = st.tabs(["National Programs", "Online Support", "Education"])
        with tab1:
            st.write("""
            **Government Initiatives:**
            - Ayushman Bharat: 14555
            - National Health Mission
            - Jan Aushadhi Kendras
            """)
            st.write("Find local clinics: [Health Center Locator](https://abdm.gov.in)")
        
        with tab2:
            st.markdown("""
            **Digital India Health Services:**
            - eSanjeevani Telemedicine: [esanjeevani.in](https://esanjeevani.in)
            - Practo: [practo.com](https://www.practo.com)
            - 1mg: [1mg.com](https://www.1mg.com)
            """)
        
        with tab3:
            st.video("https://www.youtube.com/watch?v=9Z4N9pR4pD4")

        # Logging
        log_entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "age": age,
            "sex": sex,
            "trestbps": trestbps,
            "chol": chol,
            "glucose": glucose,
            "bmi": bmi,
            "thalach": thalach,
            "heart_risk": heart_risk,
            "dia_risk": dia_risk
        }
        
        try:
            with open("health_logs.csv", "a") as f:
                writer = csv.DictWriter(f, fieldnames=log_entry.keys())
                if f.tell() == 0:
                    writer.writeheader()
                writer.writerow(log_entry)
        except Exception as e:
            st.error(f"Logging error: {str(e)}")

    except Exception as e:
        st.error(f"Prediction error: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
**Preventive Care Tips for India:**
- Regular Ayurvedic checkups
- Include millets in diet
- Practice daily yoga
- Use AYUSH Ministry resources
""")

