import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Page Configuration
st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🩺",
    layout="centered"
)

# Load the trained model system
@st.cache_resource
def load_model_system():
    try:
        return joblib.load('diabetes_model_system.pkl')
    except FileNotFoundError:
        return None

artifacts = load_model_system()

# Header
st.title("🩺 Diabetes Risk Prediction System")
st.markdown("Enter the patient's details below to assess the risk of Type 2 Diabetes.")

if artifacts is None:
    st.error("Model file not found! Please run 'train_model.py' first to generate the model.")
else:
    model = artifacts['model']
    scaler = artifacts['scaler']
    threshold = artifacts['threshold']

    # --- User Input Section ---
    st.sidebar.header("Patient Data")
    
    # Group 1: Basic Stats
    st.subheader("1. Physiological Metrics")
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=45)
        bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0, format="%.1f")
    with col2:
        whr = st.number_input("Waist-to-Hip Ratio", min_value=0.5, max_value=1.5, value=0.9, format="%.2f", help="A healthy ratio is typically < 0.9 for men and < 0.85 for women.")
    
    # Group 2: History & Lifestyle
    st.subheader("2. History & Lifestyle")
    
    family_hist_input = st.selectbox("Family History of Diabetes?", ["No", "Yes"])
    family_history = 1 if family_hist_input == "Yes" else 0
    
    col3, col4 = st.columns(2)
    with col3:
        activity = st.number_input("Physical Activity (min/week)", min_value=0, max_value=1000, value=120)
    with col4:
        screen_time = st.number_input("Screen Time (hours/day)", min_value=0.0, max_value=24.0, value=6.0, format="%.1f")

    # --- Feature Engineering (Backend) ---
    # We must replicate the exact calculations from the training script
    age_bmi = age * bmi
    genetic_bmi = family_history * bmi
    central_obesity = whr * bmi

    # Create DataFrame for prediction
    input_data = pd.DataFrame({
        'age': [age],
        'bmi': [bmi],
        'waist_to_hip_ratio': [whr],
        'family_history_diabetes': [family_history],
        'physical_activity_minutes_per_week': [activity],
        'screen_time_hours_per_day': [screen_time],
        'Age_BMI': [age_bmi],              # Interaction Term 1
        'Genetic_BMI': [genetic_bmi],      # Interaction Term 2
        'Central_Obesity_Index': [central_obesity] # Interaction Term 3
    })

    # --- Prediction ---
    if st.button("Analyze Risk", type="primary"):
        # Scale the data
        input_scaled = scaler.transform(input_data)
        
        # Get Probability
        probability = model.predict_proba(input_scaled)[0, 1]
        
        # Apply the Optimized Threshold
        prediction = 1 if probability >= threshold else 0
        
        st.divider()
        
        # Display Results
        if prediction == 1:
            st.error(f"## High Risk Detected")
            st.write(f"The model predicts a high likelihood of diabetes based on the provided factors.")
            st.progress(probability)
            st.caption(f"Risk Probability: {probability:.1%}")
            
            st.warning("⚠️ **Recommendation:** Please consult with a healthcare professional for a formal diagnosis (HbA1c / Fasting Glucose test).")
        else:
            st.success(f"## Low Risk Detected")
            st.write(f"The model predicts a low likelihood of diabetes.")
            st.progress(probability)
            st.caption(f"Risk Probability: {probability:.1%}")
            
            st.info("✅ **Recommendation:** Maintain a healthy lifestyle to keep risk low.")

    # Show model info
    with st.expander("About the Model"):
        st.write(f"**Model Type:** Logistic Regression (Balanced)")
        st.write(f"**Optimized Decision Threshold:** {threshold:.2f}")
        st.write("This model uses interaction terms like 'Age × BMI' and 'Waist-to-Hip × BMI' to better detect high-risk combinations.")