# app.py - Stage 3: Streamlit Web App
# Workforce Retention Intelligence System using XGBoost

import streamlit as st
import pandas as pd
import pickle
from xgboost import XGBClassifier

model = XGBClassifier()
model.load_model('xgb_attrition_model.json')

with open('encoders.pkl', 'rb') as f:
    encoders = pickle.load(f)

# Same threshold decided during Stage 2 (balances accuracy and recall)
THRESHOLD = 0.3

st.title("Workforce Retention Intelligence System")
st.write("Predict whether an employee is likely to leave the company.")

# ---- Input Fields ----
age = st.number_input("Age", min_value=18, max_value=60, value=30)

business_travel = st.selectbox("Business Travel", ['Non-Travel', 'Travel_Rarely', 'Travel_Frequently'])
department = st.selectbox("Department", ['Sales', 'Research & Development', 'Human Resources'])
distance_from_home = st.number_input("Distance From Home (km)", min_value=0, max_value=50, value=5)

environment_satisfaction = st.slider("Environment Satisfaction", 1, 4, 3)
job_involvement = st.slider("Job Involvement", 1, 4, 3)
job_role = st.selectbox("Job Role", [
    'Sales Executive', 'Research Scientist', 'Laboratory Technician',
    'Manufacturing Director', 'Healthcare Representative', 'Manager',
    'Sales Representative', 'Research Director', 'Human Resources'
])
job_satisfaction = st.slider("Job Satisfaction", 1, 4, 3)
monthly_income = st.number_input("Monthly Income", min_value=1000, max_value=50000, value=5000)
overtime = st.selectbox("OverTime", ['Yes', 'No'])
performance_rating = st.slider("Performance Rating", 1, 4, 3)
work_life_balance = st.slider("Work Life Balance", 1, 4, 3)
years_at_company = st.number_input("Years At Company", min_value=0, max_value=40, value=3)

# ---- Predict Button ----
if st.button("Predict"):

    # ---- Apply same feature engineering used during training (Stage 1) ----
    income_per_year = monthly_income / (years_at_company + 1)
    high_overtime_risk = 1 if overtime == 'Yes' else 0

    travel_map = {'Travel_Frequently': 2, 'Travel_Rarely': 1, 'Non-Travel': 0}
    travel_stress = travel_map[business_travel] * distance_from_home

    job_hopper_risk = 1 if (job_satisfaction <= 2 and job_involvement <= 2) else 0

    # Same median used in training would be ideal, but a fixed reference income keeps this simple
    low_income_high_dissatisfaction = 1 if (monthly_income < 4919 and job_satisfaction <= 2) else 0

    # ---- Build input row (must match X_train column order/names) ----
    input_data = pd.DataFrame([{
        'Age': age,
        'BusinessTravel': encoders['BusinessTravel'].transform([business_travel])[0],
        'Department': encoders['Department'].transform([department])[0],
        'DistanceFromHome': distance_from_home,
        'EnvironmentSatisfaction': environment_satisfaction,
        'JobInvolvement': job_involvement,
        'JobRole': encoders['JobRole'].transform([job_role])[0],
        'JobSatisfaction': job_satisfaction,
        'MonthlyIncome': monthly_income,
        'OverTime': encoders['OverTime'].transform([overtime])[0],
        'PerformanceRating': performance_rating,
        'WorkLifeBalance': work_life_balance,
        'YearsAtCompany': years_at_company,
        'IncomePerYearAtCompany': income_per_year,
        'HighOvertimeRisk': high_overtime_risk,
        'TravelStress': travel_stress,
        'JobHopperRisk': job_hopper_risk,
        'LowIncomeHighDissatisfaction': low_income_high_dissatisfaction
    }])

    # ---- Predict using probability + our chosen threshold (not default 0.5) ----
    probability = model.predict_proba(input_data)[0][1]
    prediction = 1 if probability >= THRESHOLD else 0

    st.subheader("Prediction Result")
    if prediction == 1:
        st.error("Likely to Leave")
    else:
        st.success("Likely to Stay")

    st.write(f"Probability of Attrition: {probability:.2%}")
