import streamlit as st
import pandas as pd
import joblib

# Load model and features
model = joblib.load("./model/model.pkl")
features = joblib.load("./model/features.pkl")

st.title("🏦 Loan Approval Prediction")

st.write("Fill the details below to check loan approval probability:")

# User inputs
gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["No", "Yes"])
dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
education = st.selectbox("Education", ["Not Graduate", "Graduate"])
self_employed = st.selectbox("Self Employed", ["No", "Yes"])
applicant_income = st.number_input("Applicant Income", min_value=0)
coapplicant_income = st.number_input("Coapplicant Income", min_value=0)
loan_amount = st.number_input("Loan Amount  ", min_value=0)
loan_amount_term = st.number_input("Loan Amount Term (years)", min_value=0)
credit_history = st.selectbox("Credit History", [0.0, 1.0])
property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

# Prepare data for prediction
if st.button("Predict Loan Approval"):
    # Map categorical to numbers
    input_dict = {
        "Gender": 0 if gender == "Male" else 1,
        "Married": 0 if married == "No" else 1,
        "Education": 0 if education == "Not Graduate" else 1,
        "Self_Employed": 0 if self_employed == "No" else 1,
        "ApplicantIncome": applicant_income,
        "CoapplicantIncome": coapplicant_income,
        "LoanAmount": loan_amount,
        "Loan_Amount_Term": loan_amount_term,
        "Credit_History": credit_history,
        "Property_Area_Rural": 1 if property_area == "Rural" else 0,
        "Property_Area_Semiurban": 1 if property_area == "Semiurban" else 0,
        "Property_Area_Urban": 1 if property_area == "Urban" else 0,
        "Dependents_0": 1 if dependents == "0" else 0,
        "Dependents_1": 1 if dependents == "1" else 0,
        "Dependents_2": 1 if dependents == "2" else 0,
        "Dependents_3+": 1 if dependents == "3+" else 0
    }

    # Ensure correct order of features
    input_df = pd.DataFrame([input_dict])[features]

    # Predict
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    if probability >= 0.4:
        st.success(f"✅ Loan Approved!")
    else:
        st.error(f"❌ Loan Not Approved")
