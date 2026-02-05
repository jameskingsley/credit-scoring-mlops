import streamlit as st
import requests

API_URL = "https://credit-scoring-mlops-933z.onrender.com/predict"

st.set_page_config(page_title="Credit Scoring System", layout="centered")

st.title("Credit Default Risk Scoring")

with st.form("credit_form"):
    age = st.number_input("Age", 18, 100)
    employment = st.selectbox("Employment Type", ["Salaried", "Self-employed"])
    applicant = st.selectbox("Applicant Type", ["Individual", "Business"])
    income = st.number_input("Annual Income")
    loan_amount = st.number_input("Loan Amount")
    credit_score = st.number_input("Credit Score", 300, 850)

    submitted = st.form_submit_button("Predict Risk")

if submitted:
    payload = {
        "Age": age,
        "EmploymentType": employment,
        "ApplicantType": applicant,
        "AnnualIncome": income,
        "MonthlyIncome": income / 12,
        "LoanType": "Personal",
        "LoanAmount": loan_amount,
        "LoanTenureMonths": 36,
        "InterestRate": 18.5,
        "CollateralValue": 0,
        "CreditScore": credit_score,
        "PastDefaults": 0,
        "NumOpenAccounts": 2,
        "BusinessRevenue": 0,
        "ProfitMargin": 0,
        "BusinessYears": 0
    }

    response = requests.post(API_URL, json=payload)

    if response.status_code == 200:
        result = response.json()
        st.success(f"PD: {result['probability_default']:.2%}")
        st.info(f"Risk Band: {result['risk_band']}")
    else:
        st.error("Prediction failed")
