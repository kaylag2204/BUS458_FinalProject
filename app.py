import streamlit as st
import pickle
import pandas as pd
import numpy as np

# -----------------------------------------------------
# Load Model + Scaler
# -----------------------------------------------------
try:
    with open("my_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("loan_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
except Exception as e:
    st.error(f"Error loading model or scaler: {e}")
    st.stop()

# -----------------------------------------------------
# Feature list FROM YOUR TRAINED MODEL
# (These are the columns in X_train.columns)
# -----------------------------------------------------
model_columns = [
    'Requested_Loan_Amount', 'FICO_score', 'Monthly_Gross_Income',
    'Monthly_Housing_Payment', 'Ever_Bankrupt_or_Foreclose',

    # Reason (reference = cover_an_unexpected_cost)
    'Reason_credit_card_refinancing', 'Reason_debt_conslidation',
    'Reason_home_improvement', 'Reason_major_purchase', 'Reason_unknown',

    # FICO group (reference = excellent)
    'Fico_Score_group_fair', 'Fico_Score_group_good',
    'Fico_Score_group_poor', 'Fico_Score_group_very_good',

    # Employment Status (reference = full_time)
    'Employment_Status_part_time', 'Employment_Status_unemployed',

    # Employment Sector (reference = unknown)
    'Employment_Sector_communication_services',
    'Employment_Sector_consumer_discretionary',
    'Employment_Sector_consumer_staples', 'Employment_Sector_energy',
    'Employment_Sector_financials', 'Employment_Sector_health_care',
    'Employment_Sector_industrials', 'Employment_Sector_information_technology',
    'Employment_Sector_materials', 'Employment_Sector_real_estate',
    'Employment_Sector_utilities',

    # Lender (reference = A)
    'Lender_B', 'Lender_C'
]

# -----------------------------------------------------
# Title
# -----------------------------------------------------
st.markdown(
    "<h1 style='text-align: center; background-color: #4CAF50; padding: 10px; color: white;'><b>Loan Approval Prediction</b></h1>",
    unsafe_allow_html=True
)

st.header("Enter Applicant Details")

# -----------------------------------------------------
# Numeric Inputs
# -----------------------------------------------------
requested = st.number_input("Requested Loan Amount ($)", 500, 150000)
fico = st.slider("FICO Score", 300, 850)
income = st.number_input("Monthly Gross Income ($)", 0, 50000)
housing = st.number_input("Monthly Housing Payment ($)", 0, 10000)
bankrupt = st.radio("Ever Bankrupt or Foreclose?", [0,1], format_func=lambda x: "No" if x==0 else "Yes")

# -----------------------------------------------------
# Categorical Inputs
# -----------------------------------------------------
reason = st.selectbox("Reason for Loan", [
    "cover_an_unexpected_cost",
    "credit_card_refinancing",
    "debt_conslidation",
    "home_improvement",
    "major_purchase",
    "unknown"
])

fico_group = st.selectbox("FICO Score Group", [
    "excellent", "fair", "good", "poor", "very_good"
])

emp_status = st.selectbox("Employment Status", [
    "full_time", "part_time", "unemployed"
])

emp_sector = st.selectbox("Employment Sector", [
    "unknown",
    "communication_services", "consumer_discretionary", "consumer_staples",
    "energy", "financials", "health_care", "industrials",
    "information_technology", "materials", "real_estate", "utilities"
])

lender = st.selectbox("Lender", ["A","B","C"])

# -----------------------------------------------------
# Create zero-filled row for model input
# -----------------------------------------------------
row = {col: 0 for col in model_columns}

row["Requested_Loan_Amount"] = requested
row["FICO_score"] = fico
row["Monthly_Gross_Income"] = income
row["Monthly_Housing_Payment"] = housing
row["Ever_Bankrupt_or_Foreclose"] = bankrupt

# Reason
if reason != "cover_an_unexpected_cost":
    key = f"Reason_{reason}"
    if key in row: row[key] = 1

# FICO group
if fico_group != "excellent":
    key = f"Fico_Score_group_{fico_group}"
    if key in row: row[key] = 1

# Employment Status
if emp_status != "full_time":
    key = f"Employment_Status_{emp_status}"
    if key in row: row[key] = 1

# Employment Sector
if emp_sector != "unknown":
    key = f"Employment_Sector_{emp_sector}"
    if key in row: row[key] = 1

# Lender
if lender in ["B","C"]:
    key = f"Lender_{lender}"
    if key in row: row[key] = 1

# Convert to df
input_df = pd.DataFrame([row])

# -----------------------------------------------------
# Scale features (required!)
# -----------------------------------------------------
input_scaled = scaler.transform(input_df)

# -----------------------------------------------------
# Predict
# -----------------------------------------------------
if st.button("Evaluate Loan Application"):
    prob = model.predict_proba(input_scaled)[0]
    pred = model.predict(input_scaled)[0]

    if pred == 1:
        st.success(f"🎉 Loan APPROVED! (Confidence {prob[1]*100:.1f}%)")
        st.balloons()
    else:
        st.error(f"❌ Loan DENIED (Confidence {prob[0]*100:.1f}%)")

    st.subheader("Prediction Probabilities")
    st.metric("Approval", f"{prob[1]*100:.2f}%")
    st.metric("Denial", f"{prob[0]*100:.2f}%")
