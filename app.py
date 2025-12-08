# loanapp.py
# -*- coding: utf-8 -*-
import streamlit as st
import pickle
import pandas as pd
import numpy as np

# -----------------------------------------------------
# Load Model + Scaler
# -----------------------------------------------------
try:
    with open("my_model.pkl", "rb") as file:
        model = pickle.load(file)
    with open("loan_scaler.pkl", "rb") as file:
        scaler = pickle.load(file)

except Exception as e:
    st.error(f"Error loading model or scaler: {e}")
    st.stop()

# -----------------------------------------------------
# Define feature names manually
# -----------------------------------------------------
model_columns = [
    'Requested_Loan_Amount', 'FICO_score', 'Monthly_Gross_Income',
    'Monthly_Housing_Payment', 'Ever_Bankrupt_or_Foreclose',
    'Reason_credit_card_refinancing', 'Reason_debt_conslidation',
    'Reason_home_improvement', 'Reason_major_purchase', 'Reason_other',
    'Fico_Score_group_fair', 'Fico_Score_group_good',
    'Fico_Score_group_poor', 'Fico_Score_group_very_good',
    'Employment_Status_part_time', 'Employment_Status_unemployed',
    'Employment_Sector_communication_services',
    'Employment_Sector_consumer_discretionary',
    'Employment_Sector_consumer_staples', 'Employment_Sector_energy',
    'Employment_Sector_financials', 'Employment_Sector_health_care',
    'Employment_Sector_industrials',
    'Employment_Sector_information_technology',
    'Employment_Sector_materials', 'Employment_Sector_real_estate',
    'Employment_Sector_utilities',
    'Lender_B', 'Lender_C'
]

# -----------------------------------------------------
# UI
# -----------------------------------------------------
st.markdown(
    "<h1 style='text-align: center; background-color: #4CAF50; padding: 10px; color: white;'><b>Loan Approval Prediction</b></h1>",
    unsafe_allow_html=True
)

st.header("Enter Applicant Details")

requested_loan = st.number_input("Requested Loan Amount ($)", 1000, 150000, step=500)
fico_score = st.slider("FICO Score", 300, 850)
monthly_income = st.number_input("Monthly Gross Income ($)", 0, step=100)
monthly_housing = st.number_input("Monthly Housing Payment ($)", 0, step=50)
bankrupt = st.radio("Ever Bankrupt or Foreclose?", [0,1], format_func=lambda x: "No" if x==0 else "Yes")

reason = st.selectbox("Reason for Loan", [
    "credit_card_refinancing", "debt_conslidation", "home_improvement",
    "major_purchase", "other", "cover_an_unexpected_cost"
])

fico_group = st.selectbox("FICO Score Group", [
    "poor","fair","good","very_good","excellent"
])

employment_status = st.selectbox("Employment Status", [
    "full_time","self_employed","part_time","unemployed","other"
])

employment_sector = st.selectbox("Employment Sector", [
    "communication_services","consumer_discretionary","consumer_staples",
    "energy","financials","health_care","industrials",
    "information_technology","materials","real_estate","utilities","other"
])

lender = st.selectbox("Lender", ["A","B","C"])

# -----------------------------------------------------
# Build feature vector
# -----------------------------------------------------
row = {col: 0 for col in model_columns}

row["Requested_Loan_Amount"] = requested_loan
row["FICO_score"] = fico_score
row["Monthly_Gross_Income"] = monthly_income
row["Monthly_Housing_Payment"] = monthly_housing
row["Ever_Bankrupt_or_Foreclose"] = bankrupt

# Reason
if reason != "cover_an_unexpected_cost":
    dummy = f"Reason_{reason}"
    if dummy in row:
        row[dummy] = 1

# FICO Group
if fico_group != "excellent":
    dummy = f"Fico_Score_group_{fico_group}"
    if dummy in row:
        row[dummy] = 1

# Employment Status
if employment_status in ["part_time", "unemployed"]:
    dummy = f"Employment_Status_{employment_status}"
    if dummy in row:
        row[dummy] = 1

# Employment Sector
if employment_sector != "other":
    dummy = f"Employment_Sector_{employment_sector}"
    if dummy in row:
        row[dummy] = 1

# Lender
if lender in ["B","C"]:
    dummy = f"Lender_{lender}"
    if dummy in row:
        row[dummy] = 1

# Convert to dataframe
input_df = pd.DataFrame([row])

# -----------------------------------------------------
# Apply scaling (this fixes the “always approved” issue)
# -----------------------------------------------------
input_scaled = scaler.transform(input_df)

# -----------------------------------------------------
# Prediction
# -----------------------------------------------------
if st.button("Evaluate Loan Application"):

    pred = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0]

    if pred == 1:
        st.success("🎉 Loan APPROVED!")
        st.write(f"Approval Confidence: **{prob[1]*100:.2f}%**")
        st.balloons()
    else:
        st.error("❌ Loan DENIED")
        st.write(f"Denial Confidence: **{prob[0]*100:.2f}%**")

    st.subheader("Prediction Probabilities")
    col1, col2 = st.columns(2)
    col1.metric("Approval Probability", f"{prob[1]*100:.2f}%")
    col2.metric("Denial Probability", f"{prob[0]*100:.2f}%")
