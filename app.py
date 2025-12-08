# loanapp.py
# -*- coding: utf-8 -*-
import streamlit as st
import pickle
import pandas as pd
import numpy as np
import sklearn

# -----------------------------------------------------
# Load Model
# -----------------------------------------------------
try:
    with open("my_model.pkl", "rb") as file:
        model = pickle.load(file)

    # Check if model has feature names
    if hasattr(model, "feature_names_in_"):
        model_columns = model.feature_names_in_
    else:
        st.warning(
            "Model does not contain feature_names_in_. "
            "Make sure your model was trained on a DataFrame."
        )
        model_columns = None

except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# -----------------------------------------------------
# Streamlit Title
# -----------------------------------------------------
st.markdown(
    "<h1 style='text-align: center; background-color: #4CAF50;"
    "padding: 10px; color: white;'><b>Loan Approval Prediction</b></h1>",
    unsafe_allow_html=True
)

st.header("Enter Applicant's Details")

# -----------------------------------------------------
# Numeric Inputs
# -----------------------------------------------------
requested_loan = st.slider("Requested Loan Amount ($)", 1000, 150000, step=1000)
fico_score     = st.slider("FICO Score", 300, 850, step=1)
monthly_income = st.number_input("Monthly Gross Income ($)", 0, 50000, step=100)
monthly_housing = st.number_input("Monthly Housing Payment ($)", 0, 10000, step=50)
applications    = st.number_input("Number of Applications", 1, 10, step=1)

# -----------------------------------------------------
# Categorical Inputs
# -----------------------------------------------------
reason = st.selectbox("Reason for Loan", [
    "debt_conslidation",
    "home_improvement", 
    "major_purchase",
    "credit_card_refinancing",
    "cover_an_unexpected_cost",
    "other"
])

fico_group = st.selectbox("FICO Score Group", [
    "poor","fair","good","very_good","excellent"
])

employment_status = st.selectbox("Employment Status", [
    "full_time","part_time","self_employed","unemployed"
])

employment_sector = st.selectbox("Employment Sector", [
    "information_technology","health_care","consumer_discretionary","energy",
    "materials","utilities","consumer_staples","communication_services",
    "industrials","real_estate","financials"
])

lender = st.selectbox("Lender", ["A","B","C"])

bankrupt = st.radio("Ever Bankrupt or Foreclose?", [0,1],
                    format_func=lambda x: "No" if x==0 else "Yes")

# -----------------------------------------------------
# Build Input DataFrame (GRANTED LOAN REMOVED)
# -----------------------------------------------------
input_data = pd.DataFrame({
    "applications": [applications],
    "Reason": [reason],
    "Requested_Loan_Amount": [requested_loan],   # <-- only this loan amount remains
    "FICO_score": [fico_score],
    "Fico_Score_group": [fico_group],
    "Employment_Status": [employment_status],
    "Employment_Sector": [employment_sector],
    "Monthly_Gross_Income": [monthly_income],
    "Monthly_Housing_Payment": [monthly]()
