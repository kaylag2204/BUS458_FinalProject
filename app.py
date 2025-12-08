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
# -----------------------------------------------------
# Load Model
# -----------------------------------------------------
try:
    with open("loan_model.pkl", "rb") as file:
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
granted_loan   = st.slider("Granted Loan Amount ($)",   0,     150000, step=1000)
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
# Build Input DataFrame
# -----------------------------------------------------
input_data = pd.DataFrame({
    "applications": [applications],
    "Reason": [reason],
    "Granted_Loan_Amount": [granted_loan],
    "Requested_Loan_Amount": [requested_loan],
    "FICO_score": [fico_score],
    "Fico_Score_group": [fico_group],
    "Employment_Status": [employment_status],
    "Employment_Sector": [employment_sector],
    "Monthly_Gross_Income": [monthly_income],
    "Monthly_Housing_Payment": [monthly_housing],
    "Ever_Bankrupt_or_Foreclose": [bankrupt],
    "Lender": [lender]
})

# One-hot encode
categorical_columns = [
    "Reason", "Fico_Score_group", "Employment_Status",
    "Employment_Sector", "Lender"
]

input_encoded = pd.get_dummies(input_data, columns=categorical_columns, drop_first=True)

# -----------------------------------------------------
# Align with model columns
# -----------------------------------------------------
if model_columns is not None:

    # Add missing columns
    for col in model_columns:
        if col not in input_encoded.columns:
            input_encoded[col] = 0

    # Ensure ordering
    input_encoded = input_encoded[model_columns]

else:
    st.error("Cannot align features — model_columns missing.")
    st.stop()

# -----------------------------------------------------
# Prediction
# -----------------------------------------------------
if st.button("Evaluate Loan Application"):
    prediction = model.predict(input_encoded)[0]

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(input_encoded)[0]
    else:
        proba = [np.nan, np.nan]

    if prediction == 1:
        st.success("🎉 **Loan APPROVED!**")
        st.write(f"Confidence: **{proba[1]*100:.2f}%**")
        st.balloons()
    else:
        st.error("❌ **Loan DENIED**")
        st.write(f"Confidence: **{proba[0]*100:.2f}%**")

    # Extra details
    st.subheader("Prediction Details")
    col1, col2 = st.columns(2)
    col1.metric("Approval Probability", f"{proba[1]*100:.2f}%")
    col2.metric("Denial Probability",   f"{proba[0]*100:.2f}%")
