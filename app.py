# -*- coding: utf-8 -*-
import streamlit as st
import pickle
import pandas as pd
import numpy as np
import os

# Title for the app
st.markdown(
    "<h1 style='text-align: center; background-color: #4CAF50; padding: 20px; color: white; border-radius: 10px;'><b>💰 Personal Loan Approval Predictor</b></h1>",
    unsafe_allow_html=True
)

# Load the trained model and scaler
try:
    with open("my_model.pkl", "rb") as file:
        model = pickle.load(file)
    
    with open("loan_scaler.pkl", "rb") as file:
        scaler = pickle.load(file)
    
    model_type = "Logistic Regression"
    
except FileNotFoundError as e:
    st.error(f"❌ Required file not found: {str(e)}")
    st.info("Please ensure 'my_model.pkl' and 'loan_scaler.pkl' are in the same directory as the app.py script.")
    st.stop()
except Exception as e:
    st.error(f"❌ Error loading model or scaler: {str(e)}")
    st.stop()

# Feature list from trained model
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
    'Employment_Sector_industrials', 'Employment_Sector_information_technology',
    'Employment_Sector_materials', 'Employment_Sector_real_estate',
    'Employment_Sector_utilities',
    'Lender_B', 'Lender_C'
]

# Show model info
st.markdown(
    f"<p style='text-align: center; font-size: 14px; color: #666;'>Powered by {model_type}</p>",
    unsafe_allow_html=True
)

# Initialize session state for tab navigation
if 'current_tab' not in st.session_state:
    st.session_state.current_tab = 0

tab_names = ["📋 Applicant Info", "💼 Financial Details", "🎯 Prediction"]
tabs = st.tabs(tab_names)
current_tab = st.session_state.current_tab

# ------------------ TAB 1 ------------------
with tabs[0]:
    if current_tab == 0:
        st.header("Personal & Employment Information")
        col1, col2 = st.columns(2)
        with col1:
            reason_options = {
                "Cover an Unexpected Cost": "cover_an_unexpected_cost",
                "Credit Card Refinancing": "credit_card_refinancing",
                "Debt Consolidation": "debt_conslidation",
                "Home Improvement": "home_improvement",
                "Major Purchase": "major_purchase",
                "Other": "other"
            }
            reason_display = st.selectbox("Reason for Loan", list(reason_options.keys()), key="reason")
            st.session_state.reason_value = reason_options[reason_display]

            employment_status_options = {
                "Full Time": "full_time",
                "Part Time": "part_time",
                "Unemployed": "unemployed"
            }
            employment_status_display = st.selectbox("Employment Status", list(employment_status_options.keys()), key="employment_status")
            st.session_state.employment_status_value = employment_status_options[employment_status_display]

        with col2:
            employment_sector_options = {
                "Other": "other",
                "Communication Services": "communication_services",
                "Consumer Discretionary": "consumer_discretionary",
                "Consumer Staples": "consumer_staples",
                "Energy": "energy",
                "Financials": "financials",
                "Health Care": "health_care",
                "Industrials": "industrials",
                "Information Technology": "information_technology",
                "Materials": "materials",
                "Real Estate": "real_estate",
                "Utilities": "utilities"
            }
            employment_sector_display = st.selectbox("Employment Sector", list(employment_sector_options.keys()), key="employment_sector")
            st.session_state.employment_sector_value = employment_sector_options[employment_sector_display]

            lender = st.selectbox("Preferred Lender", ["A","B","C"], key="lender")
            st.session_state.lender_value = lender

        # Navigation
        col1, col2, col3 = st.columns([1,1,1])
        with col3:
            if st.button("Next ➡️", key="to_financials"):
                st.session_state.current_tab = 1

# ------------------ TAB 2 ------------------
with tabs[1]:
    if current_tab == 1:
        st.header("Financial Information")
        col1, col2 = st.columns(2)
        with col1:
            fico_score = st.slider("FICO Score", 300, 850, st.session_state.get('fico_score_value', 650), step=5, key="fico_score")
            st.session_state.fico_score_value = fico_score

            if fico_score >= 800:
                auto_fico_category = "excellent"
            elif fico_score >= 740:
                auto_fico_category = "very_good"
            elif fico_score >= 670:
                auto_fico_category = "good"
            elif fico_score >= 580:
                auto_fico_category = "fair"
            else:
                auto_fico_category = "poor"
            st.session_state.fico_group_value = auto_fico_category

            monthly_income = st.number_input("Monthly Gross Income ($)", 0, 50000, st.session_state.get('monthly_income_value', 5000), step=100, key="monthly_income")
            st.session_state.monthly_income_value = monthly_income

            housing_payment = st.number_input("Monthly Housing Payment ($)", 0, 10000, st.session_state.get('housing_payment_value', 1500), step=50, key="housing_payment")
            st.session_state.housing_payment_value = housing_payment

        with col2:
            loan_amount = st.number_input("Requested Loan Amount ($)", 500, 150000, st.session_state.get('loan_amount_value', 50000), step=1000, key="loan_amount")
            st.session_state.loan_amount_value = loan_amount

            bankrupt = st.selectbox("Ever Bankrupt or Foreclosed?", ["No","Yes"], key="bankrupt")
            st.session_state.bankrupt_value = 0 if bankrupt=="No" else 1

            # Show calculated ratios
            if monthly_income > 0:
                dti_ratio = housing_payment / monthly_income
                lti_ratio = loan_amount / (monthly_income * 12)
                st.metric("Debt-to-Income Ratio", f"{dti_ratio:.2%}")
                st.metric("Loan-to-Income Ratio", f"{lti_ratio:.2f}x")
                if dti_ratio > 0.43:
                    st.warning("⚠️ High DTI ratio (>43%) may reduce approval chances")
                if lti_ratio > 3:
                    st.warning("⚠️ High Loan-to-Income ratio (>3x) may reduce approval chances")

        # Navigation
        col1, col2, col3 = st.columns([1,1,1])
        with col1:
            if st.button("⬅️ Back", key="back_to_applicant"):
                st.session_state.current_tab = 0
        with col3:
            if st.button("Next ➡️", key="to_prediction"):
                st.session_state.current_tab = 2

# ------------------ TAB 3 ------------------
with tabs[2]:
    if current_tab == 2:
        st.header("Loan Approval Prediction")

        # Predict button
        if st.button("🔮 Predict Approval Likelihood", type="primary", use_container_width=True):
            try:
                # Build input row
                row = {col:0 for col in model_columns}
                row["Requested_Loan_Amount"] = st.session_state.loan_amount_value
                row["FICO_score"] = st.session_state.fico_score_value
                row["Monthly_Gross_Income"] = st.session_state.monthly_income_value
                row["Monthly_Housing_Payment"] = st.session_state.housing_payment_value
                row["Ever_Bankrupt_or_Foreclose"] = st.session_state.bankrupt_value

                # One-hot encoding
                reason = st.session_state.reason_value
                if reason != "cover_an_unexpected_cost":
                    key = f"Reason_{reason}"
                    if key in row: row[key] = 1

                fico_group = st.session_state.fico_group_value
                if fico_group != "excellent":
                    key = f"Fico_Score_group_{fico_group}"
                    if key in row: row[key] = 1

                emp_status = st.session_state.employment_status_value
                if emp_status != "full_time":
                    key = f"Employment_Status_{emp_status}"
                    if key in row: row[key] = 1

                emp_sector = st.session_state.employment_sector_value
                if emp_sector != "other":
                    key = f"Employment_Sector_{emp_sector}"
                    if key in row: row[key] = 1

                lender_val = st.session_state.lender_value
                if lender_val in ["B","C"]:
                    key = f"Lender_{lender_val}"
                    if key in row: row[key] = 1

                # Predict
                input_df = pd.DataFrame([row])
                input_scaled = scaler.transform(input_df)
                prediction_proba = model.predict_proba(input_scaled)[0]
                prediction = model.predict(input_scaled)[0]

                # Display results
                st.markdown("---")
                col1, col2, col3 = st.columns([1,2,1])
                with col2:
                    if prediction == 1:
                        st.success("✅ **LIKELY TO BE APPROVED**")
                        st.balloons()
                    else:
                        st.error("❌ **LIKELY TO BE DENIED**")

                    st.metric("Approval Probability", f"{prediction_proba[1]:.1%}")
                    st.progress(float(prediction_proba[1]))

                    # Confidence
                    approval_prob = prediction_proba[1]
                    if approval_prob > 0.7 or approval_prob < 0.3:
                        confidence = "High"
                        color = "green" if prediction==1 else "red"
                    elif approval_prob > 0.6 or approval_prob < 0.4:
                        confidence = "Medium"
                        color = "orange"
                    else:
                        confidence = "Low (Borderline)"
                        color = "gray"
                    st.markdown(f"**Confidence Level:** <span style='color: {color};'>{confidence}</span>", unsafe_allow_html=True)

            except Exception as e:
                st.error(f"❌ Prediction error: {str(e)}")
                st.info("Please check all inputs and try again.")
                with st.expander("Error details"):
                    st.code(str(e))

        # Navigation
        col1, col2, col3 = st.columns([1,1,1])
        with col1:
            if st.button("⬅️ Back", key="back_to_financials"):
                st.session_state.current_tab = 1

# ------------------ Sidebar ------------------
with st.sidebar:
    st.header("ℹ️ About This Tool")
    st.markdown(
        f"""
        This loan approval predictor uses machine learning to estimate
        the likelihood of loan approval based on applicant information.

        **Model Details:**
        - Algorithm: {model_type}
        - Training Data: ~100,000 applications
        """
    )
    st.markdown("---")
    st.markdown("**Created for BUS 458 Final Project**")
    st.markdown("📧 Contact: kgordon4@ncsu.edu")
