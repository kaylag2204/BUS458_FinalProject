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
    # Reason (reference = cover_an_unexpected_cost)
    'Reason_credit_card_refinancing', 'Reason_debt_conslidation',
    'Reason_home_improvement', 'Reason_major_purchase', 'Reason_other',
    # FICO group (reference = excellent)
    'Fico_Score_group_fair', 'Fico_Score_group_good',
    'Fico_Score_group_poor', 'Fico_Score_group_very_good',
    # Employment Status (reference = full_time)
    'Employment_Status_part_time', 'Employment_Status_unemployed',
    # Employment Sector (reference = other)
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

# Show model info
st.markdown(
    f"<p style='text-align: center; font-size: 14px; color: #666;'>Powered by {model_type}</p>",
    unsafe_allow_html=True
)

# Initialize session state for current tab
if 'current_tab' not in st.session_state:
    st.session_state.current_tab = 0

tab_names = ["📋 Applicant Info", "💼 Financial Details", "🎯 Prediction"]

# Create tabs
tabs = st.tabs(tab_names)

# ---------------------- TAB 1: Applicant Info ----------------------
with tabs[0]:
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
        reason_display = st.selectbox(
            "Reason for Loan",
            list(reason_options.keys()),
            key="reason"
        )
        st.session_state.reason_value = reason_options[reason_display]

        employment_status_options = {
            "Full Time": "full_time",
            "Part Time": "part_time",
            "Unemployed": "unemployed"
        }
        employment_status_display = st.selectbox(
            "Employment Status",
            list(employment_status_options.keys()),
            key="employment_status"
        )
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
        employment_sector_display = st.selectbox(
            "Employment Sector",
            list(employment_sector_options.keys()),
            key="employment_sector"
        )
        st.session_state.employment_sector_value = employment_sector_options[employment_sector_display]

        lender = st.selectbox(
            "Preferred Lender",
            ["A", "B", "C"],
            key="lender"
        )
        st.session_state.lender_value = lender

    # Navigation
    col1, col2, col3 = st.columns([1,1,1])
    with col3:
        if st.button("Next ➡️", key="to_financials"):
            st.session_state.current_tab = 1
            st.experimental_rerun()

# ---------------------- TAB 2: Financial Details ----------------------
with tabs[1]:
    st.header("Financial Information")

    col1, col2 = st.columns(2)
    with col1:
        fico_score = st.slider(
            "FICO Score",
            min_value=300,
            max_value=850,
            value=st.session_state.get('fico_score_value', 650),
            step=5,
            help="Credit score ranging from 300 (poor) to 850 (excellent)",
            key="fico_score"
        )
        st.session_state.fico_score_value = fico_score

        # Determine FICO category
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

        monthly_income = st.number_input(
            "Monthly Gross Income ($)",
            min_value=0,
            max_value=50000,
            value=st.session_state.get('monthly_income_value', 5000),
            step=100,
            key="monthly_income"
        )
        st.session_state.monthly_income_value = monthly_income

        housing_payment = st.number_input(
            "Monthly Housing Payment ($)",
            min_value=0,
            max_value=10000,
            value=st.session_state.get('housing_payment_value', 1500),
            step=50,
            key="housing_payment"
        )
        st.session_state.housing_payment_value = housing_payment

    with col2:
        loan_amount = st.number_input(
            "Requested Loan Amount ($)",
            min_value=500,
            max_value=150000,
            value=st.session_state.get('loan_amount_value', 50000),
            step=1000,
            key="loan_amount"
        )
        st.session_state.loan_amount_value = loan_amount

        bankrupt = st.selectbox(
            "Ever Bankrupt or Foreclosed?",
            ["No", "Yes"],
            key="bankrupt"
        )
        st.session_state.bankrupt_value = 0 if bankrupt == "No" else 1

        # Show ratios
        if monthly_income > 0:
            dti_ratio = housing_payment / monthly_income
            lti_ratio = loan_amount / (monthly_income * 12)
            st.metric("Debt-to-Income Ratio", f"{dti_ratio:.2%}")
            st.metric("Loan-to-Income Ratio", f"{lti_ratio:.2f}x")

    # Navigation
    col1, col2, col3 = st.columns([1,1,1])
    with col1:
        if st.button("⬅️ Back", key="back_to_applicant"):
            st.session_state.current_tab = 0
            st.experimental_rerun()
    with col3:
        if st.button("Next ➡️", key="to_prediction"):
            st.session_state.current_tab = 2
            st.experimental_rerun()

# ---------------------- TAB 3: Prediction ----------------------
with tabs[2]:
    st.header("Loan Approval Prediction")

    if st.button("🔮 Predict Approval Likelihood", key="predict_button"):
        # Prediction logic (your existing code)
        st.success("Prediction logic would run here...")

    # Navigation
    col1, col2, col3 = st.columns([1,1,1])
    with col1:
        if st.button("⬅️ Back", key="back_to_financials"):
            st.session_state.current_tab = 1
            st.experimental_rerun()

# Sidebar
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
