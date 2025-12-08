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

# Feature list from trained model (matching second file)
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

# Create tabs for better organization
tab1, tab2, tab3 = st.tabs(["📋 Applicant Info", "💼 Financial Details", "🎯 Prediction"])

with tab1:
    st.header("Personal & Employment Information")

    col1, col2 = st.columns(2)

    with col1:
        # Loan Purpose - Mapping display to actual values
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
            list(reason_options.keys())
        )
        reason = reason_options[reason_display]

        # Employment Status
        employment_status_options = {
            "Full Time": "full_time",
            "Part Time": "part_time",
            "Unemployed": "unemployed"
        }
        employment_status_display = st.selectbox(
            "Employment Status",
            list(employment_status_options.keys())
        )
        employment_status = employment_status_options[employment_status_display]

    with col2:
        # Employment Sector
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
            list(employment_sector_options.keys())
        )
        employment_sector = employment_sector_options[employment_sector_display]

        # Lender Selection
        lender = st.selectbox(
            "Preferred Lender",
            ["A", "B", "C"]
        )
        lender_val = lender[0]  # Extract just the letter

with tab2:
    st.header("Financial Information")

    col1, col2 = st.columns(2)

    with col1:
        # FICO Score (numeric)
        fico_score = st.slider(
            "FICO Score",
            min_value=300,
            max_value=850,
            value=650,
            step=5,
            help="Credit score ranging from 300 (poor) to 850 (excellent)"
        )

        # Automatically determine FICO category based on score
        if fico_score >= 800:
            auto_fico_category = "excellent"
            category_display = "Exceptional (800-850)"
            category_color = "#4CAF50"
        elif fico_score >= 740:
            auto_fico_category = "very_good"
            category_display = "Very Good (740-799)"
            category_color = "#8BC34A"
        elif fico_score >= 670:
            auto_fico_category = "good"
            category_display = "Good (670-739)"
            category_color = "#FFC107"
        elif fico_score >= 580:
            auto_fico_category = "fair"
            category_display = "Fair (580-669)"
            category_color = "#FF9800"
        else:
            auto_fico_category = "poor"
            category_display = "Poor (300-579)"
            category_color = "#F44336"

        # Display the category
        st.markdown(
            f"<p style='color: {category_color}; font-weight: bold; font-size: 16px;'>📊 Category: {category_display}</p>",
            unsafe_allow_html=True
        )
        
        # Use the auto-determined category for the model
        fico_group = auto_fico_category

        # Monthly Gross Income
        monthly_income = st.number_input(
            "Monthly Gross Income ($)",
            min_value=0,
            max_value=50000,
            value=5000,
            step=100,
            help="Your total monthly income before taxes"
        )

        # Monthly Housing Payment
        housing_payment = st.number_input(
            "Monthly Housing Payment ($)",
            min_value=0,
            max_value=10000,
            value=1500,
            step=50,
            help="Monthly rent or mortgage payment"
        )

    with col2:
        # Requested Loan Amount
        loan_amount = st.number_input(
            "Requested Loan Amount ($)",
            min_value=500,
            max_value=150000,
            value=50000,
            step=1000,
            help="Amount you're requesting to borrow"
        )

        # Bankruptcy/Foreclosure History
        bankrupt = st.selectbox(
            "Ever Bankrupt or Foreclosed?",
            ["No", "Yes"]
        )
        bankrupt_val = 0 if bankrupt == "No" else 1

        # Show calculated ratios
        if monthly_income > 0:
            dti_ratio = housing_payment / monthly_income
            lti_ratio = loan_amount / (monthly_income * 12)

            st.metric("Debt-to-Income Ratio", f"{dti_ratio:.2%}")
            st.metric("Loan-to-Income Ratio", f"{lti_ratio:.2f}x")

            # Add warnings
            if dti_ratio > 0.43:
                st.warning("⚠️ High DTI ratio (>43%) may reduce approval chances")
            if lti_ratio > 3:
                st.warning("⚠️ High Loan-to-Income ratio (>3x) may reduce approval chances")

with tab3:
    st.header("Loan Approval Prediction")

    # Predict button
    if st.button("🔮 Predict Approval Likelihood", type="primary", use_container_width=True):

        try:
            # Create zero-filled row for model input
            row = {col: 0 for col in model_columns}

            # Set numeric features
            row["Requested_Loan_Amount"] = loan_amount
            row["FICO_score"] = fico_score
            row["Monthly_Gross_Income"] = monthly_income
            row["Monthly_Housing_Payment"] = housing_payment
            row["Ever_Bankrupt_or_Foreclose"] = bankrupt_val

            # Set Reason (reference = cover_an_unexpected_cost)
            if reason != "cover_an_unexpected_cost":
                key = f"Reason_{reason}"
                if key in row:
                    row[key] = 1

            # Set FICO group (reference = excellent)
            if fico_group != "excellent":
                key = f"Fico_Score_group_{fico_group}"
                if key in row:
                    row[key] = 1

            # Set Employment Status (reference = full_time)
            if employment_status != "full_time":
                key = f"Employment_Status_{employment_status}"
                if key in row:
                    row[key] = 1

            # Set Employment Sector (reference = other)
            if employment_sector != "other":
                key = f"Employment_Sector_{employment_sector}"
                if key in row:
                    row[key] = 1

            # Set Lender (reference = A)
            if lender_val in ["B", "C"]:
                key = f"Lender_{lender_val}"
                if key in row:
                    row[key] = 1

            # Convert to DataFrame
            input_df = pd.DataFrame([row])

            # Scale features (REQUIRED)
            input_scaled = scaler.transform(input_df)

            # Get prediction probability
            prediction_proba = model.predict_proba(input_scaled)[0]
            prediction = model.predict(input_scaled)[0]

            # Display results with styling
            st.markdown("---")

            col1, col2, col3 = st.columns([1, 2, 1])

            with col2:
                if prediction == 1:
                    st.success("✅ **LIKELY TO BE APPROVED**")
                    st.balloons()
                else:
                    st.error("❌ **LIKELY TO BE DENIED**")

                # Show probability
                st.metric(
                    "Approval Probability",
                    f"{prediction_proba[1]:.1%}"
                )

                # Progress bar
                st.progress(float(prediction_proba[1]))

                # Show confidence level
                approval_prob = prediction_proba[1]
                if approval_prob > 0.7 or approval_prob < 0.3:
                    confidence = "High"
                    color = "green" if prediction == 1 else "red"
                elif approval_prob > 0.6 or approval_prob < 0.4:
                    confidence = "Medium"
                    color = "orange"
                else:
                    confidence = "Low (Borderline)"
                    color = "gray"

                st.markdown(f"**Confidence Level:** <span style='color: {color};'>{confidence}</span>", unsafe_allow_html=True)

            # Show probabilities
            st.markdown("---")
            st.subheader("📊 Prediction Probabilities")
            prob_col1, prob_col2 = st.columns(2)
            with prob_col1:
                st.metric("Approval", f"{prediction_proba[1]*100:.2f}%")
            with prob_col2:
                st.metric("Denial", f"{prediction_proba[0]*100:.2f}%")

            # Show key factors
            st.markdown("---")
            st.subheader("📊 Key Factors in This Decision")

            factor_col1, factor_col2 = st.columns(2)

            with factor_col1:
                st.markdown("**Positive Factors:**")
                positive_factors = []

                if fico_score >= 700:
                    positive_factors.append(f"✓ Good FICO score ({fico_score})")
                if monthly_income > 0 and housing_payment / monthly_income < 0.36:
                    positive_factors.append(f"✓ Low debt-to-income ratio ({housing_payment / monthly_income:.1%})")
                if monthly_income > 0 and loan_amount / (monthly_income * 12) < 2:
                    positive_factors.append(f"✓ Reasonable loan size ({loan_amount / (monthly_income * 12):.1f}x income)")
                if bankrupt_val == 0:
                    positive_factors.append("✓ No bankruptcy history")
                if employment_status == "full_time":
                    positive_factors.append("✓ Full-time employment")

                if positive_factors:
                    for factor in positive_factors:
                        st.markdown(factor)
                else:
                    st.markdown("_No strong positive factors identified_")

            with factor_col2:
                st.markdown("**Risk Factors:**")
                risk_factors = []

                if fico_score < 640:
                    risk_factors.append(f"⚠ Low FICO score ({fico_score})")
                if monthly_income > 0 and housing_payment / monthly_income > 0.43:
                    risk_factors.append(f"⚠ High debt-to-income ratio ({housing_payment / monthly_income:.1%})")
                if monthly_income > 0 and loan_amount / (monthly_income * 12) > 3:
                    risk_factors.append(f"⚠ Large loan relative to income ({loan_amount / (monthly_income * 12):.1f}x)")
                if bankrupt_val == 1:
                    risk_factors.append("⚠ Bankruptcy/foreclosure history")
                if employment_status == "part_time":
                    risk_factors.append("⚠ Part-time employment")
                if employment_status == "unemployed":
                    risk_factors.append("⚠ Unemployed")

                if risk_factors:
                    for factor in risk_factors:
                        st.markdown(factor)
                else:
                    st.markdown("_No major risk factors identified_")

            # Recommendations
            st.markdown("---")
            st.subheader("💡 Recommendations")

            if prediction == 0:
                st.info(
                    """
                    **To improve your approval chances:**
                    - Improve your credit score by paying bills on time
                    - Reduce your debt-to-income ratio by paying down existing debts
                    - Consider requesting a smaller loan amount
                    - Wait 6-12 months to build a stronger financial profile
                    """
                )
            else:
                # Show expected payout
                payout_map = {'A': 250, 'B': 350, 'C': 150}
                expected_payout = payout_map[lender_val]

                st.success(
                    f"""
                    **Next Steps:**
                    - Your application shows strong approval potential
                    - Expected platform payout if approved: ${expected_payout}
                    - Consider applying to Lender {lender_val} (selected preference)
                    - Ensure all documentation is accurate and complete
                    """
                )

        except Exception as e:
            st.error(f"❌ Prediction error: {str(e)}")
            st.info("Please check that all input values are valid and try again.")
            with st.expander("Show error details"):
                st.code(str(e))

# Sidebar with information
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
    st.markdown("�📧 Contact: kgordon4@ncsu.edu")
