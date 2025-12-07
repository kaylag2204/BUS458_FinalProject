# -*- coding: utf-8 -*-
import streamlit as st
import pickle
import pandas as pd
import numpy as np
import sklearn  # This is needed for the pickle file to load!

# Load the trained model
# --- save the model first---
with open("/content/Pickle File/my_model.pkl", "rb") as file:
    model_package = pickle.load(file)

# Extract components from the model package
model = model_package['model']
model_type = model_package['model_type']
threshold = model_package['threshold']
feature_names = model_package['feature_names']

# Get scaler if it exists (for Logistic Regression)
scaler = model_package.get('scaler', None)

# Title for the app
st.markdown(
    "<h1 style='text-align: center; background-color: #4CAF50; padding: 20px; color: white; border-radius: 10px;'><b>💰 Personal Loan Approval Predictor</b></h1>",
    unsafe_allow_html=True
)

st.markdown(
    f"<p style='text-align: center; font-size: 14px; color: #666;'>Powered by {model_type} | Optimized Threshold: {threshold:.3f}</p>",
    unsafe_allow_html=True
)

# Create tabs for better organization
tab1, tab2, tab3 = st.tabs(["📋 Applicant Info", "💼 Financial Details", "🎯 Prediction"])

with tab1:
    st.header("Personal & Employment Information")

    col1, col2 = st.columns(2)

    with col1:
        # Loan Purpose
        reason = st.selectbox(
            "Reason for Loan",
            ["home_improvement", "debt_consolidation", "major_purchase",
             "cover_an_unexpected_cost", "credit_card_refinancing", "business"]
        )

        # Employment Status
        employment_status = st.selectbox(
            "Employment Status",
            ["full_time", "part_time", "self_employed"]
        )

        # FICO Score Group
        fico_group = st.selectbox(
            "FICO Score Category",
            ["poor", "fair", "good", "very_good", "excellent"]
        )

    with col2:
        # Employment Sector
        employment_sector = st.selectbox(
            "Employment Sector",
            ["information_technology", "financials", "healthcare", "industrials",
             "consumer_discretionary", "materials", "energy", "utilities",
             "communication_services", "real_estate", "consumer_staples", "Unknown"]
        )

        # Bankruptcy/Foreclosure History
        bankrupt = st.selectbox(
            "Ever Bankrupt or Foreclosed?",
            ["No", "Yes"]
        )
        bankrupt_val = 0 if bankrupt == "No" else 1

        # Lender Selection
        lender = st.selectbox(
            "Preferred Lender",
            ["A ($250 payout)", "B ($350 payout)", "C ($150 payout)"]
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
            min_value=5000,
            max_value=500000,
            value=50000,
            step=1000,
            help="Amount you're requesting to borrow"
        )

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

        # Calculate engineered features
        debt_to_income = housing_payment / monthly_income if monthly_income > 0 else 0
        loan_to_income = loan_amount / (monthly_income * 12) if monthly_income > 0 else 0
        fico_income_interaction = fico_score * monthly_income

        # Create the input data as a DataFrame
        input_data = pd.DataFrame({
            "Requested_Loan_Amount": [loan_amount],
            "FICO_score": [fico_score],
            "Monthly_Gross_Income": [monthly_income],
            "Monthly_Housing_Payment": [housing_payment],
            "Ever_Bankrupt_or_Foreclose": [bankrupt_val],
            "Debt_to_Income_Ratio": [debt_to_income],
            "Loan_to_Income_Ratio": [loan_to_income],
            "FICO_Income_Interaction": [fico_income_interaction],
            "Reason": [reason],
            "Employment_Status": [employment_status],
            "Employment_Sector": [employment_sector],
            "Lender": [lender_val]
        })

        # One-hot encode the categorical variables
        input_data_encoded = pd.get_dummies(input_data, drop_first=True)

        # Add any missing columns the model expects (fill with 0)
        for col in feature_names:
            if col not in input_data_encoded.columns:
                input_data_encoded[col] = 0

        # Reorder columns to match the model's training data
        input_data_encoded = input_data_encoded[feature_names]

        # Scale if using Logistic Regression
        if scaler is not None:
            input_data_scaled = scaler.transform(input_data_encoded)
        else:
            input_data_scaled = input_data_encoded

        # Get prediction probability
        prediction_proba = model.predict_proba(input_data_scaled)[0][1]

        # Apply the optimized threshold
        prediction = 1 if prediction_proba >= threshold else 0

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
                f"{prediction_proba:.1%}",
                delta=f"{(prediction_proba - threshold):.1%} vs threshold"
            )

            # Progress bar
            st.progress(prediction_proba)

            # Show confidence level
            if prediction_proba > 0.7 or prediction_proba < 0.3:
                confidence = "High"
                color = "green" if prediction == 1 else "red"
            elif prediction_proba > 0.6 or prediction_proba < 0.4:
                confidence = "Medium"
                color = "orange"
            else:
                confidence = "Low (Borderline)"
                color = "gray"

            st.markdown(f"**Confidence Level:** <span style='color: {color};'>{confidence}</span>", unsafe_allow_html=True)

        # Show key factors
        st.markdown("---")
        st.subheader("📊 Key Factors in This Decision")

        factor_col1, factor_col2 = st.columns(2)

        with factor_col1:
            st.markdown("**Positive Factors:**")
            positive_factors = []

            if fico_score >= 700:
                positive_factors.append(f"✓ Good FICO score ({fico_score})")
            if debt_to_income < 0.36:
                positive_factors.append(f"✓ Low debt-to-income ratio ({debt_to_income:.1%})")
            if loan_to_income < 2:
                positive_factors.append(f"✓ Reasonable loan size ({loan_to_income:.1f}x income)")
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
            if debt_to_income > 0.43:
                risk_factors.append(f"⚠ High debt-to-income ratio ({debt_to_income:.1%})")
            if loan_to_income > 3:
                risk_factors.append(f"⚠ Large loan relative to income ({loan_to_income:.1f}x)")
            if bankrupt_val == 1:
                risk_factors.append("⚠ Bankruptcy/foreclosure history")
            if employment_status == "part_time":
                risk_factors.append("⚠ Part-time employment")

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
                - Consider applying to {lender_val} (selected preference)
                - Ensure all documentation is accurate and complete
                """
            )

# Sidebar with information
with st.sidebar:
    st.header("ℹ️ About This Tool")
    st.markdown(
        """
        This loan approval predictor uses machine learning to estimate
        the likelihood of loan approval based on applicant information.

        **Model Details:**
        - Algorithm: {model_type}
        - Optimized Threshold: {threshold:.3f}
        - Training Data: 100,000 applications
        - Recall: ~60-73%
        - Precision: ~24-28%

        **Lender Payouts:**
        - Lender A: $250 per approval
        - Lender B: $350 per approval
        - Lender C: $150 per approval

        **Important Notes:**
        - This is a prediction tool, not a guarantee
        - Actual approval depends on lender-specific criteria
        - Results should be used as guidance only
        """
    )

    st.markdown("---")
    st.markdown("**Created for BUS 458 Final Project**")
    st.markdown("📧 Contact: kgordon4@ncsu.edu")

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #888; font-size: 12px;'>⚠️ For educational purposes only. Not actual financial advice.</p>",
    unsafe_allow_html=True
)
