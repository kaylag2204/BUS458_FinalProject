# Initialize session state
if 'current_tab' not in st.session_state:
    st.session_state.current_tab = 0

tab_names = ["📋 Applicant Info", "💼 Financial Details", "🎯 Prediction"]

# Create tabs
tabs = st.tabs(tab_names)

# Map tab index for auto-selection
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
        if st.button("🔮 Predict Approval Likelihood", key="predict_button"):
            st.success("Prediction logic goes here...")

        col1, col2, col3 = st.columns([1,1,1])
        with col1:
            if st.button("⬅️ Back", key="back_to_financials"):
                st.session_state.current_tab = 1
