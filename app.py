# -*- coding: utf-8 -*-
import streamlit as st
import pickle
import pandas as pd
import sklearn  # This is needed for the pickle file to load!

# Load the trained model
# --- Put the Model in Drive First---
with open("my_model.pkl", "rb") as file:
    model_package = pickle.load(file)

# Title for the app
# st.title("Home Equity Loan Approval")
st.markdown(
    "<h1 style='text-align: center; background-color: #ffcccc; padding: 10px; color: #cc0000;'><b>Home Equity Loan Approval</b></h1>",
    unsafe_allow_html=True
)

# Numeric inputs
st.header("Enter Loan Applicant's Details")

# Input fields for numeric values - st = streamlit
loan = st.slider("Loan Amount (LOAN)", min_value=1000, max_value=500000, step=1000)
mortdue = st.slider("Mortgage Due (MORTDUE)", min_value=0.0, max_value=1000000.0, step=1000.0)
value = st.slider("Property Value (VALUE)", min_value=0.0, max_value=1000000.0, step=1000.0)
yoj = st.selectbox("Years at Job (YOJ)", options=list(range(1, 41)))  # Options from 1 to 40
derog = st.number_input("Derogatory Reports (DEROG)", min_value=0, max_value=15, step=1)
delinq = st.selectbox("Delinquent Reports (DELINQ)", options=list(range(0, 15)))  # Options from 0 to 10
clage = st.slider("Age of Oldest Trade Line in Months (CLAGE)", min_value=0.0, max_value=100.0, step=1.0)
ninq = st.slider("Number of Recent Credit Inquiries (NINQ)", min_value=0.0, max_value=15.0, step=1.0)
clno = st.slider("Number of Credit Lines (CLNO)", min_value=0.0, max_value=50.0, step=1.0)
debtinc = st.slider("Debt-to-Income Ratio (DEBTINC)", min_value=0.0, max_value=200.0, step=0.1)

# Categorical inputs with options
reason = st.selectbox("Reason for Loan (REASON)", ["HomeImp", "DebtCon"])
job = st.selectbox("Job Category (JOB)", ["ProfExe", "Other", "Mgr", "Office", "Sales"])

# Create the input data as a DataFrame
input_data = pd.DataFrame({
    "LOAN": [loan],
    "MORTDUE": [mortdue],
    "VALUE": [value],
    "YOJ": [yoj],
    "DEROG": [derog],
    "DELINQ": [delinq],
    "CLAGE": [clage],
    "NINQ": [ninq],
    "CLNO": [clno],
    "DEBTINC": [debtinc],
    "REASON": [reason],
    "JOB": [job]
})

# --- Prepare Data for Prediction ---
# 1. One-hot encode the user's input.
input_data_encoded = pd.get_dummies(input_data, columns=['REASON', 'JOB'])

# 2. Add any "missing" columns the model expects (fill with 0).
model_columns = model.feature_names_in_
for col in model_columns:
    if col not in input_data_encoded.columns:
        input_data_encoded[col] = 0

# 3. Reorder/filter columns to exactly match the model's training data.
input_data_encoded = input_data_encoded[model_columns]

# Predict button
if st.button("Evaluate Loan"):
    # Predict using the loaded model
    prediction = model.predict(input_data_encoded)[0]
    
    # Get prediction probability for confidence display
    prediction_proba = model.predict_proba(input_data_encoded)[0]
    
    # Display result with probability
    if prediction == 1:
        confidence = prediction_proba[1] * 100
        st.write(f"The prediction is: **Bad Loan** 🚫")
        st.write(f"Confidence: {confidence:.2f}%")
    else:
        confidence = prediction_proba[0] * 100
        st.write(f"The prediction is: **Good Loan** 💲")
        st.write(f"Confidence: {confidence:.2f}%")



        """
What happens if the user enters a value not in the training data?

Example: User enters REASON = 'Vacation', but the model only knows 'DebtCon' and 'HomeImp'.

1. pd.get_dummies creates a new column: REASON_Vacation = 1.
2. The code then adds the *known* columns: REASON_DebtCon = 0 and REASON_HomeImp = 0.
3. The final filtering step *drops* the unknown REASON_Vacation column because it's not in the
   model's expected feature list.

Result: The model receives REASON_DebtCon = 0 and REASON_HomeImp = 0, which correctly
treats the unknown 'Vacation' input as "none of the known categories" (i.e., "Other").

Note: Logistic Regression provides probability estimates via predict_proba(), which gives
the confidence level for each prediction. This is displayed to give users insight into
how certain the model is about its classification.
"""
