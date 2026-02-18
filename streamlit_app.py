import streamlit as st
import pandas as pd
import pickle
import base64

# =====================================
# PAGE CONFIG
# =====================================
st.set_page_config(page_title="Loan Approval System", layout="wide")

st.title("🏦 Loan Approval Prediction System")
st.markdown("### Enter Applicant Details Below")
# Background Image
# -----------------------------
def set_bg(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}

        /* Make all labels black */
        label, .stMarkdown, .stTextInput label, .stNumberInput label,
        .stSelectbox label, .stSlider label {{
            color: black !important;
            font-weight: 600;
        }}

        /* Push buttons to top-left */
        .top-left-buttons {{
            position: fixed;
            top: 10px;
            left: 10px;
            z-index: 1000;
        }}

        .top-left-buttons button {{
            background-color: #4CAF50;
            color: white;
            border: none;
            padding: 8px 16px;
            margin-right: 5px;
            cursor: pointer;
            font-size: 14px;
            border-radius: 5px;
        }}

        .top-left-buttons button:hover {{
            background-color: #45a049;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_bg("loans.jfif")

# =====================================
# LOAD MODEL + SCALER + COLUMNS
# =====================================
with open('finalized_model.sav', 'rb') as f:
    loaded_model = pickle.load(f)

with open('scaler.sav', 'rb') as f:
    sc = pickle.load(f)

with open('model_columns.pkl', 'rb') as f:
    model_columns = pickle.load(f)

# =====================================
# USER INPUT SECTION
# =====================================
col1, col2 = st.columns(2)

with col1:
    person_age = st.number_input("Age", 18, 100, 30)
    person_gender = st.selectbox("Gender", ["Male", "Female"])
    person_education = st.selectbox(
        "Education",
        ["Master", "High School", "Bachelor", "Associate", "Doctorate"]
    )
    person_income = st.number_input("Annual Income", 0.0, 1_000_000.0, 50000.0)
    person_emp_exp = st.number_input("Employment Experience (Years)", 0.0, 50.0, 5.0)
    person_home_ownership = st.selectbox(
        "Home Ownership",
        ["Rent", "Own", "Mortgage", "Other"]
    )

with col2:
    loan_amnt = st.number_input("Loan Amount", 0.0, 1_000_000.0, 10000.0)
    loan_intent = st.selectbox(
        "Loan Purpose",
        ["Personal", "Education", "Medical",
         "Venture", "Homeimprovement", "Debtconsolidation"]
    )
    loan_int_rate = st.number_input("Interest Rate (%)", 0.0, 100.0, 10.0)
    loan_percent_income = st.number_input("Loan Percent Income (0.2 = 20%)", 0.0, 1.0, 0.2)
    cb_person_cred_hist_length = st.number_input("Credit History Length (Years)", 0.0, 50.0, 5.0)
    credit_score = st.number_input("Credit Score", 300.0, 900.0, 650.0)
    previous_loan_defaults_on_file = st.selectbox("Previous Loan Defaults", ["No", "Yes"])


# =====================================
# PREDICTION BUTTON
# =====================================
if st.button("🔍 Predict Loan Status"):

    # ---------------------------------
    # MAP INPUT VALUES
    # ---------------------------------
    gender_map = {'male': 'male', 'female': 'female'}

    education_map = {
        'master': 'Master',
        'high school': 'High School',
        'bachelor': 'Bachelor',
        'associate': 'Associate',
        'doctorate': 'Doctorate'
    }

    home_map = {
        'rent': 'RENT',
        'own': 'OWN',
        'mortgage': 'MORTGAGE',
        'other': 'OTHER'
    }

    loan_intent_map = {
        'personal': 'PERSONAL',
        'education': 'EDUCATION',
        'medical': 'MEDICAL',
        'venture': 'VENTURE',
        'homeimprovement': 'HOMEIMPROVEMENT',
        'debtconsolidation': 'DEBTCONSOLIDATION'
    }

    previous_defaults_map = {'no': 'No', 'yes': 'Yes'}

    # Convert to lowercase before mapping
    person_gender_mapped = gender_map.get(person_gender.lower(), 'male')
    person_education_mapped = education_map.get(person_education.lower(), 'High School')
    person_home_ownership_mapped = home_map.get(person_home_ownership.lower(), 'RENT')
    loan_intent_mapped = loan_intent_map.get(loan_intent.lower(), 'PERSONAL')
    previous_defaults_mapped = previous_defaults_map.get(
        previous_loan_defaults_on_file.lower(), 'No'
    )

    # ---------------------------------
    # CREATE DATAFRAME
    # ---------------------------------
    input_dict = {
        'person_age': person_age,
        'person_income': person_income,
        'person_emp_exp': person_emp_exp,
        'loan_amnt': loan_amnt,
        'loan_int_rate': loan_int_rate,
        'loan_percent_income': loan_percent_income,
        'cb_person_cred_hist_length': cb_person_cred_hist_length,
        'credit_score': credit_score,
        'person_gender': person_gender_mapped,
        'person_education': person_education_mapped,
        'person_home_ownership': person_home_ownership_mapped,
        'loan_intent': loan_intent_mapped,
        'previous_loan_defaults_on_file': previous_defaults_mapped
    }

    input_df = pd.DataFrame([input_dict])

    # ---------------------------------
    # ONE HOT ENCODING
    # ---------------------------------
    input_df = pd.get_dummies(input_df)

    for col in model_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[model_columns]

    # ---------------------------------
    # SCALE INPUT
    # ---------------------------------
    input_scaled = sc.transform(input_df)

    # ---------------------------------
    # PREDICT
    # ---------------------------------
    prediction = loaded_model.predict(input_scaled)
    prediction_prob = loaded_model.predict_proba(input_scaled)[0][1]

    # =====================================
    # FORMAT OUTPUT EXACTLY LIKE CONSOLE
    # =====================================

    if prediction[0] == 1:
        final_decision = "✅ LOAN APPROVED"
    else:
        final_decision = "❌ LOAN NOT APPROVED"

    output_text = f"""
========================================
         LOAN APPLICATION SUMMARY
========================================
Applicant Details:
----------------------------------------
Age:                      {person_age} YEARS
Gender:                   {person_gender.upper()}
Education Level:          {person_education.upper()}
Annual Income:            ${person_income:,.0f}
Employment Experience:    {person_emp_exp} YEARS
Home Ownership:           {person_home_ownership.upper()}

Loan Details:
----------------------------------------
Loan Amount:              ${loan_amnt:,.0f}
Loan Purpose:             {loan_intent.upper()}
Interest Rate:            {loan_int_rate}%
Loan/Income Ratio:        {loan_percent_income*100:.0f}%

Credit Information:
----------------------------------------
Credit History Length:    {cb_person_cred_hist_length} MONTHS
Credit Score:             {credit_score}
Previous Defaults:        {previous_loan_defaults_on_file.upper()}

========================================
          FINAL DECISION
========================================
{final_decision}
"""

    
    # CENTER THE OUTPUT
    center_col1, center_col2, center_col3 = st.columns([1,2,1])
    with center_col2:
        st.code(output_text)
