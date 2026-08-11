import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("insurance_model.pkl")

st.set_page_config(
    page_title="Life Insurance Lapse Prediction",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ Life Insurance Lapse Prediction")
st.write("Enter policy details to predict whether the policy may lapse.")

st.divider()

# Input fields
col1, col2 = st.columns(2)

with col1:
    policyholder_age = st.number_input(
        "Policyholder Age",
        min_value=18,
        max_value=100,
        value=40
    )

    policy_tenure_months = st.number_input(
        "Policy Tenure (Months)",
        min_value=1,
        value=120
    )

    payment_history_score = st.number_input(
        "Payment History Score",
        min_value=0.0,
        max_value=100.0,
        value=75.0
    )

    predicted_lapse_risk = st.number_input(
        "Predicted Lapse Risk",
        min_value=0.0,
        max_value=1.0,
        value=0.5
    )

    confidence_interval_lower = st.number_input(
        "Confidence Interval Lower",
        min_value=0.0,
        max_value=1.0,
        value=0.3
    )

    confidence_interval_upper = st.number_input(
        "Confidence Interval Upper",
        min_value=0.0,
        max_value=1.0,
        value=0.7
    )

with col2:
    human_override = st.number_input(
        "Human Override",
        min_value=0,
        max_value=1,
        value=0
    )

    response_time_seconds = st.number_input(
        "Response Time (Seconds)",
        min_value=0.0,
        value=10.0
    )

    post_decision_trust_score = st.number_input(
        "Post Decision Trust Score",
        min_value=0.0,
        max_value=100.0,
        value=70.0
    )

    intervention_taken = st.number_input(
        "Intervention Taken",
        min_value=0,
        max_value=1,
        value=0
    )

    decision_delay_bucket = st.number_input(
        "Decision Delay Bucket",
        min_value=0,
        value=1
    )

    external_pressure = st.number_input(
        "External Pressure",
        min_value=0,
        value=1
    )

st.divider()

# Prediction
if st.button("🔮 Predict Lapse", use_container_width=True):

    input_data = pd.DataFrame([{
        "policyholder_age": policyholder_age,
        "policy_tenure_months": policy_tenure_months,
        "payment_history_score": payment_history_score,
        "predicted_lapse_risk": predicted_lapse_risk,
        "confidence_interval_lower": confidence_interval_lower,
        "confidence_interval_upper": confidence_interval_upper,
        "human_override": human_override,
        "response_time_seconds": response_time_seconds,
        "post_decision_trust_score": post_decision_trust_score,
        "intervention_taken": intervention_taken,
        "decision_delay_bucket": decision_delay_bucket,
        "external_pressure": external_pressure
    }])

    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.error("⚠️ Prediction: Policy may LAPSE")
    else:
        st.success("✅ Prediction: Policy may NOT LAPSE")