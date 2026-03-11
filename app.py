import streamlit as st
import pandas as pd
import joblib

# Load model
@st.cache_resource
def load_model():
    return joblib.load("fraud_xgboost_model.pkl")

model = load_model()

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("creditcard.csv")

df = load_data()

st.title("💳 Credit Card Fraud Detection")

features = [
"Time","V1","V2","V3","V4","V5","V6","V7","V8","V9",
"V10","V11","V12","V13","V14","V15","V16","V17","V18","V19",
"V20","V21","V22","V23","V24","V25","V26","V27","V28","Amount"
]

# Initialize session state
for f in features:
    if f not in st.session_state:
        st.session_state[f] = 0.0


# Random sample button
if st.button("Generate Random Transaction"):

    sample = df.sample(1).drop("Class", axis=1)

    for f in features:
        st.session_state[f] = float(sample[f].values[0])
if st.button("Load Real Fraud Transaction"):

    fraud_sample = df[df["Class"] == 1].sample(1).drop("Class", axis=1)

    for f in features:
        st.session_state[f] = float(fraud_sample[f].values[0])

# Input fields
inputs = []
for f in features:
    val = st.number_input(f, key=f)
    inputs.append(val)


# Predict button
if st.button("Predict Transaction"):

    input_df = pd.DataFrame([inputs], columns=features)

    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    st.write("Fraud Probability:", round(prob,3))

    if prediction == 1:
        st.error(f"⚠ Fraud Transaction Detected\nFraud Probability: {prob:.2f}")
    else:
        st.success(f"✅ Legitimate Transaction\nFraud Probability: {prob:.2f}")