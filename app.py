import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Appointment No-Show Predictor", page_icon="🏥")

@st.cache_resource
def load_assets():
    model = joblib.load("no_show_model.joblib")
    columns = joblib.load("model_columns.joblib")
    neighbourhoods = joblib.load("neighbourhoods.joblib")
    return model, columns, neighbourhoods

model, columns, neighbourhoods = load_assets()

st.title("🏥 Medical Appointment No-Show Predictor")
st.write("Enter the appointment details to estimate whether the patient is likely to miss it.")

col1, col2 = st.columns(2)
with col1:
    age = st.slider("Age", 0, 100, 30)
    gender = st.selectbox("Gender", ["Female", "Male"])
    neighbourhood = st.selectbox("Neighbourhood", neighbourhoods)
    waiting_days = st.slider("Days between booking and appointment", 0, 180, 7)
with col2:
    sms = st.checkbox("Reminder SMS received")
    scholarship = st.checkbox("Enrolled in welfare scholarship")
    hypertension = st.checkbox("Hypertension")
    diabetes = st.checkbox("Diabetes")
    alcoholism = st.checkbox("Alcoholism")
    handcap = st.selectbox("Handicap level (0 = none, 4 = highest)", [0, 1, 2, 3, 4])

if st.button("Predict"):
    row = pd.DataFrame(np.zeros((1, len(columns))), columns=columns)
    row["Age"] = age
    row["Gender"] = 1 if gender == "Male" else 0
    row["Scholarship"] = int(scholarship)
    row["Hipertension"] = int(hypertension)
    row["Diabetes"] = int(diabetes)
    row["Alcoholism"] = int(alcoholism)
    row["Handcap"] = handcap
    row["SMS_received"] = int(sms)
    row["waiting_days"] = waiting_days
    nb_col = f"Neighbourhood_{neighbourhood}"
    if nb_col in columns:
        row[nb_col] = 1

    score = model.predict_proba(row)[0][1]
    pred = model.predict(row)[0]

    if pred == 1:
        st.error("⚠️ High risk of no-show")
        st.write("Suggestion: send a reminder or call the patient before the appointment.")
    else:
        st.success("✅ Likely to attend")
    st.metric("Risk score", f"{score:.0%}")
    st.caption("The score comes from a model trained with balanced class weights, so treat it as a ranking score rather than an exact probability.")
