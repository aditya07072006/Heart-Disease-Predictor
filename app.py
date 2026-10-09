import streamlit as st

from model import FEATURE_COLUMNS, predict_likelihood

st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️")
st.title("Heart Disease Predictor (KNN)")
st.write("Estimate the likelihood of heart disease using a K-Nearest Neighbors model.")

col1, col2 = st.columns(2)
with col1:
    age = st.slider("Age", 20, 90, 50)
    sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
    cp = st.selectbox("Chest Pain Type (cp)", options=[0, 1, 2, 3], index=0)
    trestbps = st.slider("Resting Blood Pressure (trestbps)", 80, 220, 130)
    chol = st.slider("Serum Cholesterol (chol)", 100, 600, 240)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", options=[0, 1], index=0)
    restecg = st.selectbox("Resting ECG (restecg)", options=[0, 1, 2], index=1)

with col2:
    thalach = st.slider("Max Heart Rate (thalach)", 70, 220, 150)
    exang = st.selectbox("Exercise Induced Angina (exang)", options=[0, 1], index=0)
    oldpeak = st.slider("ST Depression (oldpeak)", 0.0, 6.0, 1.0, step=0.1)
    slope = st.selectbox("Slope", options=[0, 1, 2], index=1)
    ca = st.selectbox("Number of Major Vessels (ca)", options=[0, 1, 2, 3], index=0)
    thal = st.selectbox("Thal", options=[1, 2, 3], index=1)

if st.button("Predict"):
    values = {
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal,
    }
    likelihood = predict_likelihood(values)
    st.metric("Predicted likelihood of heart disease", f"{likelihood * 100:.1f}%")
    st.caption("Higher percentages indicate greater model-estimated likelihood.")
