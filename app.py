import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Heart Disease Prediction", page_icon="🩺", layout="centered", initial_sidebar_state="collapsed")

@st.cache_resource
def load_artifacts():
    return (joblib.load("knn_heart_model.pkl"),
            joblib.load("heart_scaler.pkl"),
            joblib.load("heart_columns.pkl"))

try:
    model, scaler, expected_columns = load_artifacts()
except Exception as e:
    st.error("Could not load the trained model files.")
    st.code(str(e))
    st.stop()

st.markdown("""
<style>
:root { color-scheme: light; }
.stApp { background:linear-gradient(145deg,#f8fbff 0%,#edf4fc 55%,#e8f1fb 100%); }
[data-testid="stHeader"], [data-testid="stToolbar"] { background:transparent!important; }
[data-testid="stToolbar"] { visibility:hidden; }
.block-container { max-width: 780px; padding: 2.4rem 1rem 3rem; }
.hero { text-align:center; margin-bottom: 2.3rem; }
.hero-kicker { display:inline-block; color:#2563a8; background:#e7f1ff; border:1px solid #d3e5fb; border-radius:999px; padding:.3rem .7rem; font-size:.62rem; font-weight:800; letter-spacing:.12em; margin-bottom:.85rem; }
.hero h1 { color:#102a56; font-size:2.3rem; letter-spacing:-.025em; font-weight:800; margin:0 0 .4rem; }
.hero p { color:#607a9d; font-size:.88rem; margin:0; }
.card { background:rgba(255,255,255,.97); border:1px solid #dce7f4; border-radius:20px; padding:1.65rem 1.55rem 1.45rem; box-shadow:0 20px 45px rgba(37,76,121,.12); }
.section-head { display:flex; align-items:center; gap:13px; margin-bottom:1.35rem; }
.icon-box { width:46px; height:46px; border-radius:14px; background:#e4efff; border:1px solid #d1e4fb; display:flex; align-items:center; justify-content:center; font-size:23px; }
.section-head h2 { color:#142f57; margin:0; font-size:1.35rem; line-height:1.1; }
.section-head p { color:#6a82a4; margin:3px 0 0; font-size:.78rem; }
div[data-baseweb="input"]>div, div[data-baseweb="select"]>div { border-radius:10px!important; border:1px solid #cbd9ea!important; background:#fff!important; min-height:38px; box-shadow:0 1px 2px rgba(30,65,105,.03); transition:box-shadow .15s ease,border-color .15s ease; }
div[data-baseweb="input"]>div:focus-within, div[data-baseweb="select"]>div:focus-within { border-color:#4b83d1!important; box-shadow:0 0 0 3px rgba(75,131,209,.14)!important; }
div[data-baseweb="input"] input { color:#173866!important; caret-color:#2867e8!important; font-size:.78rem!important; opacity:1!important; }
div[data-baseweb="input"] input::placeholder { color:#9aaec8!important; opacity:1!important; }
div[data-baseweb="select"]>div, div[data-baseweb="select"] [role="option"], div[data-baseweb="select"] input { color:#173866!important; font-size:.78rem!important; opacity:1!important; }
div[data-baseweb="select"] svg { fill:#173866!important; }
div[data-baseweb="input"] button { display:none!important; }
div[data-baseweb="input"] input[type="number"] { appearance:textfield!important; -moz-appearance:textfield!important; }
div[data-baseweb="input"] input[type="number"]::-webkit-inner-spin-button,
div[data-baseweb="input"] input[type="number"]::-webkit-outer-spin-button { appearance:none!important; -webkit-appearance:none!important; margin:0!important; }
.stTextInput label, .stSelectbox label { color:#244b78!important; font-size:.72rem!important; font-weight:700!important; margin-bottom:.3rem!important; }
.stTextInput, .stSelectbox { margin-bottom:.25rem; }
div.stButton, div[data-testid="stFormSubmitButton"] { width:100%; }
div.stButton > button, div[data-testid="stFormSubmitButton"] button { width:100%!important; border:0; border-radius:11px; padding:.78rem 1rem; font-size:.84rem; font-weight:800; color:#fff; background:linear-gradient(100deg,#1769d1,#3949db); box-shadow:0 10px 20px rgba(45,88,202,.24); transition:transform .15s ease,box-shadow .15s ease,filter .15s ease; }
div.stButton > button:hover, div[data-testid="stFormSubmitButton"] button:hover { border:0; color:#fff; filter:brightness(1.05); transform:translateY(-1px); box-shadow:0 12px 24px rgba(45,88,202,.3); }
.result-card { margin-top:1.4rem; padding:1.35rem 1.45rem; border-radius:16px; background:#f8fbff; border:1px solid #d8e6f6; box-shadow:0 10px 24px rgba(36,76,122,.07); }
.result-card h3 { color:#142f57; margin:.1rem 0 .15rem; font-size:1.15rem; }
.result-subtitle { color:#7185a1; font-size:.76rem; margin:0 0 1rem; }
.metric-grid { display:grid; grid-template-columns:1fr 1fr; gap:.8rem; margin-bottom:1rem; }
.metric { background:#fff; border:1px solid #e2eaf4; border-radius:12px; padding:.9rem 1rem; }
.metric-label { color:#667d9c; font-size:.75rem; margin-bottom:.25rem; }
.metric-value { color:#2867e8; font-size:1.8rem; font-weight:800; line-height:1; }
.metric-note { color:#8a9bb3; font-size:.7rem; margin-top:.25rem; }
.risk-banner { border-radius:11px; padding:.8rem 1rem; font-weight:800; font-size:1.05rem; margin-bottom:.85rem; }
.risk-high { color:#a82e2e; background:#ffe5e5; border:1px solid #ffcaca; }
.risk-low { color:#14734b; background:#def7e9; border:1px solid #bcebd1; }
.result-copy { color:#536b8c; font-size:.8rem; line-height:1.55; }
.result-list { color:#536b8c; font-size:.78rem; line-height:1.7; margin:.35rem 0 .75rem; }
.result-section-title { color:#263d60; font-size:.8rem; font-weight:800; margin:1rem 0 .25rem; }
.disclaimer { text-align:center; color:#7185a1; font-size:.72rem; margin-top:1.1rem; }
@media (max-width: 640px) { .block-container { padding-top:1.4rem; } .hero { margin-bottom:1.7rem; } .hero h1 { font-size:1.8rem; } .card { padding:1.2rem 1rem 1.1rem; } .metric-grid { grid-template-columns:1fr; } }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="hero"><div class="hero-kicker">HEART HEALTH SCREENING</div><h1>Heart Disease Prediction</h1><p>Complete the patient profile to generate a model-based risk estimate.</p></div>', unsafe_allow_html=True)
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="section-head"><div class="icon-box">🩺</div><div><h2>Patient Information</h2><p>Enter the patient\'s health information for an AI-powered risk estimate.</p></div></div>', unsafe_allow_html=True)

with st.form("prediction_form"):
    left,right=st.columns(2,gap="large")
    with left:
        age=st.text_input("Age *",value="45",placeholder="e.g. 45")
        chest_pain=st.selectbox("Chest Pain Type *",["ATA","NAP","ASY","TA"],format_func=lambda x:{"ATA":"ATA — Atypical Angina","NAP":"NAP — Non-Anginal Pain","ASY":"ASY — Asymptomatic","TA":"TA — Typical Angina"}[x])
        cholesterol=st.text_input("Cholesterol *",value="180",placeholder="e.g. 180 mg/dL")
        resting_ecg=st.selectbox("Resting ECG *",["Normal","ST","LVH"],format_func=lambda x:{"Normal":"Normal","ST":"ST-T Wave Abnormality","LVH":"Left Ventricular Hypertrophy"}[x])
        exercise_angina=st.selectbox("Exercise Angina *",["N","Y"],format_func=lambda x:"No" if x=="N" else "Yes")
    with right:
        sex=st.selectbox("Gender *",["M","F"],format_func=lambda x:"Male" if x=="M" else "Female")
        resting_bp=st.text_input("Resting Blood Pressure *",value="120",placeholder="e.g. 120 mmHg")
        fasting_bs=st.selectbox("Fasting Blood Sugar *",[0,1],format_func=lambda x:"No — ≤ 120 mg/dL" if x==0 else "Yes — > 120 mg/dL")
        max_hr=st.text_input("Maximum Heart Rate *",value="150",placeholder="e.g. 150 bpm")
        oldpeak=st.text_input("Oldpeak *",value="1.0",placeholder="e.g. 1.5")
    st_slope=st.selectbox("ST Slope *",["Up","Flat","Down"])
    st.markdown("<br>",unsafe_allow_html=True)
    submitted=st.form_submit_button("🔍  Analyze Heart Risk")
st.markdown('</div>',unsafe_allow_html=True)

if submitted:
    try:
        numeric_values={
            "Age":int(age),
            "RestingBP":int(resting_bp),
            "Cholesterol":int(cholesterol),
            "MaxHR":int(max_hr),
            "Oldpeak":float(oldpeak),
        }
        limits={"Age":(18,100),"RestingBP":(1,250),"Cholesterol":(1,700),"MaxHR":(1,250),"Oldpeak":(-5,10)}
        for field,(minimum,maximum) in limits.items():
            if not minimum <= numeric_values[field] <= maximum:
                raise ValueError(f"{field} must be between {minimum} and {maximum}.")
        row={c:0 for c in expected_columns}
        for c,v in {**numeric_values,"FastingBS":fasting_bs}.items():
            if c in row: row[c]=v
        for c in [f"Sex_{sex}",f"ChestPainType_{chest_pain}",f"RestingECG_{resting_ecg}",f"ExerciseAngina_{exercise_angina}",f"ST_Slope_{st_slope}"]:
            if c in row: row[c]=1
        input_df=pd.DataFrame([row],columns=expected_columns)
        scaled=scaler.transform(input_df)
        pred=int(model.predict(scaled)[0])
        prob=float(model.predict_proba(scaled)[0][1]) if hasattr(model,"predict_proba") else None
        confidence=(max(prob,1-prob) if prob is not None else None)
        risk_title="HIGHER RISK" if pred==1 else "LOWER RISK"
        risk_class="risk-high" if pred==1 else "risk-low"
        score=int(round((1-prob)*100)) if prob is not None else (35 if pred==0 else 65)
        confidence_text=f"{confidence:.0%}" if confidence is not None else "N/A"
        risk_factors=[]
        recommendations=[]
        if fasting_bs==1:
            risk_factors.append("Elevated fasting blood sugar")
            recommendations.append("Monitor blood sugar and discuss diet with a healthcare professional.")
        if exercise_angina=="Y":
            risk_factors.append("Exercise-induced angina")
            recommendations.append("Avoid strenuous exercise until cleared by your doctor.")
        if chest_pain=="ASY":
            risk_factors.append("Asymptomatic chest pain pattern")
            recommendations.append("Discuss the chest-pain pattern with a healthcare professional.")
        if numeric_values["Cholesterol"]>=240:
            risk_factors.append("High cholesterol reading")
            recommendations.append("Discuss cholesterol management and heart-healthy nutrition with your doctor.")
        if numeric_values["Age"]>=65:
            risk_factors.append("Age-related cardiovascular risk")
            recommendations.append("Arrange regular cardiovascular checkups and review your risk factors with your doctor.")
        if numeric_values["RestingBP"]>=140:
            risk_factors.append("Elevated resting blood pressure")
            recommendations.append("Recheck your blood pressure and discuss blood-pressure management with your doctor.")
        if numeric_values["MaxHR"]<100:
            risk_factors.append("Lower maximum heart-rate reading")
            recommendations.append("Discuss the heart-rate result with a healthcare professional before strenuous activity.")
        if numeric_values["Oldpeak"]>=2:
            risk_factors.append("Elevated exercise-related ECG change")
            recommendations.append("Have the ECG-related finding reviewed by a qualified healthcare professional.")
        if st_slope=="Down":
            risk_factors.append("Downward ST slope pattern")
            recommendations.append("Ask a healthcare professional to review this ECG pattern.")
        if not risk_factors:
            risk_factors.append("No specific questionnaire factor was flagged")
            recommendations.append("Continue routine checkups and maintain heart-healthy habits.")
        factors_html="".join(f"<li>{factor}</li>" for factor in risk_factors)
        recommendations_html="".join(f"<li>{recommendation}</li>" for recommendation in recommendations)
        st.markdown(f'''<div class="result-card">
<h3>Prediction Result</h3><p class="result-subtitle">AI-based Heart Disease Analysis</p>
<div class="metric-grid"><div class="metric"><div class="metric-label">Heart Health Score</div><div class="metric-value">{score}</div><div class="metric-note">out of 100</div></div>
<div class="metric"><div class="metric-label">Model Confidence</div><div class="metric-value">{confidence_text}</div><div class="metric-note">Based on trained KNN model</div></div></div>
<div class="risk-banner {risk_class}">{"⚠️" if pred==1 else "✅"} {risk_title}</div>
<p class="result-copy">Based on the information provided, the trained model estimates a <strong>{"higher" if pred==1 else "lower"} likelihood</strong> of heart disease. This result is for educational use and is not a medical diagnosis.</p>
<div class="result-section-title">Possible Risk Factors</div><ul class="result-list">{factors_html}</ul>
<div class="result-section-title">Recommended Actions</div><ul class="result-list">{recommendations_html}</ul>
<div class="result-list"><strong>Next step:</strong> Discuss this result and your health history with a qualified healthcare professional.</div>
</div>''',unsafe_allow_html=True)
        st.markdown('<p class="disclaimer">For educational/project demonstration only; this is not a medical diagnosis. Consult a qualified healthcare professional for medical advice.</p>',unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Please check the entered values: {e}")
