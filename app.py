import joblib
import pandas as pd
import streamlit as st

xgb_model = joblib.load('xgb_model.joblib')


def predict_cvd_risk(model, age, gender, height_cm, weight_kg,
                     ap_hi, ap_lo, cholesterol, gluc,
                     smoke, alco, active):

    bmi = weight_kg / ((height_cm / 100) ** 2)
    map_value = ((2 * ap_lo) + ap_hi) / 3

    data = pd.DataFrame([{
        'age': age,
        'BMI': bmi,
        'MAP': map_value,
        'cholesterol': cholesterol,
        'gluc': gluc,
        'smoke': smoke,
        'alco': alco,
        'active': active,
        'gender': gender,
    }])

    prob = model.predict_proba(data)[:, 1][0]
    return prob, bmi, map_value


# risk = predict_cvd_risk(model=xgb_model, age=55, gender=1,
#                         height_cm=165, weight_kg=80,
#                         ap_hi=150, ap_lo=95,
#                         cholesterol=3, gluc=2,
#                         smoke=1, alco=0, active=0)
#
# print(f"Predicted CVD risk: {risk[0]:.2%}")

st.set_page_config(
    page_title='Cardiovascular Risk Calculator',
    page_icon='🫀',
    layout='centered'
)

st.image("Heart Disease and Heart Problem.jpeg")

st.title("🫀 Cardiovascular Disease Risk Predictor")
st.markdown(
    """
    Estimate the probability of cardiovascular disease using clinical and 
    lifestyle indicators. This demo is powered by your trained **XGBoost** model.
    """
)

st.markdown("---")

st.sidebar.header("Patient Inputs")

age = st.sidebar.slider(
    "Age (years)",
    min_value=30,
    max_value=65,
    value=50,
    step=1
)

gender_label = st.sidebar.radio('Gender', ['Female', 'Male'])
if gender_label == 'Female':
    gender = 1
else:
    gender = 2

height_cm = st.sidebar.slider(
    "Height (cm)",
    min_value=140,
    max_value=210,
    value=168,
    step=1
)

weight_kg = st.sidebar.slider(
    "Weight (kg)",
    min_value=40,
    max_value=150,
    value=70,
    step=1
)

ap_hi = st.sidebar.slider(
    "Systolic Blood Pressure(ap_hi)",
    min_value=90,
    max_value=220,
    value=130,
    step=1
)

ap_lo = st.sidebar.slider(
    "Diastolic Blood Pressure(ap_lo)",
    min_value=50,
    max_value=130,
    value=80,
    step=1
)

chol_label = st.sidebar.selectbox(
    "Cholesterol level",
    [
        "1 - Normal",
        "2 - Above normal",
        "3 - Well above normal",
    ],
)
if chol_label == "1 - Normal":
    cholesterol = 1
elif chol_label == "2 - Above normal":
    cholesterol = 2
else:
    cholesterol = 3

gluc_label = st.sidebar.selectbox(
    "Glucose level",
    [
        "1 - Normal",
        "2 - Above normal",
        "3 - Well above normal",
    ],
)
if gluc_label == "1 - Normal":
    gluc = 1
elif gluc_label == "2 - Above normal":
    gluc = 2
else:
    gluc = 3

smoke = int(st.sidebar.checkbox("Smoker", value=False))
alco = int(st.sidebar.checkbox("Regular alcohol intake", value=False))
active = int(st.sidebar.checkbox("Physically active (several times/week)", value=True))

st.sidebar.markdown("---")
predict_button = st.sidebar.button("🔮 Predict risk")

st.sidebar.caption(
    "Note: This tool is for educational purposes and does **not** replace with"
    "professional medical advice."
)
tab_pred, tab_about = st.tabs(["📊 Prediction", "ℹ️ About this app"])

with tab_pred:
    if predict_button:
        prob, bmi, map_value = predict_cvd_risk(
            xgb_model,
            age,
            gender,
            height_cm,
            weight_kg,
            ap_hi,
            ap_lo,
            cholesterol,
            gluc,
            smoke,
            alco,
            active
        )

        if prob < 0.25:
            risk_label = "Low"
            risk_color = "🟢"
        elif prob < 0.5:
            risk_label = "Moderate"
            risk_color = "🟡"
        else:
            risk_label = "High"
            risk_color = "🔴"

        st.subheader("Result")
        st.metric(label="Predicted CVD risk", value=f"{prob * 100:.1f}%")

        st.markdown(f"**Risk category:** {risk_color} **{risk_label}**")

        c1, c2 = st.columns(2)
        c1.metric("BMI(Normal range: 18.5-24.9)", f"{bmi:.1f}")
        c2.metric("MAP(Normal range: 70-100mmHg)", f"{map_value:.0f} mmHg")

        st.info(
            "This tool is for educational purposes only and does **not** replace "
            "professional medical advice, diagnosis, or treatment."
        )
    else:
        st.caption("Adjust the inputs and click **Predict CVD Risk** to see the model output.")

with tab_about:
    st.subheader("How this works")
    st.write(
        """
        - The model is trained on the **Kaggle Cardiovascular Disease** dataset  
        - Inputs used include: age, gender, BMI, mean arterial pressure (MAP),  
          cholesterol & glucose levels, smoking status, alcohol intake, and physical activity.  
        - The backend uses a tuned **XGBoost** classifier and outputs the estimated probability
          that the patient has cardiovascular disease (`cardio = 1` in the dataset).
        """
    )
    st.write(
        """
        This demo is purely for learning and should not be used for real clinical decisions.
        """
    )
