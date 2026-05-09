import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Load trained model
# -----------------------------
model = joblib.load("model.pkl")

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Insurance Claim Prediction",
    layout="centered"
)

st.title(" Insurance Claim Prediction")
st.write("Enter patient details to predict insurance charges.")

# -----------------------------
# User Inputs
# -----------------------------
age = st.slider("Age", 18, 100, 30)

sex = st.selectbox("Sex", ["male", "female"])
bmi = st.number_input("BMI", 10.0, 50.0, 25.0)

children = st.slider("Children", 0, 5, 0)

smoker = st.selectbox("Smoker", ["yes", "no"])

region = st.selectbox(
    "Region",
    ["northeast", "northwest", "southeast", "southwest"]
)

# -----------------------------
# Encoding categorical features (Matching the 11-column training set)
# -----------------------------
# Sex encoding (often results in 2 columns: female, male)
sex_female = 1 if sex == "female" else 0
sex_male = 1 if sex == "male" else 0

# Smoker encoding (often results in 2 columns: no, yes)
smoker_no = 1 if smoker == "no" else 0
smoker_yes = 1 if smoker == "yes" else 0

# Region encoding (often results in 4 columns: northeast, northwest, southeast, southwest)
reg_ne = 1 if region == "northeast" else 0
reg_nw = 1 if region == "northwest" else 0
reg_se = 1 if region == "southeast" else 0
reg_sw = 1 if region == "southwest" else 0

# -----------------------------
# Create input DataFrame (Total: 11 Features)
# -----------------------------
# IMPORTANT: These must be in the EXACT order they were during model.fit()
input_data = pd.DataFrame([{
    "age": age,
    "bmi": bmi,
    "children": children,
    "sex_female": sex_female,
    "sex_male": sex_male,
    "smoker_no": smoker_no,
    "smoker_yes": smoker_yes,
    "region_northeast": reg_ne,
    "region_northwest": reg_nw,
    "region_southeast": reg_se,
    "region_southwest": reg_sw
}])

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Insurance Charges"):
    # Optional: ensure columns are in the right order if you're unsure
    # input_data = input_data[['age', 'bmi', 'children', ...]] 
    
    prediction = model.predict(input_data)[0]
    st.subheader("Prediction Result")
    st.success(f"Estimated Insurance Charge: ${prediction:,.2f}")