import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Load model + features
# -----------------------------
model = joblib.load("model.pkl")
features = joblib.load("features.pkl")

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Bank Customer Churn Prediction",
    layout="centered"
)

st.title("🏦 Bank Customer Churn Prediction")
st.write("Enter customer details to predict churn probability.")

# -----------------------------
# User Inputs
# -----------------------------
credit_score = st.number_input("Credit Score", 300, 900, 650)

geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
gender = st.selectbox("Gender", ["Male", "Female"])

age = st.slider("Age", 18, 100, 35)
tenure = st.slider("Tenure", 0, 10, 5)

balance = st.number_input("Balance", 0.0, 300000.0, 50000.0)

num_products = st.slider("Number of Products", 1, 4, 1)

has_cr_card = st.selectbox("Has Credit Card", [0, 1])
is_active_member = st.selectbox("Is Active Member", [0, 1])

estimated_salary = st.number_input("Estimated Salary", 0.0, 300000.0, 50000.0)

# -----------------------------
# Encoding (One-Hot Style like training)
# -----------------------------
geo_germany = 1 if geography == "Germany" else 0
geo_spain = 1 if geography == "Spain" else 0
gender_male = 1 if gender == "Male" else 0

# -----------------------------
# Create input DataFrame
# -----------------------------
input_data = pd.DataFrame([{
    "CreditScore": credit_score,
    "Age": age,
    "Tenure": tenure,
    "Balance": balance,
    "NumOfProducts": num_products,
    "HasCrCard": has_cr_card,
    "IsActiveMember": is_active_member,
    "EstimatedSalary": estimated_salary,
    "Geography_Germany": geo_germany,
    "Geography_Spain": geo_spain,
    "Gender_Male": gender_male
}])

# -----------------------------
# Align with training features (IMPORTANT FIX)
# -----------------------------
input_data = input_data.reindex(columns=features, fill_value=0)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Churn"):

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("⚠️ Customer is likely to churn")
    else:
        st.success("✅ Customer is likely to stay")

    st.write(f"Churn Probability: {probability:.2f}")