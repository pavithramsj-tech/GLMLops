import os
import streamlit as st
import pandas as pd
import joblib

# Load the model committed by the pipeline (sits next to this file)
model_path = os.path.join(os.path.dirname(__file__), "tourism_project_v1.joblib")
model = joblib.load(model_path)

st.title("Tourism Prediction App")
st.write(""" Leading travel company, is revolutionizing the tourism 
         industry by leveraging data-driven strategies".
""")

age = st.number_input("Age", 18, 100, 30)
typeofcontact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
citytier = st.selectbox("City Tier", [1, 2, 3])
occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Others"])
gender = st.selectbox("Gender", ["Male", "Female"])
productpitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
preferredstar = st.selectbox("Preferred Property Star", [3, 4, 5])
maritalstatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Unmarried"])
designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
monthlyincome = st.number_input("Monthly Income", 0, 1000000, 50000)

input_data = pd.DataFrame([{
    "Age": age,
    "TypeofContact": typeofcontact,
    "CityTier": citytier,
    "Occupation": occupation,
    "Gender": gender,
    "ProductPitched": productpitched,
    "PreferredPropertyStar": preferredstar,
    "MaritalStatus": maritalstatus,
    "Designation": designation,
    "MonthlyIncome": monthlyincome
}])

if st.button("Predict Failure"):
    prediction = model.predict(input_data)[0]
    result = "ProdTaken" if prediction == 1 else "No ProdTaken"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
