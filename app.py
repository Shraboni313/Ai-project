import streamlit as st
import numpy as np
import joblib

model = joblib.load('diabetes_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("Diabetes Risk Prediction App")
st.write("Enter patient physical parameters to predict the risk of diabetes.")

col1,col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("Pregnancies",min_value=0,max_value=20,value=1)
    glucose = st.number_input("Glucose level",min_value=0,max_value=300,value=120)
    blood_pressure = st.number_input("Blood Pressure",min_value=0,max_value=150,value=70)
    skin_thickness = st.number_input("Skin Thickness",min_value=0,max_value=100,value=20)

with col2:
    insulin = st.number_input("Insulin Level",min_value=0,max_value=900,value=79)
    bmi = st.number_input("BMI",min_value=0.0,max_value=70.0,value=25.0)
    dpf = st.number_input("Diabetes Pedigree Function",min_value=0.0,max_value=3.0,value=0.5)
    age = st.number_input("Age",min_value=1,max_value=120,value=33)

if st.button("Predict Diabetes Risk"):
    input_data = np.array([[pregnancies,glucose,blood_pressure,skin_thickness,insulin,bmi,dpf,age]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)[0][1] * 100
    st.subheader("--- Results ---")
    st.write(f"**Diabetes Risk:**{probability:.2f}%")

    if prediction[0] == 1:
        st.error("High risk of diabetes detected based on the provided inputs. Please consult a physician.")
    else:
        st.success("Low risk of diabetes detected. Results appear normal.")
