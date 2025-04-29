import streamlit as st
import numpy as np
import joblib


model = joblib.load("concrete_model.pkl")

st.title("Concrete Compressive Strength Prediction")


cement = st.number_input("Cement (kg/m^3)", min_value=0.0)
slag = st.number_input("Blast Furnace Slag", min_value=0.0)
fly_ash = st.number_input("Fly Ash", min_value=0.0)
water = st.number_input("Water", min_value=0.0)
superplasticizer = st.number_input("Superplasticizer", min_value=0.0)
coarse_agg = st.number_input("Coarse Aggregate", min_value=0.0)
fine_agg = st.number_input("Fine Aggregate", min_value=0.0)
age = st.number_input("Age (days)", min_value=1)

if st.button("Predict Strength"):
    input_data = np.array([[cement, slag, fly_ash, water, superplasticizer, coarse_agg, fine_agg, age]])
    prediction = model.predict(input_data)
    st.success(f"Predicted Compressive Strength: {prediction[0]:.2f} MPa")
