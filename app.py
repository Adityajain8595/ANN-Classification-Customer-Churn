import streamlit as st
import pickle
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model, scaler, and encoders
model = load_model('model.h5')

with open('onehotencoder_geo.pkl', 'rb') as file:
    onehotencoder_geo = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

with open('labelencoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

# Streamlit app
st.title("Customer Churn Prediction")

# Input fields
st.sidebar.header("Customer Details")
credit_score = st.sidebar.number_input("Credit Score", min_value=300, max_value=850)
geography = st.sidebar.selectbox("Geography", ["France", "Germany", "Spain"])
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
age = st.sidebar.number_input("Age", min_value=18, max_value=100)
tenure = st.sidebar.number_input("Tenure", min_value=0, max_value=10)
balance = st.sidebar.number_input("Balance", min_value=0)
num_of_products = st.sidebar.number_input("Number of Products", min_value=1, max_value=4)
has_cr_card = st.sidebar.selectbox("Has Credit Card", [0, 1])
is_active_member = st.sidebar.selectbox("Is Active Member", [0, 1])
estimated_salary = st.sidebar.number_input("Estimated Salary", min_value=0)

# Create a dictionary from the inputs
input_data = {
    'CreditScore': credit_score,
    'Geography': geography,
    'Gender': gender,
    'Age': age,
    'Tenure': tenure,
    'Balance': balance,
    'NumOfProducts': num_of_products,
    'HasCrCard': has_cr_card,
    'IsActiveMember': is_active_member,
    'EstimatedSalary': estimated_salary
}

# Convert input data to DataFrame
input_df = pd.DataFrame([input_data])

# One-hot encode 'Geography'
geo_encoded = onehotencoder_geo.transform([[input_data['Geography']]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehotencoder_geo.get_feature_names_out())

# Concatenate the encoded data with the original DataFrame
input_df = pd.concat([input_df.drop(columns='Geography').reset_index(drop=True), geo_encoded_df.reset_index(drop=True)], axis=1)

# Label encode 'Gender'
input_df['Gender'] = label_encoder_gender.transform([input_df['Gender']])

# Convert floats to int
input_df['Geography_France'] = input_df['Geography_France'].astype(int)
input_df['Geography_Germany'] = input_df['Geography_Germany'].astype(int)
input_df['Geography_Spain'] = input_df['Geography_Spain'].astype(int)

# Standardize the input data
input_df = scaler.transform(input_df)

# Predict churn
if st.sidebar.button("Predict"):
    pred = model.predict(input_df)
    churn_probability = pred[0][0]
    
    st.write(f"Churn Probability: {churn_probability:.4f}")
    
    if churn_probability > 0.5:
        st.success("The customer is likely to churn.")
    else:
        st.error("The customer is not likely to churn.")