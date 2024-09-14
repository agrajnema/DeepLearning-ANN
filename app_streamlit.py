# Load the libraries
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
import streamlit as st
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder


# Load the trained model
model = load_model('model.keras')

# Load the pickle files

with open('labelEncoder.pkl', 'rb') as file:
    label_encoder = pickle.load(file)

with open('oneHotEncoder.pkl', 'rb') as file:
    one_hot_encoder = pickle.load(file)

with open('standardScaler.pkl', 'rb') as file:
    standard_scaler = pickle.load(file)


# Set page title
st.title('Customer Churn Prediction')


# Take inputs from the user
credit_score = st.number_input('Credit Score')
geography = st.selectbox('Geography', one_hot_encoder.categories_[0])
gender = st.selectbox('Gender',label_encoder.classes_)
age = st.slider('Age',18,90)
tenure = st.slider('Tenure', 0,10)
balance = st.number_input('Balance',min_value=0, step=100)
number_of_products = st.slider('Number of products',1,4)
has_credit_card = st.selectbox('Has Credit Card', [0,1])
is_active_member = st.selectbox('Is Active Member', [0,1])
estimated_salary = st.number_input('Estimated Salary', min_value=1000, step=1000)

input_data = pd.DataFrame({
    'CreditScore' : [credit_score],
    'Gender': [label_encoder.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [number_of_products],
    'HasCrCard': [has_credit_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# Encode geography column
temp_geo = one_hot_encoder.transform([[geography]]).toarray()
temp_geo_df = pd.DataFrame(temp_geo,columns=one_hot_encoder.get_feature_names_out(['Geography']))

input_data_df = pd.concat([input_data.reset_index(drop=True),temp_geo_df], axis=1)

input_data_df_scaled = standard_scaler.transform(input_data_df)

prediction = model.predict(input_data_df_scaled)
prediction_probability = prediction[0][0]

st.write(f'Churn probability: {prediction_probability:.2f}')

if prediction_probability > 0.5:
    st.write('The customer is likely to churn')

else:
    st.write('The customer is not likely to churn')