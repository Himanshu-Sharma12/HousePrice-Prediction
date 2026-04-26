import pickle
import pandas as pd
import numpy as np
import streamlit as st

# load model
model = pickle.load(open('linear_model.pkl', 'rb'))

# load dataset for scaling
data = pd.read_csv('house.csv')

# scaling
x_scaler = StandardScaler()
y_scaler = StandardScaler()

x_scaler.fit(data[['Square_Footage', 'Num_Bedrooms', 'Num_Bathrooms', 'Year_Built', 'Lot_Size', 'Garage_Size', 'Neighborhood_Quality']])
y_scaler.fit(data[['House_Price']])

# give title
st.title("House price prediction app")

square_footage = st.number_input('Square Footage', min_value=200, max_value=10000, value=1500)
num_bedrooms = st.number_input('Number of Bedrooms', min_value=1, max_value=10, value=3)
num_bathrooms = st.number_input('Number of Bathrooms', min_value=1, max_value=10, value=2)
year_built = st.number_input('Year Built', min_value=1900, max_value=2035, value=2000)
lot_size = st.number_input('Lot Size', min_value=0.1, max_value=20.0, value=2.0)
garage_size = st.number_input('Garage Size', min_value=0, max_value=5, value=1)
neighborhood_quality = st.number_input('Neighborhood Quality', min_value=1, max_value=10, value=5)

# dataframe
input_features = pd.DataFrame({
    'Square_Footage': [square_footage],
    'Num_Bedrooms': [num_bedrooms],
    'Num_Bathrooms': [num_bathrooms],
    'Year_Built': [year_built],
    'Lot_Size': [lot_size],
    'Garage_Size': [garage_size],
    'Neighborhood_Quality': [neighborhood_quality]
})

input_features_scaled = x_scaler.transform(input_features)

# predictions
if st.button('Predict'):
    prediction_scaled = model.predict(input_features_scaled)
    output = y_scaler.inverse_transform(prediction_scaled.reshape(-1, 1))[0][0]
    st.success(f"House Price Prediction: ${round(output, 2)}")
