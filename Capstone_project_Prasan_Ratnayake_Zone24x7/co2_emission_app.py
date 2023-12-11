import streamlit as st
import pandas as pd
import pickle
import xgboost as xgb 

# Load the pre-trained model
with open('xgboost_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the encoded mappings (assuming you saved them to a file)
with open('label_encoders.pkl', 'rb') as file:
    label_encoders = pickle.load(file)

def encode_user_inputs(user_inputs, label_encoders):
    encoded_inputs = {
        'Make': None,
        'Model': None,
        'VC': None,
        'EC': None,
        'T': None,
        'FT': None,
        'FC-C': None
    }

    categorical_cols = ['Make', 'Model', 'VC', 'T', 'FT']  # Categorical columns

    for col in categorical_cols:
        label_encoder = label_encoders[col]
        encoded_inputs[col] = label_encoder.transform([user_inputs[categorical_cols.index(col)]])[0]

    # Assign 'EC' and 'FC-C' directly as floats
    encoded_inputs['EC'] = float(user_inputs[len(categorical_cols)])  # Assuming 'EC' is after the categorical columns
    encoded_inputs['FC-C'] = float(user_inputs[len(categorical_cols) + 1])  # Assuming 'FC-C' is after 'EC'
    print(encoded_inputs)
    return encoded_inputs


# Function to predict CO2 emission
def predict_co2_emission(inputs):
    df = pd.DataFrame(inputs, index=[0])  # Create a DataFrame from encoded inputs
    model = ""
    with open('xgboost_model.pkl', 'rb') as file:
        model = pickle.load(file)
    prediction = model.predict(df)  # Assuming 'model' is an XGBoost model
    return prediction[0]

# Streamlit app
st.title('CO2 Emission Prediction')

# Get user inputs as dropdowns and number inputs
make = st.selectbox('Make', ['PORSCHE', 'BMW', 'GMC', 'CHEVROLET', 'TOYOTA'])
model = st.selectbox('Model', ['Panamera 4 Executive', 'M6 CABRIOLET', 'SIERRA', 'SILVERADO', '4Runner 4WD'])
vc = st.selectbox('VC', ['FULL-SIZE', 'SUBCOMPACT', 'PICKUP TRUCK - STANDARD', 'PICKUP TRUCK - STANDARD', 'SUV - STANDARD'])
t = st.selectbox('T', ['AM8', 'AM7', 'A8', 'A6', 'AS5'])
ft = st.selectbox('FT', ['Z', 'Z', 'Z', 'X', 'X'])
ec = st.number_input('EC', value=3.0)
fc_c = st.number_input('FC-C', value=12.4)

# Create a list of user inputs
user_inputs = [make, model, vc, t, ft, ec, fc_c]

# Encode user inputs using label encoders and include numerical columns
encoded_user_inputs = encode_user_inputs(user_inputs, label_encoders)

# Make prediction when user clicks the button
if st.button('Predict CO2 Emission'):
    result = predict_co2_emission(encoded_user_inputs)
    st.success(f'Predicted CO2 Emission: {result} g/km')
