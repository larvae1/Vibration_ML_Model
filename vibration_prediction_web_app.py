import streamlit as st
import numpy as np
import joblib
import requests

# Function to load model and scaler directly from URLs
def load_model_and_scaler_from_urls(model_url, scaler_url):
    model_response = requests.get(model_url)
    scaler_response = requests.get(scaler_url)
    
    # Load model and scaler from content
    loaded_model = joblib.load(model_response.content)
    scaler = joblib.load(scaler_response.content)
    
    return loaded_model, scaler

# URLs for the model and scaler files
model_url = 'https://github.com/larvae1/Vibration_ML_Model/raw/main/Vibration_Model.sav'
scaler_url = 'https://github.com/larvae1/Vibration_ML_Model/raw/main/scaler.sav'

# Load the model and scaler
loaded_model, scaler = load_model_and_scaler_from_urls(model_url, scaler_url)

# Function for prediction
def vibration_prediction(input_data):
    # Standardize the input data using the loaded scaler
    input_data_reshaped = np.array(input_data).reshape(1, -1)
    std_data = scaler.transform(input_data_reshaped)
    
    # Predict using the loaded model
    prediction = loaded_model.predict(std_data)
    
    if prediction[0] == 0:
        return 'Manmade'
    else:
        return 'Natural'

def main():
    # Set background color to blue
    st.markdown(
        """
        <style>
        body {
            background-color: #00BFFF;
            color: #FFFFFF;
            text-align: center;
        }
        .title {
            color: #0000FF; /* Blue color */
            font-size: 36px;
            margin-bottom: 30px;
        }
        .input-container {
            background-color: #333333; /* Dark white color */
            padding: 20px;
            border-radius: 10px;
            margin-top: 50px;
        }
        .input-box {
            color: #FFFFFF;
            margin-bottom: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Giving a title
    st.markdown("<h1 class='title'>VIBRATION SOURCE PREDICTION</h1>", unsafe_allow_html=True)
    
    # Input container
    with st.container():
        st.markdown("<h2 class='input-box'>Input Parameters</h2>", unsafe_allow_html=True)
        Vibration = st.text_input('Enter Vibration')
        Amplitude = st.text_input('Enter Amplitude')
        Duration = st.text_input('Enter Duration')
        Peak_to_Peak_Vibration = st.text_input('Enter Peak-to-Peak Vibration')
    
    # Creating button for prediction
    if st.button('Predict Vibration Source', key='prediction_button'):
        # Convert input data to float
        try:
            input_data = [float(Vibration), float(Amplitude), float(Duration), float(Peak_to_Peak_Vibration)]
            diagnosis = vibration_prediction(input_data)
            st.success(f"The vibration source is predicted to be: {diagnosis} origin ")
        except ValueError:
            st.error('Please enter valid numerical values for all input fields.')

if __name__ == '__main__':
    main()
