import os
import numpy as np
import pickle
import streamlit as st

# Get the directory of the current script
current_dir = os.path.dirname(__file__)

# Define paths to your model and scaler files
MODEL_PATH = os.path.join(current_dir, "Vibration_Model.sav")
SCALER_PATH = os.path.join(current_dir, "scaler.sav")

# Load the model
with open(MODEL_PATH, 'rb') as model_file:
    loaded_model = pickle.load(model_file)

# Load the scaler
with open(SCALER_PATH, 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

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
        </style>
        """,
        unsafe_allow_html=True
    )

    # Giving a title
    st.markdown("<h1 class='title'>VIBRATION SOURCE PREDICTION</h1>", unsafe_allow_html=True)
    
    # Getting input data from user
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
