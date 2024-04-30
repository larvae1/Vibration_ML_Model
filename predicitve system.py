import numpy as np
import pickle

class CustomStandardScaler:
    def __init__(self, feature_names):
        self.feature_names = feature_names
        self.scaler = None

    def fit(self, X):
        self.scaler = StandardScaler().fit(X)
        return self

    def transform(self, X):
        return self.scaler.transform(X)

# Load the model
loaded_model = pickle.load(open('C:\\Users\\Admin\\Desktop\\ML MODEL\\Vibration_Model (2).sav', 'rb'))

# Load the scaler
scaler = pickle.load(open(r"C:\Users\Admin\Desktop\ML MODEL\scaler.sav", 'rb'))


# Define feature names
feature_names = ['Vibration', 'Amplitude', 'Duration', 'Peak-to-Peak Vibration']

# Making a prediction System
input_data = np.array([0.638793, 2.545350, 2.659317, 1.263290])  # Ensure input_data is a numpy array

# Reshape the input data
input_data_reshaped = input_data.reshape(1, -1)

# Standardize the input data using the loaded scaler
std_data = scaler.transform(input_data_reshaped)
print(std_data)

# Predict using the loaded model
prediction = loaded_model.predict(std_data)
print(prediction)

if prediction[0] == 0:
    print('Manmade')
else:
    print('Natural')
