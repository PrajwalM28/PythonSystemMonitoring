import pickle
import os

def load_scaler():
    try:
        # Check if the file exists
        model_path = 'models/scaler.pkl'
        if not os.path.exists(model_path):
            print(f"Scaler model file not found at {model_path}.")
            return None
        
        with open(model_path, 'rb') as f:
            scaler = pickle.load(f)
            return scaler
    except FileNotFoundError:
        print(f"Scaler model file not found at {model_path}.")
        return None
    except Exception as e:
        print(f"Error while loading scaler: {e}")
        return None

import joblib

def get_anomaly_detection_model():
    try:
        model_path = 'models/anomaly_detection.pkl'
        model = joblib.load(model_path)  # Using joblib instead of pickle
        print("Anomaly detection model loaded successfully.")
        return model
    except FileNotFoundError:
        print(f"Anomaly detection model file not found at {model_path}.")
        return None
    except Exception as e:
        print(f"Error while loading anomaly detection model: {e}")
        return None
