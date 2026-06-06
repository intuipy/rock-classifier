# app/predict.py
import joblib
import numpy as np

# Load model once when the app starts
model = joblib.load('model/rf_rock_classifier.pkl')

def predict_rock_class(gr: float, res: float,
                        phie: float, rhob: float) -> int:
    features = np.array([[gr, res, phie, rhob]])
    prediction = model.predict(features)
    return int(prediction[0])

