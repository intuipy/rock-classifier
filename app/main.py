# app/main.py
from fastapi import FastAPI
from pydantic import BaseModel
from app.predict import predict_rock_class

app = FastAPI(
    title='Rock Classification API',
    description='Enter 4 well log features to get rock class 1-10',
    version='1.0.0'
)

class RockFeatures(BaseModel):
    GR:   float
    RES:  float
    PHIE: float
    RHOB: float

@app.get('/')
def root():
    return {'message': 'Rock Classification API is running'}

@app.post('/predict')
def predict(features: RockFeatures):
    rock_class = predict_rock_class(
        features.GR, features.RES, features.PHIE, features.RHOB
    )
    return {'rock_class': rock_class,
            'description': f'Predicted rock class: {rock_class}'}
