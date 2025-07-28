from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import joblib
import os
from keras.models import load_model

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.abspath(os.path.join(BASE_DIR, "../models"))

rf_model = joblib.load(os.path.join(MODELS_DIR, "random_forest_model.pkl"))
logreg_model = joblib.load(os.path.join(MODELS_DIR, "logistic_regression_model.pkl"))
nn_model = load_model(os.path.join(MODELS_DIR, "nn_model.h5"))

scaler_full = joblib.load(os.path.join(MODELS_DIR, "scaler_full.pkl"))
scaler_logreg = joblib.load(os.path.join(MODELS_DIR, "scaler_logreg.pkl"))

class FullInput(BaseModel):
    radius_mean: float
    texture_mean: float
    perimeter_mean: float
    area_mean: float
    smoothness_mean: float
    compactness_mean: float
    concavity_mean: float
    concave_points_mean: float
    symmetry_mean: float
    fractal_dimension_mean: float
    radius_se: float
    texture_se: float
    perimeter_se: float
    area_se: float
    smoothness_se: float
    compactness_se: float
    concavity_se: float
    concave_points_se: float
    symmetry_se: float
    fractal_dimension_se: float
    radius_worst: float
    texture_worst: float
    perimeter_worst: float
    area_worst: float
    smoothness_worst: float
    compactness_worst: float
    concavity_worst: float
    concave_points_worst: float
    symmetry_worst: float
    fractal_dimension_worst: float

class ReducedInput(BaseModel):
    radius_mean: float
    texture_mean: float
    compactness_worst: float
    concave_points_worst: float
    area_worst: float

def get_label(prediction: int) -> str:
    return "Malignant" if prediction == 1 else "Benign"

@app.get("/")
def home():
    return {"message": "Breast Cancer Diagnosis API is running."}

@app.post("/predict/randomforest")
def predict_rf(input: FullInput):
    data = np.array([[value for value in input.dict().values()]])
    scaled = scaler_full.transform(data)
    prediction = rf_model.predict(scaled)[0]
    return {
        "model": "random_forest",
        "prediction": int(prediction),
        "label": get_label(prediction)
    }

@app.post("/predict/logistic")
def predict_logreg(input: ReducedInput):
    data = np.array([[value for value in input.dict().values()]])
    scaled = scaler_logreg.transform(data)
    prediction = logreg_model.predict(scaled)[0]
    return {
        "model": "logistic_regression",
        "prediction": int(prediction),
        "label": get_label(prediction)
    }

@app.post("/predict/neuralnet")
def predict_nn(input: FullInput):
    data = np.array([[value for value in input.dict().values()]])
    scaled = scaler_full.transform(data)
    prob = nn_model.predict(scaled)[0][0] # type: ignore
    label = int(prob >= 0.5)
    return {
        "model": "neural_network",
        "prediction": label,
        "label": get_label(label),
        "malignancy_probability": round(float(prob), 4)
    }
